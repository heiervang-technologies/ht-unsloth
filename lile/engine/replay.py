"""T4.1 — Idle replay scheduler.

When the controller's compute queue is idle and no live traffic is arriving,
we'd like to keep the GPU warm by replaying past feedback. This is the
"replay/re-weight" tier from the plan; the *hard* part is the reweighting
policy (per ``STATUS.md`` "Documented scope cuts"). We ship a defensible v0:

* **Source:** the on-disk trajectory log. We iterate ``kind == "feedback"``
  records via :meth:`lile.trajectory.TrajectoryLog.iter_from`. The records
  carry the full ``prompt``/``response``/``critique``/``better_response``
  payload (added in ``Controller.submit_feedback`` — see commit history)
  so we can rebuild a :class:`lile.objectives.Batch` without consulting the
  in-memory ``_response_cache`` (which doesn't survive restart).
* **Pacing:** before each enqueue we call
  :meth:`lile.queue.ComputeQueue.is_idle_for` with the configured threshold.
  If a live ``/v1/chat`` or ``/v1/feedback`` lands while we're waiting, the
  threshold resets and we back off until idle again.
* **Per-record cap:** each replay candidate is identified by its trajectory
  byte offset; we maintain an in-memory dict
  ``{offset: replays_done}`` and skip records that have hit ``max_replays``.
  This is the in-process bound; ``replay`` records we ourselves write to the
  log are *not* themselves replayable (we filter ``feedback_kind`` and the
  presence of a payload).
* **Reweighting:** uniform sample over eligible records, downweighted by
  recency: ``w = exp(-age_h / half_life_h)``. We pass that through to the
  Batch via the ``weight_scale`` argument on
  :meth:`lile.controller.Controller.feedback_to_batch`, so a 24h-old record
  trains at half-strength vs a fresh one. This is **policy v0** — the plan
  flags reweighting as the hard part, and we treat that as future work.

We deliberately do not block on the commit token — if the train step fails
(e.g. CCPD τ-skip returns zero loss), we just move on. The trajectory log
records both the enqueue and the completion so the audit trail is intact.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lile.trajectory import Record, TrajectoryLog

if TYPE_CHECKING:
    from lile.controller import Controller


_LOG = logging.getLogger("lile.replay")


@dataclass
class ReplayPolicy:
    """Tunables for the v0 reweighting / pacing policy."""

    idle_threshold_s: float = 30.0
    """Queue must be empty AND no enqueue within this many seconds before we
    consider replaying. Default 30 s — long enough that conversational pauses
    don't trigger replay, short enough to make use of a quiet hour."""

    poll_interval_s: float = 2.0
    """How often the scheduler thread wakes to check the idle gate."""

    max_replays_per_record: int = 3
    """Hard cap on how many times a single record can be replayed in a
    process lifetime. Prevents a single feedback from dominating training if
    the box stays idle for hours."""

    recency_half_life_h: float = 24.0
    """Sample weight halves every ``half_life_h``. Older records still get
    replayed (they just train at lower weight); we never drop them."""

    rng_seed: int | None = None
    """Optional deterministic seed for tests."""


@dataclass
class _Candidate:
    """Eligible replay candidate; pulled from the trajectory log."""

    offset: int
    age_s: float
    payload: dict[str, Any]


@dataclass
class _Stats:
    """Cheap in-memory counters for tests + introspection."""

    replays_enqueued: int = 0
    skipped_no_candidates: int = 0
    skipped_busy: int = 0
    last_replay_at: float = 0.0
    per_record: dict[int, int] = field(default_factory=dict)


class IdleReplayScheduler:
    """Background thread that replays past feedback during idle periods.

    Lifecycle:
        scheduler = IdleReplayScheduler(controller)
        scheduler.start()  # spawns daemon thread
        ...
        scheduler.stop()   # join with timeout

    Thread safety: the scheduler interacts with the controller only through
    the public ``submit_train`` path, which is itself queue-mediated. We hold
    no controller-internal locks.
    """

    def __init__(
        self,
        controller: "Controller",
        policy: ReplayPolicy | None = None,
    ) -> None:
        self.controller = controller
        self.policy = policy or ReplayPolicy()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = _Stats()
        self._rng = random.Random(self.policy.rng_seed)

    # --- public surface --------------------------------------------------

    def start(self) -> "IdleReplayScheduler":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="lile-replay", daemon=True
        )
        self._thread.start()
        if self.controller.trajectory is not None:
            self.controller.trajectory.append(
                "info", event="replay-scheduler-start",
                policy=self._policy_dict(),
            )
        return self

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None
        if self.controller.trajectory is not None:
            self.controller.trajectory.append(
                "info", event="replay-scheduler-stop",
                stats=self._stats_dict(),
            )

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats_dict()

    # --- internal --------------------------------------------------------

    def _policy_dict(self) -> dict[str, Any]:
        return {
            "idle_threshold_s": self.policy.idle_threshold_s,
            "poll_interval_s": self.policy.poll_interval_s,
            "max_replays_per_record": self.policy.max_replays_per_record,
            "recency_half_life_h": self.policy.recency_half_life_h,
        }

    def _stats_dict(self) -> dict[str, Any]:
        return {
            "replays_enqueued": self._stats.replays_enqueued,
            "skipped_no_candidates": self._stats.skipped_no_candidates,
            "skipped_busy": self._stats.skipped_busy,
            "last_replay_at": self._stats.last_replay_at,
            "distinct_records_replayed": len(self._stats.per_record),
        }

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.policy.poll_interval_s):
                return
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                # Never let the scheduler thread die; it logs and continues.
                _LOG.exception("replay tick failed")

    def _tick(self) -> None:
        q = self.controller.queue
        if q is None:
            return
        if not q.is_idle_for(self.policy.idle_threshold_s):
            self._stats.skipped_busy += 1
            return
        candidate = self._pick_candidate()
        if candidate is None:
            self._stats.skipped_no_candidates += 1
            return
        self._enqueue(candidate)

    def _trajectory_path(self) -> Path | None:
        traj = self.controller.trajectory
        if traj is None:
            return None
        return traj.path

    def _pick_candidate(self) -> _Candidate | None:
        path = self._trajectory_path()
        if path is None:
            return None
        now_ns = time.time_ns()
        eligible: list[tuple[float, _Candidate]] = []
        # Iterate the whole log per pick. Acceptable: feedback records are
        # rare relative to chat traffic, and this thread runs only during
        # idle periods. Optimization (sliding offset cursor) is straightforward
        # if profiling demands it.
        for rec in TrajectoryLog.iter_from(path, kinds={"feedback"}):
            if not _is_replayable(rec):
                continue
            done = self._stats.per_record.get(rec.offset, 0)
            if done >= self.policy.max_replays_per_record:
                continue
            age_s = max(0.0, (now_ns - rec.ts) / 1e9)
            weight = _recency_weight(age_s, self.policy.recency_half_life_h)
            eligible.append((weight, _Candidate(rec.offset, age_s, rec.payload)))
        if not eligible:
            return None
        return _weighted_choice(self._rng, eligible)

    def _enqueue(self, c: _Candidate) -> None:
        weight_scale = _recency_weight(c.age_s, self.policy.recency_half_life_h)
        kind = c.payload.get("feedback_kind")
        if kind is None:
            return
        try:
            batch = self.controller.feedback_to_batch(
                kind,
                prompt=c.payload.get("prompt", ""),
                bad_response=c.payload.get("response", ""),
                critique=c.payload.get("critique"),
                better_response=c.payload.get("better_response"),
                value=c.payload.get("value"),
                weight_scale=weight_scale,
            )
        except (ValueError, KeyError) as e:
            _LOG.warning("replay skipped record offset=%s: %s", c.offset, e)
            # Mark as exhausted so we don't re-try this broken record on every tick.
            self._stats.per_record[c.offset] = self.policy.max_replays_per_record
            return
        token = self.controller.submit_train(batch)
        self._stats.replays_enqueued += 1
        self._stats.per_record[c.offset] = self._stats.per_record.get(c.offset, 0) + 1
        self._stats.last_replay_at = time.time()
        if self.controller.trajectory is not None:
            self.controller.trajectory.append(
                "info", event="replay-enqueued",
                source_offset=c.offset, age_s=round(c.age_s, 1),
                weight_scale=round(weight_scale, 4),
                feedback_kind=kind, seq=token.seq,
            )


# --- helpers ---------------------------------------------------------------


def _is_replayable(rec: Record) -> bool:
    """Filter feedback records to ones we can rebuild a Batch for.

    We require the text payload added in the W3 controller change. Records
    written by older code paths (no ``prompt``/``response`` fields) are
    skipped — there's no way to reconstruct training input from them, and
    failing late inside ``feedback_to_batch`` would just spam the log.
    """
    p = rec.payload
    if "feedback_kind" not in p:
        return False
    if not p.get("prompt"):
        return False
    if not p.get("response"):
        return False
    return True


def _recency_weight(age_s: float, half_life_h: float) -> float:
    if half_life_h <= 0:
        return 1.0
    age_h = age_s / 3600.0
    return math.exp(-math.log(2.0) * age_h / half_life_h)


def _weighted_choice(rng: random.Random, candidates: list[tuple[float, _Candidate]]) -> _Candidate:
    total = sum(w for w, _ in candidates)
    if total <= 0:
        return candidates[rng.randrange(len(candidates))][1]
    pick = rng.uniform(0.0, total)
    acc = 0.0
    for w, c in candidates:
        acc += w
        if pick <= acc:
            return c
    return candidates[-1][1]
