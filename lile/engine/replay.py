"""Idle replay scheduler — T4.1 in the plan.

Goal: when the compute queue has been idle for a while, go find old feedback
records and re-train on them. Turns bursty operator feedback into a steady
low-priority background signal without competing with live train/infer.

Design (asyncio-native; matches the rest of this codebase):

  * A single long-lived task started by ``Controller.start()`` and cancelled by
    ``Controller.stop()``. No threads — the controller and queue are both
    asyncio, so a co-located task is the simplest thing that works.

  * Polls ``ComputeQueue.is_idle_for(threshold)`` at ``poll_interval_s``.
    When idle: pick one feedback record from the trajectory log, rebuild a
    train spec via ``Controller.feedback_to_batch`` (pure function — no live
    response index needed), and enqueue it as a normal train task.

  * Picks records via weighted choice with recency bias:
    ``w = 2 ^ (-age_hours / half_life_hours)``. Records approach equal weight
    as the log gets old, but recent feedback remains more likely — this is
    the v0 bandit we'd want to tune if replay ever drives real quality.

  * Per-record replay cap (``max_replays_per_record``): prevents a small
    feedback corpus from dominating if the daemon sits idle for a long time.
    Tracked in-memory, keyed by the record's byte offset in the log (the
    stable identifier ``TrajectoryLog.iter_with_offsets`` exposes).

Scope (what this is NOT):

  * Not a full replay buffer with priority / CoH weighting — that belongs in
    T4.2 self-distillation.
  * Not a distributed scheduler — single-process, shares the controller's event
    loop.
  * Not a "resume from last offset" mechanism — on restart the cap map resets,
    so each record can be replayed up to the cap again. Acceptable for v0;
    the trajectory log itself is the durable record.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..controller import Controller

log = logging.getLogger(__name__)


@dataclass
class ReplayPolicy:
    idle_threshold_s: float = 30.0     # queue must be empty for this long
    poll_interval_s: float = 2.0       # how often to check idleness
    max_replays_per_record: int = 3    # per-record lifetime cap (this process)
    recency_half_life_h: float = 24.0  # weight halves every N hours of age
    min_feedback_records: int = 3      # don't replay if corpus too small

    @classmethod
    def from_config(cls, cfg: Any) -> "ReplayPolicy":
        """Construct from a ServeConfig, using cfg.* where available."""
        return cls(
            idle_threshold_s=getattr(cfg, "idle_replay_threshold_s", 30.0),
            poll_interval_s=getattr(cfg, "replay_poll_interval_s", 2.0),
            max_replays_per_record=getattr(cfg, "replay_max_per_record", 3),
            recency_half_life_h=getattr(cfg, "replay_half_life_h", 24.0),
            min_feedback_records=getattr(cfg, "replay_min_records", 3),
        )


class IdleReplayScheduler:
    """Async scheduler that reinjects trajectory-logged feedback when idle."""

    def __init__(self, controller: "Controller", policy: ReplayPolicy) -> None:
        self.controller = controller
        self.policy = policy
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # offset -> replay count. Bounded by the number of feedback records
        # in the log; tens of thousands max in practice. Never pruned — we
        # want "this record has been replayed to death" to stay sticky.
        self._replayed: dict[int, int] = {}
        # Stats for /status observability; consulted by tests too.
        self.stats = {
            "idle_checks": 0,
            "replays_enqueued": 0,
            "replays_skipped_empty": 0,
            "replays_skipped_capped": 0,
            "last_replay_offset": -1,
        }

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("scheduler already started")
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="idle-replay")
        log.info("idle replay scheduler started (policy=%s)", self.policy)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self.policy.poll_interval_s * 2 + 1)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    # ------------------------------------------------------------------ main loop
    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.policy.poll_interval_s,
                    )
                    # stop was set while we were sleeping.
                    return
                except asyncio.TimeoutError:
                    pass

                self.stats["idle_checks"] += 1
                if not self.controller.queue.is_idle_for(self.policy.idle_threshold_s):
                    continue
                await self._maybe_replay_one()
        except asyncio.CancelledError:
            log.info("idle replay scheduler cancelled")
            raise
        except Exception:
            log.exception("idle replay scheduler crashed")
            raise

    # ------------------------------------------------------------------ core step
    async def _maybe_replay_one(self) -> None:
        choice = self._pick_record()
        if choice is None:
            self.stats["replays_skipped_empty"] += 1
            return
        offset, record = choice
        spec = self.controller.feedback_to_batch(record)
        if spec is None:
            # Record is un-replayable (missing fields). Burn a replay count
            # so we don't keep trying the same broken record.
            self._replayed[offset] = self.policy.max_replays_per_record
            self.stats["replays_skipped_capped"] += 1
            return
        # Tag the spec so trajectory train_step records are distinguishable
        # from live training (nice for offline analysis).
        spec = dict(spec)
        spec["_replay"] = True
        spec["_replay_offset"] = offset

        # Submit on the queue. We deliberately do NOT wait for completion —
        # the scheduler just injects work; whether it commits before the
        # next live train step is irrelevant because all tasks are serialized.
        try:
            await self.controller.submit_train(spec)
            self._replayed[offset] = self._replayed.get(offset, 0) + 1
            self.stats["replays_enqueued"] += 1
            self.stats["last_replay_offset"] = offset
            log.debug("replayed feedback at offset=%d (count=%d)",
                      offset, self._replayed[offset])
        except Exception:
            log.exception("replay submit failed for offset=%d", offset)

    # ------------------------------------------------------------------ selection
    def _pick_record(self) -> tuple[int, dict[str, Any]] | None:
        """Weighted random pick over un-capped feedback records."""
        now = time.time()
        offsets: list[int] = []
        records: list[dict[str, Any]] = []
        weights: list[float] = []
        hl_seconds = self.policy.recency_half_life_h * 3600.0
        cap = self.policy.max_replays_per_record

        for offset, rec in self.controller.trajectory.iter_with_offsets(
            kinds={"feedback"},
        ):
            if self._replayed.get(offset, 0) >= cap:
                continue
            # Decay: w = 2 ^ (-age / half_life). Newer = heavier.
            ts = float(rec.get("ts", now))
            age_s = max(0.0, now - ts)
            w = math.pow(2.0, -age_s / hl_seconds) if hl_seconds > 0 else 1.0
            if w <= 0.0:
                continue
            offsets.append(offset)
            records.append(rec)
            weights.append(w)

        if len(offsets) < self.policy.min_feedback_records:
            return None

        # random.choices does weighted sampling without installing scipy.
        idx = random.choices(range(len(offsets)), weights=weights, k=1)[0]
        return offsets[idx], records[idx]
