"""T4.1 idle replay scheduler: unit tests without GPU.

Verifies the three load-bearing behaviors:
  1. idle-gate: when the queue is not idle, _maybe_replay_one is a no-op.
  2. per-record cap: after max_replays_per_record enqueues for the same
     offset, that offset is excluded from future picks.
  3. recency decay: given identical records with different ts, the weighted
     pick favors the newer one (approximately — sampling is random).

We stub out the Controller with a tiny asyncio-friendly fake that records
every ``submit_train`` call. The scheduler is driven synchronously via
``_maybe_replay_one`` rather than the polling loop, so tests don't need
asyncio.sleep contortions.

Run: python -m lile.tests.test_replay
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Import the modules under test.
from lile.controller import Controller
from lile.engine.replay import IdleReplayScheduler, ReplayPolicy
from lile.trajectory import TrajectoryLog

pytestmark = pytest.mark.cpu_only


class _FakeQueue:
    def __init__(self, idle: bool = True) -> None:
        self.idle = idle

    def is_idle_for(self, seconds: float) -> bool:
        return self.idle


class _FakeController:
    """Minimal Controller stand-in for the replay scheduler."""

    def __init__(self, trajectory: TrajectoryLog, queue: _FakeQueue) -> None:
        self.trajectory = trajectory
        self.queue = queue
        self.submitted: list[dict] = []

    # IdleReplayScheduler calls these two:
    @staticmethod
    def feedback_to_batch(record, prompt_fallback=None, response_fallback=None):
        return Controller.feedback_to_batch(record, prompt_fallback, response_fallback)

    async def submit_train(self, spec):
        self.submitted.append(spec)
        return {"commit_token": len(self.submitted) - 1}


def _write_feedback(log: TrajectoryLog, *, kind: str, prompt: str,
                    response: str, critique: str | None = None,
                    better_response: str | None = None,
                    ts: float | None = None) -> int:
    """Append a feedback event via the public ``append_raw`` helper.

    Tests need ``ts`` control that the canonical ``log_feedback`` (which
    stamps ``time.time()``) can't provide. Using ``append_raw`` keeps this
    test decoupled from the log's internal lock/path layout.
    """
    fields = {"prompt": prompt, "response": response}
    if critique:
        fields["critique"] = critique
    if better_response:
        fields["better_response"] = better_response
    payload = {
        "kind": "feedback",
        "ts": ts if ts is not None else time.time(),
        "response_id": f"r_test_{id(fields)}",
        "feedback_kind": kind,
        **fields,
    }
    return log.append_raw(payload)


# ---------------------------------------------------------------------- tests
def test_idle_gate_blocks_replay():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        for i in range(5):
            _write_feedback(log, kind="nl_critique_with_rewrite",
                            prompt=f"p{i}", response="bad",
                            critique="be nicer", better_response="good")
        ctrl = _FakeController(log, _FakeQueue(idle=False))
        sched = IdleReplayScheduler(ctrl, ReplayPolicy(
            idle_threshold_s=10, poll_interval_s=0.01, max_replays_per_record=3,
        ))

        async def run():
            # Drive the polling loop for a brief window; nothing should replay
            # because queue claims non-idle.
            await sched.start()
            await asyncio.sleep(0.05)
            await sched.stop()

        asyncio.run(run())
        assert len(ctrl.submitted) == 0, f"expected 0 replays, got {len(ctrl.submitted)}"
        assert sched.stats["idle_checks"] > 0
        print("[replay] idle-gate OK (no replays while queue claims busy)")


def test_per_record_cap_is_respected():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        # Only ONE feedback record — scheduler should replay it until cap.
        offset = _write_feedback(log, kind="nl_critique_with_rewrite",
                                 prompt="p", response="bad",
                                 critique="fix", better_response="good")
        # min_feedback_records default is 3; relax to 1 for this test.
        ctrl = _FakeController(log, _FakeQueue(idle=True))
        sched = IdleReplayScheduler(ctrl, ReplayPolicy(
            idle_threshold_s=0, poll_interval_s=0.01,
            max_replays_per_record=2, min_feedback_records=1,
        ))

        async def run():
            # Fire _maybe_replay_one four times directly; cap is 2.
            for _ in range(4):
                await sched._maybe_replay_one()

        asyncio.run(run())
        assert len(ctrl.submitted) == 2, (
            f"expected exactly 2 replays (cap), got {len(ctrl.submitted)}"
        )
        assert sched._replayed[offset] == 2
        # Third+ calls should have hit the empty-after-cap branch.
        assert sched.stats["replays_skipped_empty"] >= 1
        print("[replay] per-record cap respected (stopped at 2)")


def test_recency_decay_prefers_newer():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        now = time.time()
        # One old record (10 half-lives ago: weight ≈ 2^-10 ≈ 1e-3) and one
        # fresh record (age 0: weight = 1). Over 50 picks the fresh one should
        # be picked almost every time. Deterministic-ish (no fixed seed to
        # keep the test honest; tolerance is generous).
        half_life_h = 1.0
        old_ts = now - 10 * 3600 * half_life_h
        old_off = _write_feedback(log, kind="nl_critique_with_rewrite",
                                  prompt="p_old", response="bad",
                                  critique="c", better_response="g", ts=old_ts)
        new_off = _write_feedback(log, kind="nl_critique_with_rewrite",
                                  prompt="p_new", response="bad",
                                  critique="c", better_response="g", ts=now)

        ctrl = _FakeController(log, _FakeQueue(idle=True))
        sched = IdleReplayScheduler(ctrl, ReplayPolicy(
            idle_threshold_s=0,
            max_replays_per_record=9_999,  # no cap
            recency_half_life_h=half_life_h,
            min_feedback_records=1,
        ))

        picks_new = 0
        picks_old = 0
        for _ in range(50):
            result = sched._pick_record()
            assert result is not None
            off, _ = result
            if off == new_off:
                picks_new += 1
            elif off == old_off:
                picks_old += 1
        # With ~1000x weight ratio, 50 picks should land on new ≥45 times.
        assert picks_new >= 45, (
            f"recency decay not biased enough: new={picks_new} old={picks_old}"
        )
        print(f"[replay] recency decay OK (new={picks_new}/50, old={picks_old}/50)")


def test_under_min_records_no_pick():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        _write_feedback(log, kind="nl_critique_with_rewrite",
                        prompt="p", response="bad",
                        critique="c", better_response="g")
        ctrl = _FakeController(log, _FakeQueue(idle=True))
        sched = IdleReplayScheduler(ctrl, ReplayPolicy(
            idle_threshold_s=0, min_feedback_records=3,
        ))
        assert sched._pick_record() is None
        print("[replay] min_feedback_records gate OK")


def test_feedback_to_batch_rewrite_routing():
    """Sanity: pure function handles all 4 feedback kinds."""
    r = Controller.feedback_to_batch({
        "feedback_kind": "binary", "prompt": "p", "response": "r", "value": "down",
    })
    assert r["objective"] == "kto"
    assert r["samples"][0]["label"] == "undesirable"

    r = Controller.feedback_to_batch({
        "feedback_kind": "rewrite", "prompt": "p", "response": "r",
        "better_response": "better", "weight": 2.0,
    })
    assert r["objective"] == "weighted_sft"
    assert r["samples"][0]["weight"] == 2.0

    r = Controller.feedback_to_batch({
        "feedback_kind": "nl_critique", "prompt": "p", "response": "r",
        "critique": "c",
    })
    assert r["objective"] == "coh"
    assert "critique" in r["samples"][0]

    r = Controller.feedback_to_batch({
        "feedback_kind": "nl_critique_with_rewrite", "prompt": "p", "response": "r",
        "critique": "c", "better_response": "g",
    })
    assert r["objective"] == "coh"
    assert r["samples"][0]["good"] == "g"

    # Under-specified rewrite → None, not exception.
    r = Controller.feedback_to_batch({
        "feedback_kind": "rewrite", "prompt": "p", "response": "r",  # no better_response
    })
    assert r is None

    # Missing prompt → None.
    r = Controller.feedback_to_batch({"feedback_kind": "binary", "value": "up"})
    assert r is None

    print("[replay] feedback_to_batch routing OK across all 4 kinds")


def main() -> int:
    test_feedback_to_batch_rewrite_routing()
    test_under_min_records_no_pick()
    test_per_record_cap_is_respected()
    test_recency_decay_prefers_newer()
    test_idle_gate_blocks_replay()
    print("[test_replay] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
