"""Tests for the T4.1 idle-replay scheduler.

The scheduler is built on three primitives we can exercise without CUDA:

* ``ComputeQueue.is_idle_for`` — extended in W3 to track ``_last_enqueue_ts``.
* ``TrajectoryLog.iter_from(kinds=...)`` — extended in W3 to filter records.
* ``Controller.feedback_to_batch`` — refactored in W3 to be a pure static
  method.

We don't construct a real :class:`Controller` here (it pulls in Unsloth +
CUDA). Instead we hand-roll a minimal stand-in with just the attributes the
scheduler reads. That keeps the test true to the production wiring (the
scheduler's surface contract on the controller is small) while skipping the
GPU dependency. A full end-to-end exercise lives in ``test_smoke.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from lile.controller import Controller
from lile.engine.replay import (
    IdleReplayScheduler,
    ReplayPolicy,
    _is_replayable,
    _recency_weight,
)
from lile.objectives import Batch
from lile.queue import ComputeQueue, ComputeWorker, CommitToken
from lile.trajectory import Record, TrajectoryLog


# --- helpers ---------------------------------------------------------------


class _FakeController:
    """Stand-in with the surface area the scheduler depends on."""

    def __init__(self, trajectory: TrajectoryLog) -> None:
        self.trajectory = trajectory
        self.queue = ComputeQueue()
        self.submitted: list[Batch] = []
        # Drain the queue immediately so submit_train returns quickly and
        # ``is_idle_for`` resets sensibly between submissions.
        self._worker = ComputeWorker(self.queue, handler=self._noop_handler)
        self._worker.start()

    def _noop_handler(self, payload: Any) -> Any:
        return None

    def submit_train(self, batch: Batch) -> CommitToken:
        self.submitted.append(batch)
        token, _fut = self.queue.enqueue(("train", batch))
        return token

    @staticmethod
    def feedback_to_batch(*args, **kwargs) -> Batch:
        return Controller.feedback_to_batch(*args, **kwargs)

    def shutdown(self) -> None:
        self._worker.stop(timeout=2.0)


def _seed_feedback(traj: TrajectoryLog, **payload) -> None:
    """Append a feedback record with W3-shaped payload."""
    defaults = dict(
        response_id="r1",
        feedback_kind="binary",
        prompt="What is 2+2?",
        response="It's four.",
        critique=None,
        better_response=None,
        value="up",
        has_critique=False,
        has_rewrite=False,
    )
    defaults.update(payload)
    traj.append("feedback", **defaults)


# --- unit tests ------------------------------------------------------------


def test_is_replayable_filters_records():
    rec_ok = Record(
        ts=0, kind="feedback", offset=0,
        payload={"feedback_kind": "binary", "prompt": "p", "response": "r"},
    )
    rec_no_kind = Record(ts=0, kind="feedback", offset=0,
                         payload={"prompt": "p", "response": "r"})
    rec_no_prompt = Record(ts=0, kind="feedback", offset=0,
                           payload={"feedback_kind": "binary", "response": "r"})
    rec_old_schema = Record(ts=0, kind="feedback", offset=0,
                            payload={"feedback_kind": "binary",
                                     "has_critique": False, "has_rewrite": False})
    assert _is_replayable(rec_ok)
    assert not _is_replayable(rec_no_kind)
    assert not _is_replayable(rec_no_prompt)
    assert not _is_replayable(rec_old_schema)


def test_recency_weight_decays_with_age():
    # Half-life 24h → 24h-old record gets weight 0.5.
    assert _recency_weight(24 * 3600.0, 24.0) == pytest.approx(0.5, rel=1e-3)
    # Fresh record gets ~1.0.
    assert _recency_weight(0.0, 24.0) == pytest.approx(1.0)
    # 48h old → 0.25.
    assert _recency_weight(48 * 3600.0, 24.0) == pytest.approx(0.25, rel=1e-3)


def test_iter_from_kinds_filter(tmp_path):
    """Verify W3 trajectory filter reads only the requested record kinds."""
    log_path = tmp_path / "trajectory.jsonl"
    with TrajectoryLog(log_path) as log:
        log.append("chat", response_id="x", prompt="p", response="r")
        log.append("feedback", response_id="x", feedback_kind="binary",
                   prompt="p", response="r", value="up")
        log.append("info", event="something")
    feedback = list(TrajectoryLog.iter_from(log_path, kinds={"feedback"}))
    assert len(feedback) == 1
    assert feedback[0].kind == "feedback"
    assert feedback[0].payload["feedback_kind"] == "binary"


def test_queue_is_idle_for(tmp_path):
    q = ComputeQueue()
    # Brand-new queue: should report idle for tiny thresholds.
    time.sleep(0.05)
    assert q.is_idle_for(0.01)
    assert not q.is_idle_for(60.0)
    # Enqueue resets the timer.
    q.enqueue(("noop", None))
    assert not q.is_idle_for(0.01)


def test_scheduler_picks_and_enqueues_one_record(tmp_path):
    """Happy path: one feedback in the log → one replay enqueued."""
    log_path = tmp_path / "trajectory.jsonl"
    with TrajectoryLog(log_path) as traj:
        _seed_feedback(traj)

    traj = TrajectoryLog(log_path)
    fake = _FakeController(traj)
    try:
        sched = IdleReplayScheduler(
            fake,
            policy=ReplayPolicy(
                idle_threshold_s=0.01,
                poll_interval_s=0.01,
                max_replays_per_record=1,
                rng_seed=0,
            ),
        )
        # Drive one tick directly — avoids the thread + sleep flakiness.
        time.sleep(0.05)
        sched._tick()
        assert sched.stats["replays_enqueued"] == 1
        assert len(fake.submitted) == 1
        # Per-record cap → second tick is a no-op.
        sched._tick()
        assert sched.stats["replays_enqueued"] == 1
    finally:
        traj.close()
        fake.shutdown()


def test_scheduler_respects_busy_queue(tmp_path):
    """If the queue isn't idle long enough, scheduler must not enqueue."""
    log_path = tmp_path / "trajectory.jsonl"
    with TrajectoryLog(log_path) as traj:
        _seed_feedback(traj)

    traj = TrajectoryLog(log_path)
    fake = _FakeController(traj)
    try:
        # Long idle threshold — never satisfied in this test window.
        sched = IdleReplayScheduler(
            fake,
            policy=ReplayPolicy(
                idle_threshold_s=60.0,
                poll_interval_s=0.01,
                rng_seed=0,
            ),
        )
        sched._tick()
        assert sched.stats["replays_enqueued"] == 0
        assert sched.stats["skipped_busy"] >= 1
        assert fake.submitted == []
    finally:
        traj.close()
        fake.shutdown()


def test_scheduler_skips_records_without_payload(tmp_path):
    """Records from older code (no prompt/response) must not crash the loop."""
    log_path = tmp_path / "trajectory.jsonl"
    with TrajectoryLog(log_path) as traj:
        # Old-shape record (no prompt/response text).
        traj.append("feedback", response_id="x", feedback_kind="binary",
                    has_critique=False, has_rewrite=False)

    traj = TrajectoryLog(log_path)
    fake = _FakeController(traj)
    try:
        sched = IdleReplayScheduler(
            fake,
            policy=ReplayPolicy(
                idle_threshold_s=0.01,
                poll_interval_s=0.01,
                rng_seed=0,
            ),
        )
        time.sleep(0.05)
        sched._tick()
        assert sched.stats["replays_enqueued"] == 0
        assert sched.stats["skipped_no_candidates"] >= 1
    finally:
        traj.close()
        fake.shutdown()


def test_scheduler_per_record_cap(tmp_path):
    """A single record can be replayed at most max_replays_per_record times."""
    log_path = tmp_path / "trajectory.jsonl"
    with TrajectoryLog(log_path) as traj:
        _seed_feedback(traj)
        _seed_feedback(traj, response_id="r2", value="down")

    traj = TrajectoryLog(log_path)
    fake = _FakeController(traj)
    try:
        sched = IdleReplayScheduler(
            fake,
            policy=ReplayPolicy(
                idle_threshold_s=0.01,
                poll_interval_s=0.01,
                max_replays_per_record=2,
                rng_seed=0,
            ),
        )
        # Tick enough times that, with only 2 records and cap=2, we max out at 4.
        for _ in range(20):
            time.sleep(0.02)
            sched._tick()
        assert sched.stats["replays_enqueued"] <= 4
        # Both records should have been touched.
        assert sched.stats["distinct_records_replayed"] == 2
    finally:
        traj.close()
        fake.shutdown()


def test_feedback_to_batch_weight_scale_applies():
    """Replay weight_scale must propagate to the produced Sample."""
    batch = Controller.feedback_to_batch(
        "binary", prompt="p", bad_response="r", value="up", weight_scale=0.25,
    )
    assert batch.samples[0].weight == pytest.approx(0.25)
    batch2 = Controller.feedback_to_batch(
        "binary", prompt="p", bad_response="r", value="up", weight_scale=1.0,
    )
    assert batch2.samples[0].weight == pytest.approx(1.0)
