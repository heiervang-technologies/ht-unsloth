"""PR A — snapshot-load resets the optimizer.

Adam-family `m` / `v` moments are conditioned on the trajectory of weights
that produced recent gradients. After `snapshot_load` jumps weights to a
different point, those moments mis-scale the next few updates and can erase
the snapshot's restoration in one or two steps. The fix is to clear
`train_engine._opts` on `snapshot_load` so `_optimizer()` lazily rebuilds
fresh state against the restored weights.

See `lile/docs/research/optimizer-sample-efficiency.md` §1 concern #3.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lile.controller import Controller
from lile.engine.train import TrainEngine

pytestmark = pytest.mark.cpu_only


def test_reset_optimizer_drops_instance():
    engine = TrainEngine.__new__(TrainEngine)
    engine._opts = {"": MagicMock()}  # stand in for a live bnb.optim.AdamW8bit
    assert engine._opts
    engine.reset_optimizer()
    assert engine._opts == {}


def test_handle_task_snapshot_load_resets_optimizer():
    """The controller's single-worker task handler must invoke
    `train_engine.reset_optimizer()` as part of the snapshot_load branch.
    """
    controller = Controller.__new__(Controller)
    controller.state = MagicMock()
    controller.snapshots = MagicMock()
    controller.snapshots.load.return_value = {"residual_fingerprint": "fake"}
    controller.train_engine = TrainEngine.__new__(TrainEngine)
    controller.train_engine._opts = {"": MagicMock()}  # live optimizer

    task = SimpleNamespace(kind="snapshot_load", payload={"name": "test_snap"},
                           token=1, batch_id="b1")
    result = controller._handle_task(task)

    assert result["loaded"] == "test_snap"
    controller.snapshots.load.assert_called_once_with("test_snap", controller.state)
    assert controller.train_engine._opts == {}, (
        "snapshot_load must reset the optimizer so Adam m/v rebuild against "
        "the restored weights"
    )


def test_handle_task_snapshot_save_does_not_reset_optimizer():
    """Control case: snapshot_save must NOT reset the optimizer — saving is
    read-only on weights, so the existing `m`/`v` are still valid.
    """
    controller = Controller.__new__(Controller)
    controller.state = MagicMock()
    controller.snapshots = MagicMock()
    controller.trajectory = MagicMock()
    controller.train_engine = TrainEngine.__new__(TrainEngine)
    sentinel_opt = MagicMock()
    controller.train_engine._opts = {"": sentinel_opt}

    task = SimpleNamespace(kind="snapshot_save", payload={"name": "save_only"},
                           token=2, batch_id="b2")
    controller._handle_task(task)

    assert controller.train_engine._opts.get("") is sentinel_opt, (
        "snapshot_save must preserve the live optimizer state"
    )


def main() -> int:
    test_reset_optimizer_drops_instance()
    test_handle_task_snapshot_load_resets_optimizer()
    test_handle_task_snapshot_save_does_not_reset_optimizer()
    print("[test_snapshot_optimizer_reset] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
