"""Invariant tests for trajectory log + snapshot byte-exact round-trip.

The snapshot contract from LIVELEARN §3.1: save→reset→restore yields a state
that is bit-identical to the pre-save state for `merged_deltas` and the
LoRA A/B matrices. This test uses a tiny synthetic state (no GPU) to verify
that the on-disk format round-trips faithfully.

Run with: python -m lile.tests.test_trajectory_snapshot
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lile.snapshot import SnapshotManager
from lile.state import ModelState
from lile.trajectory import TrajectoryLog, new_response_id

pytestmark = pytest.mark.cpu_only


def _make_fake_state() -> ModelState:
    """A ModelState with a stub model (no GPU). Only merged_deltas matter here."""
    state = ModelState.__new__(ModelState)
    state.model = MagicMock()
    # Make model.named_parameters iterable returning (name, param) pairs.
    fake_params: list[tuple[str, torch.nn.Parameter]] = [
        ("base.0.self_attn.q_proj.lora_A.default.weight",
         torch.nn.Parameter(torch.randn(4, 8))),
        ("base.0.self_attn.q_proj.lora_B.default.weight",
         torch.nn.Parameter(torch.randn(8, 4))),
    ]
    state.model.named_parameters = lambda: iter(fake_params)
    state.tokenizer = MagicMock()
    state.base_model_name = "test/fake"
    state.lora_rank = 4
    state.lora_alpha = 8
    state.merges_applied = 2
    state.merged_deltas = {
        "layer.0.q_proj.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        "layer.1.k_proj.weight": torch.randn(16, 16, dtype=torch.bfloat16),
    }
    return state


def test_trajectory_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        rid = new_response_id()
        off1 = log.log_inference(rid, "hi", "hello", "fp123")
        off2 = log.log_feedback(rid, "binary", value="up")
        off3 = log.log_train("b1", "sft", 0.12, 4, 7)
        assert off1 < off2 < off3

        events = list(log.iter_events())
        assert len(events) == 3
        assert events[0]["kind"] == "inference"
        assert events[0]["response_id"] == rid
        assert events[1]["feedback_kind"] == "binary"
        assert events[2]["loss"] == 0.12

        # Size is the final offset + line length; tail(1) returns the last.
        assert log.size() > 0
        assert log.tail(1)[0]["kind"] == "train_step"
    print("[test_trajectory] roundtrip OK")


def test_snapshot_residual_byte_exact():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        state = _make_fake_state()
        fp_before = state.residual_fingerprint()
        mgr = SnapshotManager(root / "snaps")
        mgr.save("snap_a", state)

        # Wipe residual, then restore.
        state.merged_deltas = {}
        assert state.residual_fingerprint() != fp_before

        mgr.load("snap_a", state)
        fp_after = state.residual_fingerprint()
        assert fp_before == fp_after, f"{fp_before} != {fp_after}"

        # Manifest written.
        manifest = json.loads((root / "snaps" / "snap_a" / "manifest.json").read_text())
        assert manifest["residual_fingerprint"] == fp_before
        assert manifest["merges_applied"] == 2
    print("[test_snapshot] residual byte-exact OK")


def test_snapshot_list_and_names():
    with tempfile.TemporaryDirectory() as td:
        mgr = SnapshotManager(Path(td))
        state = _make_fake_state()
        mgr.save("alpha", state)
        mgr.save("beta", state)
        assert mgr.list() == ["alpha", "beta"]
    print("[test_snapshot] list OK")


def main() -> int:
    test_trajectory_roundtrip()
    test_snapshot_residual_byte_exact()
    test_snapshot_list_and_names()
    print("[test_trajectory_snapshot] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
