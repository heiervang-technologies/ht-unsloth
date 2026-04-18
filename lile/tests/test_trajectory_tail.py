"""Tests for the incremental tail_structured shape and log_train components.

Covers:
  - log_train back-compat: old 5-arg call produces an event without
    a "components" key.
  - log_train with a components dict: floats/bools/str survive, junk is
    dropped, and the record has "components" as a nested dict.
  - tail_structured(since_offset=0) returns the last n events each
    carrying its byte offset; next_offset equals total_size equals
    file size.
  - tail_structured(since_offset=N) skips earlier events and returns
    only those at or after N, irrespective of n (catch-up semantics).
  - Empty log: events list empty, next_offset == total_size == 0.

Run with: python -m lile.tests.test_trajectory_tail
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from lile.trajectory import TrajectoryLog

pytestmark = pytest.mark.cpu_only


def _fresh_log(td: Path) -> TrajectoryLog:
    return TrajectoryLog(td / "t.jsonl")


def test_log_train_back_compat_no_components():
    with tempfile.TemporaryDirectory() as td:
        log = _fresh_log(Path(td))
        log.log_train("b1", "sft", 0.5, 2, 1)
        ev = log.tail(1)[0]
        assert ev["kind"] == "train_step"
        assert ev["loss"] == 0.5
        assert "components" not in ev


def test_log_train_with_components_serializes_types():
    with tempfile.TemporaryDirectory() as td:
        log = _fresh_log(Path(td))
        log.log_train("b1", "kto", 0.42, 4, 2, components={
            "loss": 0.42,
            "kto_z0": 0.11,
            "kto_z0_source": "adapter_disabled",
            "grad_norm_total": 1.3,
            "grad_clipped": False,
            "junk": object(),  # unserializable → dropped
        })
        ev = log.tail(1)[0]
        comp = ev["components"]
        assert comp["loss"] == 0.42
        assert comp["kto_z0"] == 0.11
        assert comp["kto_z0_source"] == "adapter_disabled"
        assert comp["grad_norm_total"] == 1.3
        assert comp["grad_clipped"] is False
        assert "junk" not in comp


def test_tail_structured_since_zero_caps_to_n():
    with tempfile.TemporaryDirectory() as td:
        log = _fresh_log(Path(td))
        for i in range(5):
            log.log_train(f"b{i}", "sft", float(i), 1, i)
        resp = log.tail_structured(n=3, since_offset=0)
        assert len(resp["events"]) == 3
        # last 3 → batch_ids b2, b3, b4
        assert [e["batch_id"] for e in resp["events"]] == ["b2", "b3", "b4"]
        # offsets strictly increasing, and each is an int
        offsets = [e["offset"] for e in resp["events"]]
        assert offsets == sorted(offsets)
        assert all(isinstance(o, int) for o in offsets)
        # next_offset == total_size == file size after all writes
        assert resp["next_offset"] == resp["total_size"] == log.size()


def test_tail_structured_catch_up_from_offset_ignores_n_cap():
    with tempfile.TemporaryDirectory() as td:
        log = _fresh_log(Path(td))
        for i in range(5):
            log.log_train(f"b{i}", "sft", float(i), 1, i)
        # Grab the offset of the third event; callers would learn this
        # from a prior tail_structured response.
        all_events = log.tail_structured(n=100, since_offset=0)["events"]
        cutoff = all_events[2]["offset"]  # start of b2
        resp = log.tail_structured(n=1, since_offset=cutoff)
        # n=1 must NOT truncate catch-up reads: all events at/after cutoff.
        assert [e["batch_id"] for e in resp["events"]] == ["b2", "b3", "b4"]
        assert resp["next_offset"] == log.size()


def test_tail_structured_empty_log():
    with tempfile.TemporaryDirectory() as td:
        log = _fresh_log(Path(td))
        resp = log.tail_structured(n=10, since_offset=0)
        assert resp == {"events": [], "next_offset": 0, "total_size": 0}


def main() -> int:
    test_log_train_back_compat_no_components()
    test_log_train_with_components_serializes_types()
    test_tail_structured_since_zero_caps_to_n()
    test_tail_structured_catch_up_from_offset_ignores_n_cap()
    test_tail_structured_empty_log()
    print("[test_trajectory_tail] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
