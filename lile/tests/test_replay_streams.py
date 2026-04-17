"""Composition + validator tests for lile/teach/replay_streams/.

Covers the shape contract of `mixed_500.jsonl` without requiring the live
cold-model endpoint or teacher API — builds a synthetic stream that matches
the composition spec and asserts the validator accepts it.

Run: python -m lile.tests.test_replay_streams
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import pytest

from lile.teach.replay_streams.build_mixed_500 import COMPOSITION
from lile.teach.replay_streams.validate import validate

pytestmark = pytest.mark.cpu_only


def _synth_record(idx: int, kind: str, domain: str) -> dict:
    base = {
        "response_id": f"mixed_500-{idx:04d}",
        "feedback_kind": kind,
        "prompt": f"test prompt {idx}",
        "response": f"test response {idx}",
        "domain": domain,
        "source_idx": 1000 + idx,
    }
    if kind == "binary":
        base["value"] = "up" if idx % 2 == 0 else "down"
    elif kind == "rewrite":
        base["better_response"] = f"better {idx}"
        base["weight"] = 3.0
    elif kind == "preferred":
        base["chosen"] = f"chosen {idx}"
        base["rejected"] = f"rejected {idx}"
    elif kind == "nl_critique":
        base["critique"] = f"critique {idx}"
    return base


def _build_valid_stream(seed: int = 42) -> list[dict]:
    records: list[dict] = []
    split = COMPOSITION.per_kind_by_domain()
    idx = 0
    for kind, _ in COMPOSITION.by_kind:
        for domain in COMPOSITION.domains:
            for _ in range(split[kind][domain]):
                idx += 1
                records.append(_synth_record(idx, kind, domain))
    random.Random(seed).shuffle(records)
    return records


def test_composition_totals():
    """Per-kind counts and per-kind-domain splits match the declared spec."""
    assert sum(n for _, n in COMPOSITION.by_kind) == COMPOSITION.total
    split = COMPOSITION.per_kind_by_domain()
    for kind, n in COMPOSITION.by_kind:
        assert sum(split[kind].values()) == n
        # Largest-remainder tiebreak: no domain differs by more than 1 from base.
        base = n // len(COMPOSITION.domains)
        for v in split[kind].values():
            assert base <= v <= base + 1


def test_validator_accepts_well_formed_stream(tmp_path: Path):
    records = _build_valid_stream()
    assert len(records) == COMPOSITION.total
    p = tmp_path / "stream.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert validate(p) == 0


def test_validator_rejects_wrong_count(tmp_path: Path):
    records = _build_valid_stream()[:-1]  # drop one
    p = tmp_path / "stream.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert validate(p) == 1


def test_validator_rejects_missing_fields(tmp_path: Path):
    records = _build_valid_stream()
    del records[0]["value"]  # binary record now missing its label
    p = tmp_path / "stream.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert validate(p) == 1


def test_validator_rejects_bad_kind(tmp_path: Path):
    records = _build_valid_stream()
    records[0]["feedback_kind"] = "bogus"
    p = tmp_path / "stream.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert validate(p) == 1


def test_nl_critique_with_rewrite_is_valid(tmp_path: Path):
    """Upgrading a record to nl_critique_with_rewrite stays valid under spec."""
    records = _build_valid_stream()
    for r in records:
        if r["feedback_kind"] == "nl_critique":
            r["feedback_kind"] = "nl_critique_with_rewrite"
            r["better_response"] = "upgraded"
            break
    p = tmp_path / "stream.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert validate(p) == 0


def main() -> int:
    test_composition_totals()
    print("[replay-streams] composition totals OK")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_validator_accepts_well_formed_stream(tmp_path)
        print("[replay-streams] validator accepts well-formed stream")
        test_validator_rejects_wrong_count(tmp_path)
        print("[replay-streams] validator rejects wrong count")
        test_validator_rejects_missing_fields(tmp_path)
        print("[replay-streams] validator rejects missing fields")
        test_validator_rejects_bad_kind(tmp_path)
        print("[replay-streams] validator rejects bad kind")
        test_nl_critique_with_rewrite_is_valid(tmp_path)
        print("[replay-streams] nl_critique_with_rewrite is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
