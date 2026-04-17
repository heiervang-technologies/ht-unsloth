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

from lile.teach.replay_streams.build_mixed_500 import (
    COMPOSITION,
    Composition,
    SOURCE_OFFSET,
    build_records,
    default_teacher,
)
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


# -------------------------------------------------------------------- pipeline
# Exercise `build_records` end-to-end with injected mocks — no HF downloads,
# no cold daemon, no teacher API. Pins the wire shape for every kind.


_MINI = Composition(
    total=20,
    by_kind=(
        ("binary", 8),
        ("nl_critique", 6),
        ("rewrite", 4),
        ("preferred", 2),
    ),
    domains=("math", "code", "common-sense", "general"),
)


class _FakeTeacher:
    """In-process teacher stub — deterministic outputs per kind."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def label_binary(self, *, prompt: str, response: str, domain: str,
                     ground_truth) -> str:  # noqa: ANN001
        self.calls.append(f"binary:{domain}")
        return "up" if len(prompt) % 2 == 0 else "down"

    def rewrite(self, *, prompt: str, response: str, domain: str) -> str:
        self.calls.append(f"rewrite:{domain}")
        return f"rewritten[{domain}]: {response}"

    def critique(self, *, prompt: str, response: str, domain: str) -> str:
        self.calls.append(f"critique:{domain}")
        return f"critique[{domain}]: {response[:40]}"

    def preferred_pair(self, *, prompt: str, response_a: str, response_b: str,
                       domain: str) -> tuple[str, str]:
        self.calls.append(f"pref:{domain}")
        return response_a, response_b


def _fake_prompts(domain: str, n: int, seed: int) -> list[dict]:
    return [
        {
            "prompt": f"[{domain}] question {i}",
            "ground_truth": f"gt-{domain}-{i}",
            "source_idx": SOURCE_OFFSET + i,
        }
        for i in range(n)
    ]


def _fake_cold(prompt: str, domain: str, endpoint: str, seed: int) -> str:
    return f"[{domain}] cold reply (seed={seed})"


def test_build_records_pipeline_with_mocks(tmp_path: Path):
    """End-to-end pipeline on a 20-event fixture slice, no network."""
    teacher = _FakeTeacher()
    records = build_records(
        endpoint="http://127.0.0.1:0/v1",
        teacher_model="mock",
        seed=42,
        composition=_MINI,
        teacher=teacher,
        prompts_fn=_fake_prompts,
        cold_fn=_fake_cold,
    )
    assert len(records) == _MINI.total

    # Every record has the shape the validator needs.
    p = tmp_path / "mini.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    # Validator is hard-coded for total=500; just assert required fields
    # directly so this test doesn't depend on a configurable validator.
    for r in records:
        assert r["response_id"].startswith("mixed_500-")
        assert r["source_idx"] >= SOURCE_OFFSET
        kind = r["feedback_kind"]
        assert kind in {"binary", "rewrite", "preferred",
                        "nl_critique", "nl_critique_with_rewrite"}
        if kind == "binary":
            assert r["value"] in {"up", "down"}
        elif kind == "rewrite":
            assert isinstance(r["better_response"], str)
            assert r["weight"] == 3.0
        elif kind == "preferred":
            assert r["chosen"] and r["rejected"]
        elif kind == "nl_critique":
            assert r["critique"]
        elif kind == "nl_critique_with_rewrite":
            assert r["critique"] and r["better_response"]

    # Teacher was called for every non-cold path.
    kinds_seen = {c.split(":", 1)[0] for c in teacher.calls}
    assert kinds_seen == {"binary", "rewrite", "critique", "pref"}


def test_build_records_is_deterministic_on_seed():
    """Same seed → same record sequence (ordering + content)."""
    def run():
        return build_records(
            endpoint="x", teacher_model="mock", seed=123,
            composition=_MINI, teacher=_FakeTeacher(),
            prompts_fn=_fake_prompts, cold_fn=_fake_cold,
        )
    a, b = run(), run()
    assert [r["response_id"] for r in a] == [r["response_id"] for r in b]
    assert [r["feedback_kind"] for r in a] == [r["feedback_kind"] for r in b]
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]


def test_default_teacher_errors_clearly_without_keys(monkeypatch):
    """No API key → a RuntimeError that names both env vars."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        default_teacher("claude-opus-4-7")
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "OPENAI_API_KEY" in msg


def test_default_teacher_prefers_anthropic(monkeypatch):
    """ANTHROPIC_API_KEY wins over OPENAI_API_KEY and keeps the caller's model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    t = default_teacher("claude-opus-4-7")
    assert t.api_key == "sk-ant-test"  # type: ignore[attr-defined]
    assert t.model == "claude-opus-4-7"  # type: ignore[attr-defined]
    assert t.base_url == "https://api.anthropic.com/v1/"  # type: ignore[attr-defined]


def test_default_teacher_openai_fallback_swaps_claude_model(monkeypatch):
    """OPENAI_API_KEY only + claude-* model → gpt-4o-mini fallback, no base_url."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    t = default_teacher("claude-opus-4-7")
    assert t.model == "gpt-4o-mini"  # type: ignore[attr-defined]
    assert t.base_url is None  # type: ignore[attr-defined]


def test_cold_generate_is_explicitly_unimplemented():
    """Scaffold contract: cold_generate must raise NotImplementedError with
    a message pointing at the cold-daemon recipe so a future implementer
    knows where to look."""
    from lile.teach.replay_streams.build_mixed_500 import cold_generate
    with pytest.raises(NotImplementedError) as exc:
        cold_generate("hi", "math", "http://localhost:0/v1", 0)
    assert "cold-daemon recipe" in str(exc.value).lower()


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
        test_build_records_pipeline_with_mocks(tmp_path)
        print("[replay-streams] build_records pipeline with mocks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
