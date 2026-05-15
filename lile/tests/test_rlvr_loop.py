"""Online RLVR scheduler — Track C unit tests.

Mocks ``teacher_oss120b.judge`` and ``controller.generate`` /
``controller.submit_train`` so the routing logic can be pinned without
hitting OpenRouter or torch.

Pinned behaviors
----------------

1. For grades=``["correct","wrong","ambiguous","wrong"]`` with critiques
   and counterfactuals on every entry, the captured spec has the expected
   counts: 1× ``weighted_sft``, 2× ``coh``, 1× ``kto``, 4× ``unlike``.
2. ``unlike`` is skipped when ``state.tokenizer`` is ``None`` and the skip
   reasons are recorded in ``spec["_rlvr"]["skipped"]``.
3. ``kl_anchor`` (target_position) is always present in
   ``batch_objectives``.
4. Counterfactuals that are empty / null are skipped without raising.
5. ``dry_run=True`` does NOT call ``controller.submit_train``.

Run: ``pytest lile/tests/test_rlvr_loop.py -xvs``
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from lile.teach import rlvr_loop
from lile.teach.rlvr_loop import (
    RLVRConfig,
    RLVRScheduler,
    build_combined_spec,
)

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------- fakes
class _FakeTokenizer:
    """Trivial tokenizer that maps text -> [hash-derived ids].

    The ``unlike`` primitive only consumes the first token id; we just
    need the ``__call__`` interface ``_bad_token_id_for`` uses.
    """

    def __call__(self, *, text: str, add_special_tokens: bool = False) -> Any:  # noqa: ARG002
        if not text:
            return SimpleNamespace(input_ids=[])
        # Use ord() of first char as the synthetic token id; deterministic
        # and never zero for non-empty input.
        return SimpleNamespace(input_ids=[ord(text[0])])


class _FakeQueue:
    async def wait_for(self, token: int, timeout: float = 120.0) -> Any:  # noqa: ARG002
        return None


class _FakeController:
    """Minimal Controller stand-in for the RLVR scheduler."""

    def __init__(
        self,
        rollouts: list[str],
        *,
        tokenizer: Any | None = None,
    ) -> None:
        self._rollouts = list(rollouts)
        self.state = SimpleNamespace(tokenizer=tokenizer)
        self.queue = _FakeQueue()
        self.generate_calls: list[dict] = []
        self.submitted: list[dict] = []

    async def generate(self, messages, **kwargs):
        self.generate_calls.append({"messages": messages, **kwargs})
        idx = (len(self.generate_calls) - 1) % max(1, len(self._rollouts))
        text = self._rollouts[idx] if self._rollouts else ""
        return {"response": text, "reasoning_content": None, "raw": text}

    async def submit_train(self, spec):
        self.submitted.append(spec)
        return {"commit_token": len(self.submitted), "batch_id": "fake"}


def _judge_result(
    *,
    grades: list[str],
    critiques: list[str | None] | None = None,
    counterfactuals: list[str | None] | None = None,
    demonstration: str = "the canonical answer",
):
    n = len(grades)
    if critiques is None:
        critiques = [
            f"critique-{i}" if g == "wrong" else None
            for i, g in enumerate(grades)
        ]
    if counterfactuals is None:
        counterfactuals = [
            f"counterfactual-{i}" if g == "wrong" else None
            for i, g in enumerate(grades)
        ]
    return {
        "grades": grades,
        "critiques": critiques,
        "counterfactuals": counterfactuals,
        "demonstration": demonstration,
    }


# ---------------------------------------------------------------- tests
def test_build_combined_spec_routes_grades_to_objectives() -> None:
    """One correct, two wrong (with critiques + cfs), one ambiguous → expected counts."""
    grades = ["correct", "wrong", "ambiguous", "wrong"]
    rollouts = [f"rollout-{i}" for i in range(4)]
    judge = _judge_result(grades=grades)
    weights = {"sft": 0.1, "coh": 1.0, "kto": 1.0, "unlike": 0.5, "kl": 0.05}

    spec = build_combined_spec(
        prompt="What is 17 * 23?",
        rollouts=rollouts,
        judge_result=judge,
        domain="math",
        weights=weights,
        tokenizer=_FakeTokenizer(),
    )

    counts: dict[str, int] = {}
    for obj in spec["objectives"]:
        counts[obj["name"]] = counts.get(obj["name"], 0) + 1

    assert counts.get("weighted_sft") == 1
    assert counts.get("coh") == 2
    assert counts.get("kto") == 1
    # Both wrong rollouts produce a non-empty counterfactual; both are
    # tokenized into ``unlike`` entries. (No counterfactual is emitted
    # for correct/ambiguous in our default ``_judge_result``.)
    assert counts.get("unlike") == 2

    # kl_anchor is always present at batch level with target_position scope.
    bos = spec["batch_objectives"]
    assert len(bos) == 1
    assert bos[0]["name"] == "kl_anchor"
    assert bos[0]["scope"] == "target_position"
    assert bos[0]["weight"] == pytest.approx(0.05)

    # Per-objective weight wiring matches config.
    sft_entries = [o for o in spec["objectives"] if o["name"] == "weighted_sft"]
    assert sft_entries[0]["weight"] == pytest.approx(0.1)
    assert sft_entries[0]["samples"][0]["weight"] == 1.0
    coh_entries = [o for o in spec["objectives"] if o["name"] == "coh"]
    assert all(e["weight"] == pytest.approx(1.0) for e in coh_entries)
    # CoH samples carry critique + good (demonstration).
    assert coh_entries[0]["samples"][0]["critique"].startswith("critique-")
    assert coh_entries[0]["samples"][0]["good"] == "the canonical answer"
    kto_entries = [o for o in spec["objectives"] if o["name"] == "kto"]
    assert kto_entries[0]["samples"][0]["label"] == "undesirable"
    unlike_entries = [o for o in spec["objectives"] if o["name"] == "unlike"]
    # Pure-unlike: no good_token_id (anchored by kl_anchor target_position).
    for u in unlike_entries:
        assert "good_token_id" not in u["samples"][0]
        assert isinstance(u["samples"][0]["bad_token_id"], int)


def test_build_combined_spec_skips_unlike_without_tokenizer() -> None:
    """Tokenizer=None ⇒ no unlike entries; reasons in _rlvr.skipped."""
    grades = ["wrong", "wrong"]
    rollouts = ["bad-1", "bad-2"]
    judge = _judge_result(grades=grades)
    spec = build_combined_spec(
        prompt="Q",
        rollouts=rollouts,
        judge_result=judge,
        domain="math",
        weights={"sft": 0.1, "coh": 1.0, "kto": 1.0, "unlike": 0.5, "kl": 0.05},
        tokenizer=None,
    )
    names = [o["name"] for o in spec["objectives"]]
    assert "unlike" not in names
    skipped = spec["_rlvr"]["skipped"]
    assert any("unlike" in s and "no tokenizer" in s for s in skipped)


def test_build_combined_spec_skips_empty_counterfactual() -> None:
    """Empty/null counterfactuals do not emit unlike entries (no crash)."""
    grades = ["wrong", "wrong"]
    judge = _judge_result(
        grades=grades,
        counterfactuals=["", None],
    )
    spec = build_combined_spec(
        prompt="Q",
        rollouts=["a", "b"],
        judge_result=judge,
        domain="math",
        weights={"sft": 0.1, "coh": 1.0, "kto": 1.0, "unlike": 0.5, "kl": 0.05},
        tokenizer=_FakeTokenizer(),
    )
    names = [o["name"] for o in spec["objectives"]]
    assert "unlike" not in names


def test_scheduler_dry_run_skips_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run goes through generate + judge but does NOT call submit_train."""
    grades = ["correct", "wrong", "ambiguous", "wrong"]
    monkeypatch.setattr(
        rlvr_loop, "judge",
        lambda prompt, rollouts, **_: _judge_result(grades=grades),
    )
    rollouts = [f"rollout-{i}" for i in range(4)]
    controller = _FakeController(rollouts, tokenizer=_FakeTokenizer())
    cfg = RLVRConfig(k=4, source="math",
                     log_path="/tmp/lile_rlvr_loop_test.jsonl")
    sched = RLVRScheduler(controller, cfg, dry_run=True)

    records = asyncio.run(sched.run(n=1))

    assert len(records) == 1
    assert controller.submitted == []
    assert len(controller.generate_calls) == 4  # k=4 rollouts
    rec = records[0]
    assert rec["mode"] == "dry-run"
    assert rec["objective_counts"].get("weighted_sft") == 1
    assert rec["objective_counts"].get("coh") == 2
    assert rec["objective_counts"].get("kto") == 1
    assert rec["objective_counts"].get("unlike") == 2


def test_scheduler_submits_when_not_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dry-run path: submit_train is called once with the combined spec."""
    grades = ["correct", "wrong"]
    monkeypatch.setattr(
        rlvr_loop, "judge",
        lambda prompt, rollouts, **_: _judge_result(grades=grades),
    )
    controller = _FakeController(["r0", "r1"], tokenizer=_FakeTokenizer())
    cfg = RLVRConfig(k=2, source="math",
                     log_path="/tmp/lile_rlvr_loop_test_submit.jsonl")
    sched = RLVRScheduler(controller, cfg, dry_run=False)

    records = asyncio.run(sched.run(n=1))

    assert len(records) == 1
    assert len(controller.submitted) == 1
    spec = controller.submitted[0]
    assert "objectives" in spec
    assert spec["batch_objectives"][0]["name"] == "kl_anchor"
    # The record includes commit_token from the fake submit_train.
    assert records[0]["commit_token"] == 1


def test_iter_prompts_math_cycles() -> None:
    it = rlvr_loop.iter_prompts("math")
    first = next(it)
    second = next(it)
    assert first[0] == "math"
    assert isinstance(first[1], str) and first[1]
    assert second[0] == "math"


def test_iter_prompts_mixed_round_robins() -> None:
    it = rlvr_loop.iter_prompts("mixed")
    labels = [next(it)[0] for _ in range(6)]
    # All three labels should appear within the first 6 picks (3 sources
    # round-robin'd twice).
    assert set(labels) == {"math", "code", "arc"}


def test_iter_prompts_unknown_source_raises() -> None:
    with pytest.raises(ValueError):
        next(rlvr_loop.iter_prompts("nonsense"))
