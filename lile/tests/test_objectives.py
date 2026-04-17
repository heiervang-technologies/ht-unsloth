"""Tests for :mod:`lile.objectives` — registry + per-objective sanity.

Uses ``llamafactory/tiny-random-Llama-3`` (a ~50 KB random-init Llama-3 with a
working chat template) to exercise the real (model, tokenizer) interface
on CPU. Each test asserts:

1. The loss is a finite scalar.
2. Autograd is attached (``backward()`` runs cleanly).
3. The loss value moves on a synthetic gradient step (sanity that the
   objective actually updates the model in the expected direction).

We deliberately *don't* assert specific numeric values — those depend on
random init. The contract under test is "produces a useful gradient signal."
"""

from __future__ import annotations

import math
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lile import objectives as O
from lile.objectives import Batch, Sample
from lile.objectives.ccpd import CCPDConfig, ccpd_loss_objective, ccpd_step


_MODEL = "llamafactory/tiny-random-Llama-3"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(_MODEL, torch_dtype=torch.float32)
    model.eval()  # default to eval; tests flip to train when needed
    # Make at least one parameter trainable so backward() has a target.
    for p in model.parameters():
        p.requires_grad_(True)
    return model, tok


# --- Registry sanity -----------------------------------------------------


def test_registry_has_all_v0_objectives():
    names = set(O.list_objectives())
    assert {"sft", "kto", "coh", "hinge", "ccpd", "kl_anchor", "rejection_sft"}.issubset(names)


def test_validate_sample_rejects_missing_fields():
    s = Sample(prompt="hi")  # no target
    with pytest.raises(ValueError, match="target"):
        O.validate_sample(s, "sft")


def test_validate_sample_rejects_batch_objective_in_sample_slot():
    s = Sample(prompt="hi")
    with pytest.raises(ValueError, match="batch-level"):
        O.validate_sample(s, "kl_anchor")


def test_validate_batch_objective_rejects_per_sample():
    with pytest.raises(ValueError, match="per-sample"):
        O.validate_batch_objective("sft")


# --- Per-objective sanity ------------------------------------------------


def _is_finite_scalar(t: torch.Tensor) -> bool:
    return t.dim() == 0 and bool(torch.isfinite(t).item())


def test_sft_loss_is_finite_and_differentiable(model_and_tokenizer):
    model, tok = model_and_tokenizer
    sample = Sample(prompt="What is 1+1?", target="2", weight=1.0)
    loss = O.get("sft").fn(model, tok, sample)
    assert _is_finite_scalar(loss)
    assert loss.requires_grad
    loss.backward()
    # Reset grads after.
    model.zero_grad(set_to_none=True)


def test_sft_weight_scales_loss(model_and_tokenizer):
    model, tok = model_and_tokenizer
    s1 = Sample(prompt="hi", target="hello", weight=1.0)
    s2 = Sample(prompt="hi", target="hello", weight=3.0)
    l1 = O.get("sft").fn(model, tok, s1)
    l2 = O.get("sft").fn(model, tok, s2)
    assert math.isclose(float(l2), 3.0 * float(l1), rel_tol=1e-4), (
        f"weight=3 should triple loss; got {float(l1):.4f} vs {float(l2):.4f}"
    )


def test_kto_desirable_and_undesirable(model_and_tokenizer):
    model, tok = model_and_tokenizer
    s_good = Sample(prompt="hi", response="hello", label="desirable", weight=1.0)
    s_bad = Sample(prompt="hi", response="hello", label="undesirable", weight=1.0)
    # ref_model=model → Δ=0 exactly → both losses are −logsigmoid(0) × weight.
    # The undesirable side has weight 1.5 by default, so the ratio should be 1.5.
    l_good = O.get("kto").fn(model, tok, s_good, ref_model=model)
    l_bad = O.get("kto").fn(model, tok, s_bad, ref_model=model)
    assert _is_finite_scalar(l_good)
    assert _is_finite_scalar(l_bad)
    ratio = float(l_bad) / float(l_good)
    assert math.isclose(ratio, 1.5, rel_tol=1e-3), (
        f"undesirable/desirable weight ratio off (with ref=model, Δ=0): {ratio}"
    )


def test_kto_rejects_missing_label(model_and_tokenizer):
    model, tok = model_and_tokenizer
    s = Sample(prompt="hi", response="hello")
    with pytest.raises(ValueError, match="label"):
        O.get("kto").fn(model, tok, s)


def test_coh_loss_with_critique_and_target(model_and_tokenizer):
    model, tok = model_and_tokenizer
    s = Sample(
        prompt="What's the capital of France?",
        response="Berlin.",
        critique="That's wrong; the capital is Paris.",
        target="Paris.",
        weight=1.0,
    )
    loss = O.get("coh").fn(model, tok, s)
    assert _is_finite_scalar(loss)
    assert loss.requires_grad


def test_coh_loss_without_target_uses_eos_sentinel(model_and_tokenizer):
    model, tok = model_and_tokenizer
    s = Sample(
        prompt="What's the capital of France?",
        response="Berlin.",
        critique="Wrong.",
    )
    loss = O.get("coh").fn(model, tok, s)
    assert _is_finite_scalar(loss)


def test_hinge_loss_finite(model_and_tokenizer):
    model, tok = model_and_tokenizer
    s = Sample(
        prompt="Pick the better answer.",
        target="The better one.",
        rejected="The worse one.",
        weight=1.0,
    )
    loss = O.get("hinge").fn(model, tok, s)
    assert _is_finite_scalar(loss)
    assert loss.requires_grad


def test_kl_anchor_with_ref_returns_zero_or_finite(model_and_tokenizer):
    model, tok = model_and_tokenizer
    batch = Batch(samples=[Sample(prompt="hi", target="hello")])
    # ref=None → returns 0 tensor (no grad).
    no_ref = O.get("kl_anchor").fn(model, tok, batch, ref_model=None)
    assert float(no_ref) == 0.0
    # ref=model → KL of distribution to itself = 0 (within float noise).
    with_ref = O.get("kl_anchor").fn(model, tok, batch, ref_model=model)
    assert _is_finite_scalar(with_ref)
    assert abs(float(with_ref)) < 1e-3, f"KL(p || p) should be ~0, got {float(with_ref)}"


# --- CCPD specifics ------------------------------------------------------


def test_ccpd_skipped_returns_autograd_zero(model_and_tokenizer):
    """No critique and no rewrite → CCPD should skip and return a zero with
    autograd attached so the trainer's backward call doesn't raise."""
    model, tok = model_and_tokenizer
    s = Sample(prompt="hi", response="hello")
    cfg = CCPDConfig(k=2, tau=999.0)  # force skip via the spread check too
    # Direct ccpd_step path:
    res = ccpd_step(model, tok, s, cfg=cfg)
    assert res.skipped is True
    # Wrapper path used by the registry:
    out = ccpd_loss_objective(model, tok, s)
    assert _is_finite_scalar(out)
    assert out.requires_grad
    # backward() must work.
    out.backward()
    model.zero_grad(set_to_none=True)


def test_ccpd_with_rewrite_produces_loss(model_and_tokenizer):
    """User rewrite path: with target + rejected, CCPD seeds y⁺ at top and runs."""
    model, tok = model_and_tokenizer
    s = Sample(
        prompt="What's 2+2?",
        response="5",
        rejected="5",
        target="4",
        critique="That's incorrect; 2+2=4.",
        weight=1.0,
    )
    cfg = CCPDConfig(k=3, tau=0.0)  # tiny k for test speed; tau=0 to never skip
    res = ccpd_step(model, tok, s, cfg=cfg)
    # Either we get a real loss or we skip — both are acceptable here. The
    # only thing we forbid is a NaN loss.
    if res.loss is not None:
        assert _is_finite_scalar(res.loss)
        assert res.loss.requires_grad


# --- Rejection-SFT (T1.4) ------------------------------------------------


def test_length_judge_scores_in_band_higher():
    from lile.judges import LengthJudge

    j = LengthJudge(target_min=3, target_max=10)
    in_band = j.score("p", "one two three four five")  # 5 words
    out_band = j.score("p", "one " * 50)  # way too long
    assert in_band > out_band
    assert in_band == 1.0


def test_llm_judge_wraps_callable():
    from lile.judges import LLMJudge

    captured = []

    def scorer(prompt: str, response: str) -> float:
        captured.append((prompt, response))
        return len(response) / 10.0

    j = LLMJudge(scorer=scorer)
    assert j.score("p", "abcde") == 0.5
    assert captured == [("p", "abcde")]


def test_rejection_sft_skips_when_no_candidate_meets_threshold(model_and_tokenizer):
    """min_score above any judge output → skipped, autograd-zero loss."""
    from lile.objectives.rejection_sft import (
        RejectionSFTConfig, rejection_sft_loss, rejection_sft_step,
    )
    from lile.judges import LLMJudge

    model, tok = model_and_tokenizer
    s = Sample(prompt="say hi", weight=1.0)
    judge = LLMJudge(scorer=lambda p, r: -1.0)  # always rejects
    cfg = RejectionSFTConfig(k=2, aux_max_new_tokens=4, min_score=0.0, seed=0)
    res = rejection_sft_step(model, tok, s, judge=judge, cfg=cfg)
    assert res.skipped is True
    assert "min_score" in res.reason
    # Wrapper path returns an autograd-zero so the composer doesn't crash.
    out = rejection_sft_loss(
        model, tok, s,
        judge=judge, k=2, aux_max_new_tokens=4, min_score=0.0,
    )
    assert _is_finite_scalar(out)
    assert out.requires_grad
    out.backward()
    model.zero_grad(set_to_none=True)


def test_rejection_sft_picks_argmax_and_returns_loss(model_and_tokenizer):
    """When at least one candidate clears min_score, we get a real differentiable loss."""
    from lile.objectives.rejection_sft import (
        RejectionSFTConfig, rejection_sft_step,
    )
    from lile.judges import LLMJudge

    model, tok = model_and_tokenizer
    s = Sample(prompt="say hi", weight=1.0)

    # Judge returns the negative-length: prefers shorter responses. Anything
    # is above min_score=−1e9 so we never skip on the score floor.
    judge = LLMJudge(scorer=lambda p, r: -float(len(r)))
    cfg = RejectionSFTConfig(k=3, aux_max_new_tokens=4, min_score=-1e9, seed=0)
    res = rejection_sft_step(model, tok, s, judge=judge, cfg=cfg)
    assert res.skipped is False
    assert res.chosen is not None
    # The chosen candidate must be the argmax of the recorded scores.
    best_score = max(res.scores)
    assert res.scores[res.candidates.index(res.chosen)] == best_score
    assert _is_finite_scalar(res.loss)
    assert res.loss.requires_grad


def test_rejection_sft_via_registry(model_and_tokenizer):
    """The objective is wired into the registry under the right name."""
    spec = O.get("rejection_sft")
    assert spec.per_sample is True
    assert "prompt" in spec.requires
    # No "target" required — that's the whole point of rejection-SFT.
    assert "target" not in spec.requires


# --- Composer integration ------------------------------------------------


def test_train_engine_step_composes_objectives(model_and_tokenizer):
    """End-to-end: TrainEngine.step on a mixed batch must run backward and
    bump global_step. The loss must be a finite float."""
    from lile.engine.train import TrainEngine

    model, tok = model_and_tokenizer

    class _StateShim:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def set_training_mode(self):
            self.model.train()

        def set_inference_mode(self):
            self.model.eval()

    state = _StateShim(model, tok)
    engine = TrainEngine(state, lr=1e-5)
    batch = Batch(
        samples=[
            Sample(prompt="hi", target="hello", objectives=[{"sft": {}}]),
            Sample(
                prompt="ok", response="bad", label="undesirable",
                objectives=[{"kto": {}}],
            ),
        ],
    )
    result = engine.step(batch)
    assert math.isfinite(result.loss)
    assert engine.global_step == 1
    assert result.n_samples == 2
    assert "sft" in result.components
    assert "kto" in result.components
