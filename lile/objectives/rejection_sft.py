"""T1.4 — Rejection-SFT.

Pipeline (per the plan §5b.4 and §5c.16):

1. Sample ``k`` candidate completions ``Y = {y_1, …, y_k}`` from ``π_old(·|x)``
   under ``no_grad`` (memory-neutral; same path CCPD uses for aux rollouts).
2. Score each candidate with the configured judge ``J(x, y) → ℝ``.
3. Pick ``y* = argmax_y J(x, y)``.
4. If ``J(x, y*) < min_score`` skip the step (no good candidate).
5. Run weighted SFT on the (prompt, y*) pair.

The judge is the load-bearing piece. Two reference judges live in
:mod:`lile.judges`:
* :class:`lile.judges.LengthJudge` — rule-based; great for tests and for
  "be more concise" feedback.
* :class:`lile.judges.LLMJudge` — wraps a callable scorer; the production
  hook for reward models or self-judge prompts.

The objective is registered with ``per_sample=True`` and ``requires={"prompt"}``
so it composes with the rest of the v0 stack. ``sample.target`` is **not**
required — that's the whole point: rejection-SFT generates its own targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from lile.judges import Judge, LengthJudge
from lile.objectives import ObjectiveSpec, register
from lile.objectives._utils import (
    chat_prefix,
    completion_logprobs,
)


# Module-level default judge. The trainer wires a real judge via the
# ``judge=`` kwarg in the per-sample objective config. We keep a default so
# tests and early integration paths don't need to thread a judge through every
# call site.
_DEFAULT_JUDGE: Judge = LengthJudge()


def set_default_judge(judge: Judge) -> None:
    """Override the default judge used when no ``judge`` kwarg is supplied."""
    global _DEFAULT_JUDGE
    _DEFAULT_JUDGE = judge


def get_default_judge() -> Judge:
    return _DEFAULT_JUDGE


@dataclass
class RejectionSFTConfig:
    k: int = 4                 # candidates to sample per call
    min_score: float = 0.0     # skip step if best candidate scores below this
    aux_temperature: float = 0.9
    aux_top_p: float = 0.95
    aux_max_new_tokens: int = 96
    seed: int = 0
    length_normalise: bool = True


@dataclass
class RejectionSFTResult:
    skipped: bool
    reason: str
    best_score: float
    chosen: str | None
    candidates: list[str]
    scores: list[float]
    loss: torch.Tensor | None


@torch.no_grad()
def _sample_candidates(
    model_old,
    tokenizer,
    prefix: str,
    *,
    cfg: RejectionSFTConfig,
) -> list[str]:
    """Sample ``cfg.k`` candidate completions from ``model_old`` under no_grad."""
    device = next(model_old.parameters()).device
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    out: list[str] = []
    for i in range(cfg.k):
        torch.manual_seed(cfg.seed + i)
        gen = model_old.generate(
            **inputs,
            max_new_tokens=cfg.aux_max_new_tokens,
            do_sample=True,
            temperature=cfg.aux_temperature,
            top_p=cfg.aux_top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        out.append(text.strip())
    return out


def rejection_sft_step(
    model,
    tokenizer,
    sample,
    *,
    model_old=None,
    judge: Judge | Callable[[str, str], float] | None = None,
    cfg: RejectionSFTConfig | None = None,
) -> RejectionSFTResult:
    """Run one rejection-SFT step. See module docstring for the pipeline.

    ``model_old`` defaults to ``model`` (samples from current policy under
    no_grad — same EMA-1 caveat as CCPD's π_old default; pass a frozen
    reference for stronger guarantees).

    ``judge`` defaults to whatever :func:`get_default_judge` returns.
    """
    cfg = cfg or RejectionSFTConfig()
    if model_old is None:
        model_old = model
    judge_fn: Callable[[str, str], float] = judge or _DEFAULT_JUDGE

    prefix = chat_prefix(tokenizer, sample.prompt)
    candidates = _sample_candidates(model_old, tokenizer, prefix, cfg=cfg)
    if not candidates:
        return RejectionSFTResult(
            skipped=True, reason="no candidates sampled",
            best_score=float("-inf"), chosen=None,
            candidates=[], scores=[], loss=None,
        )

    scores = [float(judge_fn(sample.prompt, c)) for c in candidates]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    if best_score < cfg.min_score:
        return RejectionSFTResult(
            skipped=True,
            reason=f"best score {best_score:.3f} < min_score {cfg.min_score:.3f}",
            best_score=best_score, chosen=None,
            candidates=candidates, scores=scores, loss=None,
        )

    chosen = candidates[best_idx]
    out = completion_logprobs(model, tokenizer, prefix, chosen, requires_grad=True)
    if out.n_tokens == 0:
        return RejectionSFTResult(
            skipped=True, reason="chosen completion is empty after tokenization",
            best_score=best_score, chosen=chosen,
            candidates=candidates, scores=scores, loss=None,
        )
    total = -out.log_probs.sum()
    if cfg.length_normalise:
        total = total / out.n_tokens
    loss = float(sample.weight) * total
    return RejectionSFTResult(
        skipped=False, reason="",
        best_score=best_score, chosen=chosen,
        candidates=candidates, scores=scores, loss=loss,
    )


def rejection_sft_loss(model, tokenizer, sample, *, ref_model=None, **kwargs) -> torch.Tensor:
    """Adapter for the per-sample objective registry.

    Recognised kwargs:
        judge        — a Judge instance or callable (overrides the default).
        k            — candidates to sample.
        min_score    — judge floor below which we skip.
        aux_*        — sampling controls (temperature, top_p, max_new_tokens).
        length_normalise, seed.
    """
    judge = kwargs.pop("judge", None)
    cfg = RejectionSFTConfig(**{
        k: v for k, v in kwargs.items() if k in RejectionSFTConfig.__annotations__
    })
    result = rejection_sft_step(
        model, tokenizer, sample,
        model_old=ref_model, judge=judge, cfg=cfg,
    )
    if result.loss is None:
        # Skipped: return zero with autograd attached so the composer doesn't crash.
        zero = torch.zeros((), device=next(model.parameters()).device)
        return zero + 0.0 * sum(
            p.sum() for p in model.parameters() if p.requires_grad
        )
    return result.loss


register(ObjectiveSpec(
    name="rejection_sft",
    fn=rejection_sft_loss,
    per_sample=True,
    requires=("prompt",),
    description=(
        "T1.4 — sample k candidates from π_old, judge each, run weighted SFT "
        "on argmax. Skip if best score < min_score."
    ),
))
