"""T2.1 — CCPD v2 (light) — Critique-Conditional Policy Distillation v2.

Implementation of §5c.11–§5c.16 of ``LIVELEARN.md``.

Pipeline (single feedback event, sample with prompt ``x``, critiqued response
``y⁻``, optional critique ``c`` and optional user rewrite ``y⁺_user``):

1. **Auxiliary sampling.** Sample ``k`` candidate responses ``Y⁺`` from
   ``π_old(·|x, c)`` (or ``π_old(·|x)`` if no critique, e.g. user-rewrite
   feedback). Memory-neutral via shared KV pool — these are inference-only.
2. **Detached scoring.** For each candidate compute
   ``r_c(y) = β · [log π_old(y|x, c) − log π_old(y|x)] / |y|``. All under
   ``no_grad``; r_c is a scalar weight, *never* a differentiated loss term.
3. **Rank-based advantage.** ``A(y) = rank(r_c(y)) − (k+1)/2`` so advantages
   are zero-mean. If user supplied a rewrite ``y⁺_user``, it's seeded at the
   top of the rank (advantage = ``+(k−1)/2``) by construction.
4. **Loss = REINFORCE + top-m SFT distill + KL anchor:**

       L_policy   = − E_y[ A(y) · log π_θ(y|x) ]   (gradient through hidden states)
       L_distill  = − Σ_{y ∈ top-m} log π_θ(y|x)
       L_KL       = KL(π_θ(·|x) ‖ π_ref(·|x))      (handled by kl_anchor)

The Razin et al. (2025) critique of DPO-shaped log-ratio gradients is sidestepped
by construction: the differentiated quantity is ``log π_θ`` weighted by a
scalar, not a log-ratio. See §5c.10–§5c.15 for the derivation.

Important: this objective is *batch-level* — one feedback event = one CCPD step.
Per-sample composition would not make sense for an aux-sampled rollout group.
The composer dispatches CCPD as a special-cased "expand-and-train" path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from lile.objectives import ObjectiveSpec, register
from lile.objectives._utils import (
    average_logprob,
    chat_prefix,
    completion_logprobs,
    stack_completions_logprob_sum,
)


@dataclass
class CCPDConfig:
    k: int = 6                 # auxiliary rollouts per feedback event
    beta: float = 0.1          # log-ratio temperature in r_c
    alpha: float = 0.3         # weight on L_distill
    distill_top_m: int = 2     # how many top-ranked samples flow through L_distill
    tau: float = 0.5           # min advantage spread; below this we skip
    aux_temperature: float = 0.9
    aux_top_p: float = 0.95
    aux_max_new_tokens: int = 96
    seed: int = 0


@dataclass
class CCPDStepResult:
    skipped: bool
    reason: str
    loss: torch.Tensor | None
    advantages: list[float]
    r_c_scores: list[float]
    candidates: list[str]
    n_aux_sampled: int


@torch.no_grad()
def _sample_aux(
    model,
    tokenizer,
    prefix: str,
    *,
    n: int,
    seed: int,
    cfg: CCPDConfig,
) -> list[str]:
    device = next(model.parameters()).device
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    out_texts: list[str] = []
    for i in range(n):
        torch.manual_seed(seed + i)
        gen = model.generate(
            **inputs,
            max_new_tokens=cfg.aux_max_new_tokens,
            do_sample=True,
            temperature=cfg.aux_temperature,
            top_p=cfg.aux_top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            gen[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        out_texts.append(text.strip())
    return out_texts


@torch.no_grad()
def _r_c_scores(
    model_old,
    tokenizer,
    prefix_with_critique: str,
    prefix_without_critique: str,
    candidates: Sequence[str],
    *,
    beta: float,
) -> list[float]:
    scores: list[float] = []
    for cand in candidates:
        ll_with = average_logprob(
            model_old, tokenizer, prefix_with_critique, cand, requires_grad=False
        ).item()
        ll_without = average_logprob(
            model_old, tokenizer, prefix_without_critique, cand, requires_grad=False
        ).item()
        scores.append(float(beta) * (ll_with - ll_without))
    return scores


def _ranks_centered(scores: Sequence[float]) -> list[float]:
    """Average-rank with ties (matches scipy.stats.rankdata). Centered to mean 0."""
    import numpy as np  # local — keeps top-level import light.

    arr = np.array(scores, dtype=np.float64)
    order = arr.argsort()
    ranks = np.empty_like(arr)
    ranks[order] = np.arange(len(arr))
    # Average ties.
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    for r, group in zip(ranks, inv):
        sums[group] += r
    avg = sums / counts
    centered = avg[inv] - (len(arr) - 1) / 2.0
    return [float(x) for x in centered]


def ccpd_step(
    model,            # π_θ trainable
    tokenizer,
    sample,           # one Sample with prompt + (critique | rejected | response)
    *,
    model_old=None,   # π_old, frozen; defaults to model under no_grad
    cfg: CCPDConfig | None = None,
) -> CCPDStepResult:
    """Run one CCPD v2 step. Returns the loss tensor (None if skipped).

    Routing per §5c.16:
        critique only         → aux from π_old(·|x, c), score by r_c
        rewrite only          → user y⁺ at top, fill k−1 from π_old(·|x), score by ll-only proxy
        critique + rewrite    → user y⁺ at top, fill k−2 from π_old(·|x, c), score by r_c
        binary feedback       → caller should use KTO; CCPD asserts critique||target present
    """
    cfg = cfg or CCPDConfig()
    if model_old is None:
        model_old = model  # treat π_θ as π_old under no_grad — acceptable for v0

    has_critique = bool(sample.critique)
    has_rewrite = bool(sample.target)
    if not (has_critique or has_rewrite):
        return CCPDStepResult(
            skipped=True, reason="no critique or rewrite supplied",
            loss=None, advantages=[], r_c_scores=[], candidates=[], n_aux_sampled=0,
        )

    prefix_neutral = chat_prefix(tokenizer, sample.prompt)
    prefix_critiqued = (
        chat_prefix(tokenizer, sample.prompt, system=sample.critique)
        if has_critique else prefix_neutral
    )

    # 1. Auxiliary rollouts (memory-neutral, no_grad).
    aux_prefix = prefix_critiqued
    aux_count = cfg.k
    candidates: list[str] = []
    if has_rewrite:
        candidates.append(sample.target)
        aux_count = max(0, cfg.k - 1)
    if has_rewrite and sample.rejected:
        # Mirror y⁻ as a known-bad anchor in the candidate set.
        candidates.append(sample.rejected)
        aux_count = max(0, aux_count - 1)
    if aux_count > 0:
        aux_samples = _sample_aux(
            model_old, tokenizer, aux_prefix,
            n=aux_count, seed=cfg.seed, cfg=cfg,
        )
        candidates.extend(aux_samples)

    n_aux = len(candidates)
    if n_aux < 2:
        return CCPDStepResult(
            skipped=True, reason="not enough candidates",
            loss=None, advantages=[], r_c_scores=[], candidates=candidates, n_aux_sampled=n_aux,
        )

    # 2. Score (detached) — r_c when critique exists; else log-likelihood proxy.
    if has_critique:
        scores = _r_c_scores(
            model_old, tokenizer, prefix_critiqued, prefix_neutral,
            candidates, beta=cfg.beta,
        )
    else:
        # No critique — use log π_old(y|x). This is a coarse proxy; the
        # user-supplied y⁺ should still rank high because it's a clean
        # in-distribution rewrite, but quality of the auxiliary fill matters.
        scores = []
        with torch.no_grad():
            for cand in candidates:
                lp = average_logprob(
                    model_old, tokenizer, prefix_neutral, cand, requires_grad=False
                ).item()
                scores.append(lp)

    # When the user supplied a rewrite, force it to the top of the ranking by
    # nudging its score (rank-based advantage doesn't care about absolute scale).
    if has_rewrite:
        max_score = max(scores)
        scores[0] = max_score + 1.0
        if sample.rejected:
            min_score = min(scores)
            scores[1] = min_score - 1.0

    advantages = _ranks_centered(scores)

    spread = max(advantages) - min(advantages)
    if spread < cfg.tau:
        return CCPDStepResult(
            skipped=True, reason=f"advantage spread {spread:.3f} < tau {cfg.tau}",
            loss=None, advantages=advantages, r_c_scores=scores,
            candidates=candidates, n_aux_sampled=n_aux,
        )

    # 3. L_policy: REINFORCE with detached scalar advantage on log π_θ(y|x).
    # We use length-normalised log-prob; this matches the rank computation and
    # avoids the trivial "long sequences dominate gradient" issue.
    pairs = [(prefix_neutral, cand) for cand in candidates]
    logp_avg = stack_completions_logprob_sum(
        model, tokenizer, pairs, requires_grad=True, length_normalise=True,
    )
    A = torch.tensor(advantages, device=logp_avg.device, dtype=logp_avg.dtype)
    L_policy = -(A * logp_avg).mean()

    # 4. L_distill: SFT on the top-m ranked candidates.
    # Use unnormalised log-prob sum; SFT is naturally token-summed.
    order = sorted(range(n_aux), key=lambda i: advantages[i], reverse=True)
    top_idx = order[: cfg.distill_top_m]
    top_pairs = [(prefix_neutral, candidates[i]) for i in top_idx]
    logp_sum_top = stack_completions_logprob_sum(
        model, tokenizer, top_pairs, requires_grad=True, length_normalise=True,
    )
    L_distill = -logp_sum_top.mean()

    loss = L_policy + float(cfg.alpha) * L_distill
    return CCPDStepResult(
        skipped=False, reason="",
        loss=loss, advantages=advantages, r_c_scores=scores,
        candidates=candidates, n_aux_sampled=n_aux,
    )


def ccpd_loss_objective(model, tokenizer, sample, *, ref_model=None, **kwargs) -> torch.Tensor:
    """Adapter so CCPD plays in the per-sample objective registry.

    Returns the loss tensor or a zero (preserving autograd graph) when the
    feedback event was skipped.
    """
    cfg = CCPDConfig(**{k: v for k, v in kwargs.items() if k in CCPDConfig.__annotations__})
    result = ccpd_step(model, tokenizer, sample, model_old=ref_model, cfg=cfg)
    if result.loss is None:
        # Return a zero loss with autograd attached to the model so the
        # backward call doesn't raise; effectively a no-op step.
        zero = torch.zeros((), device=next(model.parameters()).device)
        return zero + 0.0 * sum(
            p.sum() for p in model.parameters() if p.requires_grad
        )
    return float(sample.weight) * result.loss


register(ObjectiveSpec(
    name="ccpd",
    fn=ccpd_loss_objective,
    per_sample=True,
    requires=("prompt",),
    description="CCPD v2 light: aux-sampled REINFORCE + top-m distill + (optional) KL anchor.",
))
