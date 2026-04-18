"""Surgical unlikelihood — single-position negative-token correction.

Primitive for the household-AI use case: user sees the model generate a bad
token at a specific context (e.g. "The antagonist is" → argmax="Voldemort"),
and wants one piece of feedback to push that choice out — ideally paired with
a positive teacher so the model learns what *should* go there.

Per-sample semantics:

``{
    "prefix":        str,            # context leading up to the position
    "bad_token_id":  int,            # token to push DOWN
    "good_token_id": int | None,     # optional positive teacher
    "rank_below":    int | None,     # trigger if rank(bad_token) < this
    "prob_above":    float | None,   # trigger if p(bad_token) > this
    "weight":        float,          # per-sample weight (default 1.0)
}``

Trigger fires when **either** ``rank(bad) < rank_below`` **or**
``p(bad) > prob_above`` (at least one criterion must be provided). Argmax-only
surgery would miss near-miss cases where the model would have picked the bad
token ~30% of the time and fail to generalize across prefixes; rank/prob
thresholds convert a single feedback into a batched, prefix-generalized push.

Loss at a triggered position:

    L_ul   = -log(1 - p_bad)                 # Welleck et al. 2019, pointwise
    L_sft  = -log(p_good)                    # optional positive teacher
    L      = L_ul + positive_weight * L_sft

At a non-triggered position the sample contributes zero — gradient still
flows through the forward (so KL anchor in the same batch stays honest), but
the unlikelihood term is null.

Compose with ``kl_anchor`` at batch level to bound collateral drift on the
rest of the vocab; this objective does not do the anchor itself.

Razin safety (see GLOSSARY.md and ``docs/research/proofs/razin-safety-sharpened.md``):

With a positive teacher (``good_token_id`` set) the dominant gradient is
likelihood-up on a concrete target, which puts this objective in the
aggregate-safe SFT-family class. Per-token (pointwise) safety is NOT
guaranteed — Cleo's characterization theorem identifies a grower set of
tail tokens with prior mass below ``M_p(η)`` under any SFT-family step.

**⚠️  KNOWN-UNSAFE REGIME: small η with a positive teacher.**

At small η (typically ``lr <= 1e-5``), the positive-teacher side of unlike
is exactly one SFT step at target=good. By the B theorem applied to that
step, if ``p_bad < M_p(η)`` computed at target=good, the positive teacher
**pushes p(bad) UP** — opposing unlike's push-down. The net effect can be
that the correction *raises* the probability of the bad token rather
than lowering it.

**DO NOT default to lr=1e-5 on reflex.** The conventional "smaller LR is
safer" heuristic is inverted here. Use the eta_min from Cleo's A bound
(closed-form, in flight) or the empirical safe-η lower bound from the
calibration sweep (``docs/research/unlike-defaults-calibration.md``, in
flight). Typical safe-floor values land around ``lr >= 5e-5``; validate
for your specific use case.

**Pure unlike** (no positive teacher) is a push-down with no concrete
target and accumulates likelihood displacement fast — a KL anchor
batch-level composition is **required**, not optional, in that mode.
Principled scope: ``{"name": "kl_anchor", "scope": "target_position"}``
with the surgery tokens excluded (derived per-sample via schema
fallback). The unlike primitive enforces tiered preconditions
(error / warn / warn) on pure-unlike calls — pass ``allow_unanchored=True``
to override for research / adversarial-testing workflows.

The ``unlike`` samples carry ``prefix`` rather than ``prompt``;
``kl._sample_text`` accepts either, so the composition is drop-in.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ._utils import _to_int_list


def _should_trigger(rank_bad: int, p_bad: float,
                    rank_below: int | None, prob_above: float | None) -> bool:
    """Fire if *either* criterion is met. At least one must be provided.

    - rank_below: trigger when rank(bad_token) < rank_below (rank 0 = argmax).
    - prob_above: trigger when p(bad_token) > prob_above.

    A sample where neither criterion matches makes zero unlikelihood
    contribution — the bad token isn't a threat at that position.
    """
    if rank_below is None and prob_above is None:
        raise ValueError("at least one of rank_below or prob_above is required")
    hit_rank = rank_below is not None and rank_bad < rank_below
    hit_prob = prob_above is not None and p_bad > prob_above
    return bool(hit_rank or hit_prob)


def _prefix_ids(tokenizer: Any, prefix: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        # Honor chat template if one exists — the "prefix" is a user message
        # plus the assistant generation prompt, matching how the model would
        # actually see the context at generation time.
        raw = tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix}],
            add_generation_prompt=True, tokenize=True, return_tensors=None,
        )
        return _to_int_list(raw)
    return _to_int_list(tokenizer(text=prefix, add_special_tokens=False).input_ids)


def _pad_prefixes(tokenized: list[list[int]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(t) for t in tokenized)
    b = len(tokenized)
    ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    last_idx = torch.zeros((b,), dtype=torch.long)
    for i, t in enumerate(tokenized):
        n = len(t)
        ids[i, :n] = torch.tensor(t, dtype=torch.long)
        attn[i, :n] = 1
        last_idx[i] = n - 1
    return {"input_ids": ids, "attention_mask": attn, "last_idx": last_idx}


def unlike_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
                positive_weight: float = 1.0,
                default_rank_below: int | None = 5,
                default_prob_above: float | None = 0.1,
                eps: float = 1e-6,
                **_: Any) -> dict[str, Any]:
    """Surgical unlikelihood loss.

    See module docstring for sample shape. ``default_rank_below`` and
    ``default_prob_above`` apply when a sample doesn't set its own trigger.
    """
    if not samples:
        raise ValueError("unlike_loss requires at least one sample")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    tokenized = [_prefix_ids(tokenizer, s["prefix"]) for s in samples]
    batch = _pad_prefixes(tokenized, pad_id=pad_id)

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    last_idx = batch["last_idx"].to(device)

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits                                                  # (B, T, V)
    # Gather logits at the last real token position for each row.
    B, _T, V = logits.size()
    row_idx = torch.arange(B, device=device)
    last_logits = logits[row_idx, last_idx]                              # (B, V)
    last_logits_f = last_logits.float()
    log_probs = F.log_softmax(last_logits_f, dim=-1)                     # (B, V)

    bad_ids = torch.tensor([int(s["bad_token_id"]) for s in samples],
                           dtype=torch.long, device=device)              # (B,)
    logp_bad = log_probs.gather(-1, bad_ids.unsqueeze(-1)).squeeze(-1)   # (B,)
    p_bad = logp_bad.exp()

    # Rank = number of tokens with *strictly greater* logit than the bad token.
    # Rank 0 means argmax; rank 4 means 5th-best.
    bad_logits = last_logits_f.gather(-1, bad_ids.unsqueeze(-1)).squeeze(-1)  # (B,)
    rank_bad = (last_logits_f > bad_logits.unsqueeze(-1)).sum(dim=-1)         # (B,)

    # Build per-sample trigger mask.
    trigger_mask = torch.zeros((B,), dtype=torch.bool, device=device)
    for i, s in enumerate(samples):
        rk = s.get("rank_below", default_rank_below)
        pk = s.get("prob_above", default_prob_above)
        trigger_mask[i] = _should_trigger(
            rank_bad=int(rank_bad[i].item()),
            p_bad=float(p_bad[i].item()),
            rank_below=rk, prob_above=pk,
        )

    # Unlikelihood: -log(1 - p_bad), safe for p_bad close to 1.
    one_minus_p = (1.0 - p_bad).clamp_min(eps)
    ul_per_sample = -torch.log(one_minus_p)                              # (B,)

    # Optional positive teacher: -log p(good) at the same position.
    has_good = [s.get("good_token_id") is not None for s in samples]
    if any(has_good):
        good_ids = torch.tensor(
            [int(s["good_token_id"]) if s.get("good_token_id") is not None else 0
             for s in samples], dtype=torch.long, device=device,
        )
        logp_good_all = log_probs.gather(-1, good_ids.unsqueeze(-1)).squeeze(-1)   # (B,)
        good_mask = torch.tensor(has_good, dtype=torch.bool, device=device)
        sft_per_sample = torch.where(good_mask, -logp_good_all,
                                     torch.zeros_like(logp_good_all))
    else:
        sft_per_sample = torch.zeros((B,), dtype=logp_bad.dtype, device=device)

    weights = torch.tensor([float(s.get("weight", 1.0)) for s in samples],
                           dtype=ul_per_sample.dtype, device=device)
    trigger_f = trigger_mask.float()

    # Unlikelihood term only fires on triggered positions. Positive teacher
    # always fires when provided — it's an instruction, not conditional on
    # whether the bad token was currently a threat.
    per_sample = trigger_f * ul_per_sample + positive_weight * sft_per_sample
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)

    with torch.no_grad():
        n_triggered = int(trigger_mask.sum().item())
        ul_mean = float((ul_per_sample * trigger_f).sum().detach().cpu()
                        / max(n_triggered, 1))
        sft_mean = (float(sft_per_sample.sum().detach().cpu()
                          / max(sum(has_good), 1))
                    if any(has_good) else 0.0)
        p_bad_mean = float(p_bad.mean().detach().cpu())
        rank_bad_mean = float(rank_bad.float().mean().detach().cpu())

    return {
        "loss": loss,
        "components": {
            "unlike_loss": float(loss.detach().cpu()),
            "unlike_ul": ul_mean,
            "unlike_sft": sft_mean,
            "unlike_triggered": n_triggered,
            "unlike_n": B,
            "unlike_p_bad_mean": p_bad_mean,
            "unlike_rank_bad_mean": rank_bad_mean,
            "positive_weight": positive_weight,
        },
    }
