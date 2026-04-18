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
fallback).

The unlike primitive enforces four-tier preconditions at dispatch:

- **Tier 1 (error)** — pure-unlike sample with no ``kl_anchor`` in
  ``batch_objectives`` ⇒ ``ValueError``. Pass ``allow_unanchored=True``
  to override for research / adversarial-testing workflows.
- **Tier 2 (warn)** — pure-unlike + ``kl_anchor`` with
  ``scope != "target_position"``: the anchor does not brake the mass
  movement the push-down is driving.
- **Tier 3 (warn)** — pure-unlike + target-position anchor whose
  exclude set omits the surgery tokens; the anchor fights the push-down.
- **Tier 4 (warn)** — any unlike sample (pure OR positive-teacher) when
  ``effective_lr < _UNLIKE_LR_HEURISTIC_FLOOR`` (5e-5). Flags the
  known-unsafe-η regime on the SFT-on-good side. Upgrades to Cleo's
  closed-form per-sample ``eta_min`` when task A lands.

The ``unlike`` samples carry ``prefix`` rather than ``prompt``;
``kl._sample_text`` accepts either, so the composition is drop-in.
"""
from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F

from ._utils import _to_int_list


# Tier-4 small-η safety floor. Upgrades to Cleo's A closed-form per-sample
# eta_min(p_bad, p_good, V, λ_kl, ε) when that lands — see
# ``docs/research/proofs/razin-safety-sharpened.md`` and the tier-4 upgrade
# note in ``unlike-tiered-preconditions.md``. A static 5e-5 is the stop-gap:
# below it, the positive-teacher side of unlike enters the known-unsafe
# regime where -log p(good) pushes p(bad) UP rather than down.
_UNLIKE_LR_HEURISTIC_FLOOR = 5e-5


def _check_preconditions(
    samples: list[dict[str, Any]],
    batch_objectives: list[dict[str, Any]] | None,
    allow_unanchored: bool,
    effective_lr: float | None,
) -> None:
    """Four-tier anchor-shape + small-η safety gate.

    Tiers 1-3 check pure-unlike (``good_token_id is None``) samples against
    the batch's ``kl_anchor`` configuration. Tier 4 checks the effective
    learning rate for *all* unlike samples (pure OR positive-teacher) —
    Cleo's razin-safety-sharpened.md theorem shows the unsafe-small-η
    failure lives on the SFT-on-good side, not the push-down side.

    When called as a bare primitive (no ``batch_objectives`` passed),
    Tiers 1-3 are skipped — the caller has opted out of the anchor-shape
    contract. Tier 4 is skipped when ``effective_lr`` is unknown.

    Tier 1 is ordered before Tier 4 so a ``ValueError`` preempts the
    warn — the user sees the more urgent error first.
    """
    if batch_objectives is not None:
        pure = [s for s in samples if s.get("good_token_id") is None]
        if pure:
            anchors = [bo for bo in batch_objectives
                       if bo.get("name") == "kl_anchor"]
            if not anchors and not allow_unanchored:
                raise ValueError(
                    "pure-unlike requires kl_anchor in batch_objectives "
                    "(see GLOSSARY.md / design-notes-2026-04-18.md §9); "
                    "pass allow_unanchored=True to override",
                )
            for bo in anchors:
                scope = bo.get("scope", "prompt")
                if scope != "target_position":
                    warnings.warn(
                        f"pure-unlike detected with kl_anchor scope="
                        f"'{scope}'; the anchor does not brake "
                        "target-position mass movement. Consider "
                        "scope='target_position' (see design-notes §9).",
                        RuntimeWarning, stacklevel=2,
                    )
                    continue
                batch_exclude = set(bo.get("exclude_token_ids") or [])
                for s in pure:
                    # Schema fallback in kl._derive_exclude_ids unions the
                    # per-sample {bad,good} automatically — Tier 3 only
                    # fires when the manual ``exclude_token_ids`` batch
                    # list omits surgery tokens AND per-sample derivation
                    # would also miss them. Check the same union the
                    # anchor actually applies.
                    sample_surgery = {
                        s.get("bad_token_id"), s.get("good_token_id"),
                    } - {None}
                    per_sample_derived = sample_surgery  # schema fallback
                    if not sample_surgery.issubset(
                        batch_exclude | per_sample_derived,
                    ):
                        warnings.warn(
                            "kl_anchor scope='target_position' but "
                            "exclude_token_ids does not cover the surgery "
                            "tokens; the anchor fights the push-down at "
                            "the bad/good positions, producing a muddier "
                            "gradient.",
                            RuntimeWarning, stacklevel=2,
                        )

    if effective_lr is not None and effective_lr < _UNLIKE_LR_HEURISTIC_FLOOR:
        warnings.warn(
            f"unlike dispatched with effective_lr={effective_lr:g} < 5e-5 "
            "(known-unsafe regime — the positive-teacher side can push "
            "p_bad UP; see unlike.py docstring and GLOSSARY). Override via "
            "per_objective_lr={\"unlike\": 5e-5} or higher. This heuristic "
            "floor will be replaced by Cleo's closed-form eta_min when "
            "task A lands.",
            RuntimeWarning, stacklevel=2,
        )


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
                allow_unanchored: bool = False,
                batch_objectives: list[dict[str, Any]] | None = None,
                effective_lr: float | None = None,
                **_: Any) -> dict[str, Any]:
    """Surgical unlikelihood loss.

    See module docstring for sample shape. ``default_rank_below`` and
    ``default_prob_above`` apply when a sample doesn't set its own trigger.

    ``batch_objectives`` / ``effective_lr`` / ``allow_unanchored`` drive
    the four-tier precondition gate — see ``_check_preconditions``. When
    called bare (no ``batch_objectives`` passed) the anchor-shape tiers
    are skipped; when ``effective_lr`` is ``None`` the small-η tier is
    skipped. ``TrainEngine.step`` plumbs both through so live dispatches
    run the full gate.
    """
    if not samples:
        raise ValueError("unlike_loss requires at least one sample")

    _check_preconditions(
        samples, batch_objectives, allow_unanchored, effective_lr,
    )

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
