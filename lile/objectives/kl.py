"""KL anchor — a batch-level objective modifier.

Computes KL( π_θ(·|x) || π_ref(·|x) ) averaged over sample positions. Requires
a reference model passed in. For the daemon this is typically the base model
(LoRA turned off) or an EMA/snapshot policy.

Scopes
------

- ``scope="prompt"`` — default; anchor over prompt tokens.
- ``scope="full_sequence"`` — anchor prompt + first available response-like
  field.
- ``scope="target_position"`` — single-position anchor at each sample's last
  real token position, with caller-chosen token IDs excluded from the KL
  domain. Built for composition with ``unlike`` (single-position push-down):
  excluding the surgery tokens lets them move freely while the KL pins the
  rest of the vocab at that position. Schema fallback derives the exclude
  set per-sample from ``{bad_token_id, good_token_id}`` when the caller has
  not set ``exclude_token_ids`` explicitly; explicit + per-sample are
  UNIONed so a batch-level list widens (never shrinks) the derived one.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ._utils import _to_int_list


_RESPONSE_FIELDS = ("response", "good", "chosen", "better_response")


def _sample_text(s: dict[str, Any], scope: str) -> str:
    """Pick the text to anchor against.

    scope="prompt": prompt-only (legacy, default). Anchors how the model
        *reads* the prompt.
    scope="full_sequence": prompt + first available response-like field.
        Anchors the joint distribution including generation. For chat-
        templated prompts the closing turn marker (e.g. <|im_end|>) is
        intentionally omitted — KL anchoring targets the pre-closer
        distribution the model actually produces during generation.
    Extending scope to response-only (mask out prompt positions) is the
    logical next step but needs per-sample boundaries; deferred until a
    downstream PR needs it. See sample-efficiency-synthesis.md §1a.

    Schema fallback: accepts ``prompt`` OR ``prefix``. The KL anchor is
    semantically "anchor against the same context the main objective
    trained on", so for objectives whose sample schema uses a different
    field name for context (the ``unlike`` objective uses ``prefix``),
    we fall back transparently. Callers don't need to duplicate fields.
    """
    prompt = s.get("prompt") or s.get("prefix") or ""
    if scope == "prompt":
        return prompt
    if scope == "full_sequence":
        for f in _RESPONSE_FIELDS:
            v = s.get(f)
            if isinstance(v, str) and v:
                return prompt + v
        return prompt
    if scope == "target_position":
        # target_position is not a text-based scope — the caller uses the
        # dedicated target-position branch in ``kl_anchor_loss``. Surfacing
        # this here would hide that mis-routing; raise loudly so the top-level
        # dispatch stays honest.
        raise ValueError(
            "scope='target_position' does not map to a text slice; "
            "kl_anchor_loss handles it via a dedicated branch"
        )
    raise ValueError(f"unknown kl scope {scope!r}")


def _derive_exclude_ids(
    samples: list[dict[str, Any]],
    explicit: list[int] | None,
) -> list[set[int]]:
    """Per-sample exclude-set, UNION of explicit batch list + schema fallback.

    The schema fallback derives ``{bad_token_id, good_token_id}`` when they
    live on the sample (the ``unlike`` schema). Missing fields are simply
    not contributed — the caller controls whether the surgery tokens
    actually exist on each sample.
    """
    explicit_set = set(_to_int_list(explicit)) if explicit else set()
    out: list[set[int]] = []
    for s in samples:
        per_sample: set[int] = set(explicit_set)
        for field in ("bad_token_id", "good_token_id"):
            v = s.get(field)
            if v is None:
                continue
            per_sample.add(int(v))
        out.append(per_sample)
    return out


def _tokenize_prefixes_with_last_idx(
    tokenizer: Any, samples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Build ``(input_ids, attention_mask, last_idx)`` for target-position KL.

    Mirrors the unlike objective's prefix-padding layout so that when unlike
    and this scope land in the same batch, the two forward passes share
    identical token geometry. Schema: uses ``prefix`` if present, otherwise
    falls back to ``prompt`` (consistent with ``_sample_text``).
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    tokenized: list[list[int]] = []
    for s in samples:
        text = s.get("prefix") or s.get("prompt") or ""
        if getattr(tokenizer, "chat_template", None):
            raw = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=True, return_tensors=None,
            )
            ids = _to_int_list(raw)
        else:
            ids = _to_int_list(
                tokenizer(text=text, add_special_tokens=False).input_ids,
            )
        if not ids:
            # Empty prefix → one pad token so the forward still has a last
            # position to gather from. Produces a logits row but the KL at
            # that position is computed against the ref's same position —
            # the degenerate case cancels to ~0 without crashing.
            ids = [pad_id]
        tokenized.append(ids)
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


def _target_position_kl(
    model: Any, tokenizer: Any, samples: list[dict[str, Any]],
    pi_ref: Any | None, use_self_ref: bool, weight: float,
    exclude_ids_per_sample: list[set[int]],
) -> dict[str, Any]:
    """Single-position KL at each sample's last real token, masking out
    caller-supplied exclude IDs before renormalizing.

    KL( π_ref || π_θ ) is the direction the existing anchor uses (log_p =
    log π_θ, log_q = log π_ref, compute ``p * (log_p - log_q)`` with
    ``p = π_θ``). Keep the same direction here so ``scope="target_position"``
    is a drop-in narrower scope, not a different loss.
    """
    if not samples:
        raise ValueError("kl_anchor_loss requires at least one sample")
    batch = _tokenize_prefixes_with_last_idx(tokenizer, samples)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    last_idx = batch["last_idx"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attn,
                   use_cache=False).logits                                 # (B,T,V)
    with torch.no_grad():
        if use_self_ref:
            with model.disable_adapter():
                ref_logits = model(input_ids=input_ids, attention_mask=attn,
                                   use_cache=False).logits
        else:
            ref_logits = pi_ref(input_ids=input_ids,
                                attention_mask=attn, use_cache=False).logits

    B, _T, V = logits.size()
    row_idx = torch.arange(B, device=device)
    last_logits = logits[row_idx, last_idx].float()                        # (B,V)
    last_ref = ref_logits[row_idx, last_idx].float()                       # (B,V)

    # Build per-row exclude mask then mask-fill -inf on BOTH distributions
    # before the softmax so the renormalization is over the same un-excluded
    # vocab on each side — otherwise KL(p||q) mixes supports and is not a
    # valid divergence.
    exclude_mask = torch.zeros((B, V), dtype=torch.bool, device=device)
    excluded_total = 0
    for i, ex in enumerate(exclude_ids_per_sample):
        for tid in ex:
            if 0 <= tid < V:
                exclude_mask[i, tid] = True
                excluded_total += 1
    neg_inf = torch.finfo(last_logits.dtype).min
    masked_logits = last_logits.masked_fill(exclude_mask, neg_inf)
    masked_ref = last_ref.masked_fill(exclude_mask, neg_inf)

    log_p = F.log_softmax(masked_logits, dim=-1)                           # π_θ
    log_q = F.log_softmax(masked_ref, dim=-1)                              # π_ref
    p = log_p.exp()
    # Multiply by ``~exclude_mask`` before the sum so any residual mass that
    # the finite-precision softmax assigned to -inf rows is zeroed — not
    # strictly necessary (softmax(-inf)=0) but cheap belt-and-suspenders.
    kl_per_token = p * (log_p - log_q)
    kl_per_token = kl_per_token.masked_fill(exclude_mask, 0.0)
    kl_per_row = kl_per_token.sum(dim=-1)                                  # (B,)
    kl_mean = kl_per_row.mean()
    loss = weight * kl_mean
    return {
        "loss": loss,
        "components": {
            "kl": float(kl_mean.detach().cpu()),
            "kl_weight": weight,
            "kl_scope": "target_position",
            "kl_excluded_total": excluded_total,
        },
    }


def kl_anchor_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
                   pi_ref: Any | None = None, weight: float = 0.1,
                   max_len: int = 512,
                   pi_ref_mode: str | None = "adapter_disabled",
                   scope: str = "prompt",
                   exclude_token_ids: list[int] | None = None,
                   **_: Any) -> dict[str, Any]:
    # If no external pi_ref is provided, fall back to the LoRA-disabled path on
    # the same model (standard PEFT context manager). Only bail out to a no-op
    # zero loss when neither route is available.
    use_self_ref = (pi_ref is None and pi_ref_mode == "adapter_disabled"
                    and hasattr(model, "disable_adapter"))
    if pi_ref is None and not use_self_ref:
        zero = torch.zeros((), device=next(model.parameters()).device, requires_grad=True)
        return {"loss": zero * weight,
                "components": {"kl": 0.0, "kl_weight": weight, "kl_scope": scope}}
    if not samples:
        raise ValueError("kl_anchor_loss requires at least one sample")

    if scope == "target_position":
        exclude_per_sample = _derive_exclude_ids(samples, exclude_token_ids)
        return _target_position_kl(
            model=model, tokenizer=tokenizer, samples=samples,
            pi_ref=pi_ref, use_self_ref=use_self_ref, weight=weight,
            exclude_ids_per_sample=exclude_per_sample,
        )

    texts = [_sample_text(s, scope) for s in samples]
    tok = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len)
    device = next(model.parameters()).device
    tok = {k: v.to(device) for k, v in tok.items()}

    logits = model(**tok).logits                                  # (B, T, V)
    with torch.no_grad():
        if use_self_ref:
            with model.disable_adapter():
                ref_logits = model(**tok).logits
        else:
            ref_logits = pi_ref(**tok).logits

    # Mean token KL over the prompt positions.
    log_p = F.log_softmax(logits.float(), dim=-1)
    log_q = F.log_softmax(ref_logits.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)                         # (B, T)
    attn = tok.get("attention_mask")
    if attn is not None:
        kl = kl * attn.float()
        kl_mean = kl.sum() / attn.float().sum().clamp_min(1.0)
    else:
        kl_mean = kl.mean()
    loss = weight * kl_mean
    return {
        "loss": loss,
        "components": {
            "kl": float(kl_mean.detach().cpu()),
            "kl_weight": weight,
            "kl_scope": scope,
        },
    }
