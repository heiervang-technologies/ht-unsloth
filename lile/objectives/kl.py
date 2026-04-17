"""KL anchor — a batch-level objective modifier.

Computes KL( π_θ(·|x) || π_ref(·|x) ) averaged over sample positions. Requires
a reference model passed in. For the daemon this is typically the base model
(LoRA turned off) or an EMA/snapshot policy.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


_RESPONSE_FIELDS = ("response", "good", "chosen", "better_response")


def _sample_text(s: dict[str, Any], scope: str) -> str:
    """Pick the text to anchor against.

    scope="prompt": prompt-only (legacy, default). Anchors how the model
        *reads* the prompt.
    scope="full_sequence": prompt + first available response-like field.
        Anchors the joint distribution including generation.
    Extending scope to response-only (mask out prompt positions) is the
    logical next step but needs per-sample boundaries; deferred until a
    downstream PR needs it. See sample-efficiency-synthesis.md §1a.
    """
    prompt = s["prompt"]
    if scope == "prompt":
        return prompt
    if scope == "full_sequence":
        for f in _RESPONSE_FIELDS:
            v = s.get(f)
            if isinstance(v, str) and v:
                return prompt + v
        return prompt
    raise ValueError(f"unknown kl scope {scope!r}")


def kl_anchor_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
                   pi_ref: Any | None = None, weight: float = 0.1,
                   max_len: int = 512,
                   pi_ref_mode: str | None = "adapter_disabled",
                   scope: str = "prompt",
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
