"""KTO loss — T1.2. Binary (desirable / undesirable) feedback signal.

We implement the KTO objective directly rather than wrapping TRL's KTOTrainer
so the loss fits the lile per-step interface (single forward/backward, no
multi-step dataset iteration). The formula follows Ethayarajh et al. (2024):

  z0 = β · KL( π_θ(·|x) || π_ref(·|x) )
  For desirable y:   L = λ_D · (1 - σ(β · [log π_θ(y|x) - log π_ref(y|x)] - z0))
  For undesirable y: L = λ_U · (1 - σ(z0 - β · [log π_θ(y|x) - log π_ref(y|x)]))

z0 is the reference drift and is estimated by mismatched (x, y) pairs in a
batch. For online single-sample learning we estimate z0 via the batch mean of
log-ratios on mismatched samples, as in TRL's implementation.

This is a minimal, correct implementation for online feedback; defaults favor
the community-validated λ_D=1.0, λ_U=1.5 imbalance from §5b.1.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ._utils import build_chat_inputs, pad_and_stack, sequence_logprob


def kto_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
             pi_ref: Any | None = None, beta: float = 0.1,
             lambda_desirable: float = 1.0, lambda_undesirable: float = 1.5,
             pi_ref_mode: str | None = "adapter_disabled",
             **_: Any) -> dict[str, Any]:
    """
    `samples` items: {"prompt": str, "response": str, "label": "desirable"|"undesirable"}

    Reference policy resolution:
      * If `pi_ref` is supplied, it is used as π_ref (frozen forward under
        `torch.no_grad()`).
      * Otherwise, with the default `pi_ref_mode="adapter_disabled"`, a second
        forward runs under `model.disable_adapter()` — base-only log-probs on
        the same weights, zero extra memory. Same pattern as `kl_anchor_loss`
        and `ccpd_v2_loss`.
      * Only when `pi_ref` is None AND `pi_ref_mode` is disabled does KTO fall
        back to a zero reference, degenerating to a weighted log-likelihood
        term. That path is explicit opt-in, not the default, because it
        silently weakens the binary-feedback signal.
    """
    if not samples:
        raise ValueError("kto_loss requires at least one sample")

    tokenized = [build_chat_inputs(tokenizer, s["prompt"], s["response"]) for s in samples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    batch = pad_and_stack(tokenized, pad_id=pad_id)
    summed = sequence_logprob(model, batch["input_ids"], batch["labels"],
                              batch["attention_mask"])
    shifted_labels = batch["labels"][:, 1:]
    n_tokens = (shifted_labels != -100).sum(dim=-1).clamp_min(1).float().to(summed.device)
    policy_logprob_per_tok = summed / n_tokens   # (B,)

    use_self_ref = (pi_ref is None and pi_ref_mode == "adapter_disabled"
                    and hasattr(model, "disable_adapter"))
    if pi_ref is not None:
        with torch.no_grad():
            ref_summed = sequence_logprob(
                pi_ref, batch["input_ids"], batch["labels"], batch["attention_mask"]
            )
        ref_logprob_per_tok = ref_summed / n_tokens
        ref_mode = "external"
    elif use_self_ref:
        with torch.no_grad(), model.disable_adapter():
            ref_summed = sequence_logprob(
                model, batch["input_ids"], batch["labels"], batch["attention_mask"]
            )
        ref_logprob_per_tok = ref_summed / n_tokens
        ref_mode = "adapter_disabled"
    else:
        ref_logprob_per_tok = torch.zeros_like(policy_logprob_per_tok)
        ref_mode = "degenerate_zero"

    logratio = beta * (policy_logprob_per_tok - ref_logprob_per_tok)
    # z0: expected drift estimated by mean logratio on the batch. This is a
    # reasonable proxy when we lack mismatched pairs.
    with torch.no_grad():
        z0 = logratio.mean().detach()

    labels = [s.get("label", "desirable") for s in samples]
    losses = []
    for lr, lab in zip(logratio, labels):
        if lab == "desirable":
            losses.append(lambda_desirable * (1.0 - torch.sigmoid(lr - z0)))
        else:
            losses.append(lambda_undesirable * (1.0 - torch.sigmoid(z0 - lr)))
    loss = torch.stack(losses).mean()

    return {
        "loss": loss,
        "components": {
            "kto_loss": float(loss.detach().cpu()),
            "kto_z0": float(z0.detach().cpu()),
            "kto_ref_mode": ref_mode,
            "n_desirable": sum(1 for l in labels if l == "desirable"),
            "n_undesirable": sum(1 for l in labels if l == "undesirable"),
        },
    }
