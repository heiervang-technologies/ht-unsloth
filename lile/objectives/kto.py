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

Batch-1 degeneracy and the EMA fix
----------------------------------
With batch-size 1 the "batch mean of log-ratios" degenerates to ``logratio[0]``
itself, so ``logratio - z0 == 0`` and ``1 - σ(0) == 0.5`` — a constant for the
desirable singleton case. TRL has the same artifact. The live-feedback path
almost always submits one sample at a time (one thumbs-up/down per event), so
we maintain a module-level EMA of ``z0`` keyed by ``(id(model), beta)``. When
batch-size < 2 the EMA provides a real drift reference; when batch-size ≥ 2 we
use the batch mean (and update the EMA). Callers that want the old TRL-style
behavior can pass ``z0_ema_alpha=None`` to disable the EMA and fall back to
the batch-mean (which is a constant at batch=1, per above).

Defaults favor the community-validated λ_D=1.0, λ_U=1.5 imbalance from §5b.1.
"""
from __future__ import annotations

from typing import Any

import torch

from ._utils import build_chat_inputs, pad_and_stack, sequence_logprob

# Module-level EMA of z0, keyed by (id(model), beta). Keeps single-sample
# feedback events from collapsing to a zero-signal update (see docstring).
_Z0_EMA: dict[tuple[int, float], float] = {}


def kto_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
             pi_ref: Any | None = None, beta: float = 0.1,
             lambda_desirable: float = 1.0, lambda_undesirable: float = 1.5,
             pi_ref_mode: str | None = "adapter_disabled",
             z0_ema_alpha: float | None = 0.9,
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

    ``z0_ema_alpha``: when set (default 0.9), ``z0`` is tracked as an EMA
    across calls at batch-size 1 and used as the drift reference instead of
    the degenerate-to-zero batch mean. Pass ``None`` to restore TRL-style
    batch-mean behavior (warning: at batch=1 this zero-signal desirable case).
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
    # z0 = expected drift. Batch-mean is a reasonable proxy when we have
    # multiple samples; at batch=1 it collapses to logratio[0] so σ(lr - z0)
    # is constant 0.5. Use a module-level EMA to restore real drift signal
    # on single-sample calls (the production feedback path).
    use_ema = z0_ema_alpha is not None and len(samples) < 2
    ema_key = (id(model), float(beta))
    with torch.no_grad():
        batch_z0 = float(logratio.mean().detach().cpu())
        if use_ema and ema_key in _Z0_EMA:
            z0_val = _Z0_EMA[ema_key]
            z0_source = "ema"
        else:
            z0_val = batch_z0
            z0_source = "batch_mean"
        # Always update EMA from the current batch mean (cold-start included).
        if z0_ema_alpha is not None:
            prev = _Z0_EMA.get(ema_key)
            if prev is None:
                _Z0_EMA[ema_key] = batch_z0
            else:
                a = float(z0_ema_alpha)
                _Z0_EMA[ema_key] = a * prev + (1.0 - a) * batch_z0
        z0 = torch.tensor(z0_val, device=logratio.device, dtype=logratio.dtype)

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
            "kto_z0_source": z0_source,
            "kto_ref_mode": ref_mode,
            "n_desirable": sum(1 for l in labels if l == "desirable"),
            "n_undesirable": sum(1 for l in labels if l == "undesirable"),
        },
    }
