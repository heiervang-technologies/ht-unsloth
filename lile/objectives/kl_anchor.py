"""Batch-level KL anchor against a reference model.

Implements the §5c L_KL term — a forward KL from the trainable π_θ to a frozen
reference π_ref over the prompt's continuation distribution.

Two modes:
    * Token-level KL on the *target* completion: KL(π_θ(·|prefix, target_<t) ‖
      π_ref(·|prefix, target_<t)) averaged over the completion.
    * Prompt-only KL: KL on the next-token distribution at the assistant cue,
      a cheaper "drift indicator" that doesn't require a target.

Both are batch-level — the per-batch composer adds one ``L_KL`` term per step.
"""

from __future__ import annotations

import torch

from lile.objectives import ObjectiveSpec, register
from lile.objectives._utils import chat_prefix, kl_against_reference


def _completion_logits(model, tokenizer, prefix: str, completion: str):
    device = next(model.parameters()).device
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prefix + completion, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    n_prefix = prefix_ids.shape[1]
    if full_ids.shape[1] <= n_prefix:
        return None, None
    return model(full_ids).logits[:, n_prefix - 1 : -1, :], n_prefix


def kl_anchor_batch(
    model,
    tokenizer,
    batch,
    *,
    ref_model,
    weight: float = 0.1,
) -> torch.Tensor:
    """L_KL averaged over the (prompt, target) pairs in ``batch.samples``.

    Samples without a ``target`` are skipped. If no sample qualifies, returns 0.
    """
    if ref_model is None:
        return torch.tensor(0.0, requires_grad=False)

    pieces: list[torch.Tensor] = []
    for s in batch.samples:
        if not s.target:
            continue
        prefix = chat_prefix(tokenizer, s.prompt)
        student_logits, _ = _completion_logits(model, tokenizer, prefix, s.target)
        with torch.no_grad():
            teacher_logits, _ = _completion_logits(ref_model, tokenizer, prefix, s.target)
        if student_logits is None or teacher_logits is None:
            continue
        pieces.append(kl_against_reference(student_logits, teacher_logits))

    if not pieces:
        return torch.tensor(0.0, requires_grad=False)
    return float(weight) * torch.stack(pieces).mean()


register(ObjectiveSpec(
    name="kl_anchor",
    fn=kl_anchor_batch,
    per_sample=False,
    requires=(),
    description="Batch-level forward KL to a reference policy.",
))
