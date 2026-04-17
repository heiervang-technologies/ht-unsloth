"""T1.1 — Weighted SFT (the baseline).

Loss for a single sample:

    L = − (weight) · Σ_t log π_θ(target_t | prompt, target_<t)

Per the §5b feedback pipeline, ``rewrite`` feedback routes here with a high
``weight`` (default 3.0). For ordinary SFT samples the weight is 1.0.
"""

from __future__ import annotations

import torch

from lile.objectives import ObjectiveSpec, register
from lile.objectives._utils import chat_prefix, completion_logprobs


def sft_loss(
    model,
    tokenizer,
    sample,
    *,
    length_normalise: bool = True,
    chat_template: bool = True,
) -> torch.Tensor:
    if sample.target is None or sample.target == "":
        raise ValueError("SFT objective requires sample.target")

    prefix = chat_prefix(tokenizer, sample.prompt) if chat_template else sample.prompt
    out = completion_logprobs(model, tokenizer, prefix, sample.target, requires_grad=True)
    if out.n_tokens == 0:
        return out.log_probs.new_zeros(())
    total = -out.log_probs.sum()
    if length_normalise:
        total = total / out.n_tokens
    return float(sample.weight) * total


register(ObjectiveSpec(
    name="sft",
    fn=sft_loss,
    per_sample=True,
    requires=("prompt", "target"),
    description="Weighted next-token cross-entropy on the (prompt, target) pair.",
))
