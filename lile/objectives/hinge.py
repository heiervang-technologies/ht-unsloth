"""T2.2 — Contrastive SFT with hinge margin.

A "Razin-safer" replacement for vanilla DPO when the user supplies a
``(y⁻, y⁺)`` pair but auxiliary sampling is not available. The y⁺ side flows
through its own log-prob (EX-RM-style pathway through hidden states); the
rejection side uses a clipped hinge so it stops once the margin is satisfied.

Loss:

    L_pos = − log π_θ(y⁺|x)            # SFT on the chosen response
    L_neg = max(0, log π_θ(y⁻|x)
                  − log π_θ(y⁺|x) + margin)   # hinge rejection
    L     = L_pos + λ · L_neg
"""

from __future__ import annotations

import torch

from lile.objectives import ObjectiveSpec, register
from lile.objectives._utils import chat_prefix, completion_logprobs


def hinge_loss(
    model,
    tokenizer,
    sample,
    *,
    margin: float = 1.0,
    rejection_weight: float = 0.5,
    length_normalise: bool = True,
) -> torch.Tensor:
    if not sample.target:
        raise ValueError("hinge objective requires sample.target (the chosen y⁺)")
    if not sample.rejected:
        raise ValueError("hinge objective requires sample.rejected (the y⁻)")

    prefix = chat_prefix(tokenizer, sample.prompt)
    pos = completion_logprobs(model, tokenizer, prefix, sample.target, requires_grad=True)
    neg = completion_logprobs(model, tokenizer, prefix, sample.rejected, requires_grad=True)

    if pos.n_tokens == 0:
        return pos.log_probs.new_zeros(())

    pos_sum = pos.log_probs.sum()
    neg_sum = neg.log_probs.sum() if neg.n_tokens else neg.log_probs.new_zeros(())
    if length_normalise:
        pos_avg = pos_sum / pos.n_tokens
        neg_avg = neg_sum / max(neg.n_tokens, 1)
    else:
        pos_avg = pos_sum
        neg_avg = neg_sum

    L_pos = -pos_avg
    L_neg = torch.relu(neg_avg - pos_avg + float(margin))
    return float(sample.weight) * (L_pos + float(rejection_weight) * L_neg)


register(ObjectiveSpec(
    name="hinge",
    fn=hinge_loss,
    per_sample=True,
    requires=("prompt", "target", "rejected"),
    description="Hinge contrastive SFT — Razin-safer DPO replacement (T2.2).",
))
