"""T1.2 — Kahneman-Tversky Optimization (Ethayarajh et al. 2024) single-step.

The original KTO paper uses paired desirable/undesirable batches with a global KL
estimate. For online/per-event learning we approximate that estimate with the
log-ratio against a reference policy on the *same* sample (effectively a
running EMA-of-one). This is the standard "online KTO" simplification used by
TRL when batch size = 1.

Loss for a single sample:

    Δ ≡ β · [ log π_θ(y|x) − log π_ref(y|x) ]
    if label == "desirable":
        L = − w_d · σ( Δ − KL_ref )
    else:
        L = − w_u · σ( KL_ref − Δ )

where ``KL_ref`` is a small running-mean estimate; we set it to zero in the
single-step approximation (per Ethayarajh 2024 §3 footnote 3) so the loss
collapses to a sigmoid pull on Δ.

Defaults follow the KTO paper's recommendation to overweight undesirable
examples (``desirable_weight=1.0``, ``undesirable_weight=1.5``) per §5b.1.
"""

from __future__ import annotations

import torch

from lile.objectives import ObjectiveSpec, register
from lile.objectives._utils import chat_prefix, average_logprob


def kto_loss(
    model,
    tokenizer,
    sample,
    *,
    ref_model=None,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.5,
) -> torch.Tensor:
    if sample.label not in ("desirable", "undesirable"):
        raise ValueError(
            f"KTO requires sample.label in {{desirable,undesirable}}; got {sample.label!r}"
        )
    if not sample.response:
        raise ValueError("KTO requires sample.response")

    prefix = chat_prefix(tokenizer, sample.prompt)
    pi_logp = average_logprob(model, tokenizer, prefix, sample.response, requires_grad=True)
    if ref_model is not None:
        ref_logp = average_logprob(ref_model, tokenizer, prefix, sample.response, requires_grad=False)
    else:
        # No reference policy available: anchor against a constant zero. This
        # is a degenerate KTO that still produces a useful sigmoid pull.
        ref_logp = pi_logp.detach() * 0.0

    delta = beta * (pi_logp - ref_logp)
    if sample.label == "desirable":
        loss = -torch.nn.functional.logsigmoid(delta) * float(desirable_weight)
    else:
        loss = -torch.nn.functional.logsigmoid(-delta) * float(undesirable_weight)
    return float(sample.weight) * loss


register(ObjectiveSpec(
    name="kto",
    fn=kto_loss,
    per_sample=True,
    requires=("prompt", "response", "label"),
    description="Single-step KTO with desirable/undesirable weighting.",
))
