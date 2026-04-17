"""T1.3 — Chain of Hindsight (Liu, Sferrazza, Abbeel 2023).

Format the sample as a textual feedback transcript:

    Prompt: <x>
    Bad response: <y⁻>
    Feedback: "<c>"
    Good response: <y⁺>

then SFT on the entire sequence. The model learns to *associate* the critique
with the bad response, and (when ``y⁺`` is provided) to produce the corrected
response after seeing the feedback.

This is the recommended path for ``nl_critique`` feedback that lacks both a
rewrite and the auxiliary-sampling budget of CCPD v2 (T2.1).
"""

from __future__ import annotations

import torch

from lile.objectives import ObjectiveSpec, register
from lile.objectives._utils import chat_prefix, completion_logprobs


CoH_TEMPLATE = (
    "Prompt: {prompt}\n"
    "Bad response: {response}\n"
    "Feedback: \"{critique}\"\n"
    "Good response:"
)


def coh_loss(
    model,
    tokenizer,
    sample,
    *,
    length_normalise: bool = True,
) -> torch.Tensor:
    if not (sample.critique and sample.response):
        raise ValueError("CoH objective requires sample.critique and sample.response")

    user_text = CoH_TEMPLATE.format(
        prompt=sample.prompt, response=sample.response, critique=sample.critique
    )
    prefix = chat_prefix(tokenizer, user_text)

    # If a target is given, train the model to produce it as the "Good response".
    # If not, train on a one-token sentinel — i.e. teach the *association*
    # without teaching a specific rewrite. The sentinel is the EOS token by
    # default.
    target = sample.target if sample.target else tokenizer.eos_token or "</s>"

    out = completion_logprobs(model, tokenizer, prefix, " " + target, requires_grad=True)
    if out.n_tokens == 0:
        return out.log_probs.new_zeros(())
    total = -out.log_probs.sum()
    if length_normalise:
        total = total / out.n_tokens
    return float(sample.weight) * total


register(ObjectiveSpec(
    name="coh",
    fn=coh_loss,
    per_sample=True,
    requires=("prompt", "critique", "response"),
    description="Chain of Hindsight (Liu et al. 2023): SFT on prompt/bad-response/critique transcript.",
))
