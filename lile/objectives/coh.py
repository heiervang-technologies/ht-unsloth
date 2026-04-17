"""Chain of Hindsight (CoH) — T1.3.

Liu, Sferrazza & Abbeel (2023). Convert feedback to a sequence and SFT on
the whole thing. The trick is the framing: the model sees its own bad
response, the critique, and (optionally) a good response, and learns the
association. Razin-safe because the gradient is just SFT.

Sample shape:
  {
    "prompt":    "...",         # x
    "bad":       "...",         # y⁻
    "critique":  "...",         # c
    "good":      "...",         # y⁺ — optional, enables the richer variant
  }
"""
from __future__ import annotations

from typing import Any

from ._utils import build_chat_inputs, pad_and_stack, sequence_logprob


_COH_TEMPLATE_WITH_GOOD = (
    "Previous response: {bad}\n"
    "Feedback: {critique}\n"
    "Revised response: {good}"
)
_COH_TEMPLATE_CRITIQUE_ONLY = (
    "Previous response: {bad}\n"
    "Feedback: {critique}\n"
    "Revised response: "   # model learns to produce what comes next
)


def _format_coh_sample(prompt: str, bad: str, critique: str, good: str | None) -> tuple[str, str]:
    if good:
        body = _COH_TEMPLATE_WITH_GOOD.format(bad=bad, critique=critique, good=good)
        target = body        # SFT on the whole formatted body
    else:
        body = _COH_TEMPLATE_CRITIQUE_ONLY.format(bad=bad, critique=critique)
        target = body
    return prompt, target


def coh_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
             **_: Any) -> dict[str, Any]:
    if not samples:
        raise ValueError("coh_loss requires at least one sample")

    tokenized = []
    for s in samples:
        p, r = _format_coh_sample(
            s["prompt"], s.get("bad", ""), s.get("critique", ""), s.get("good"),
        )
        tokenized.append(build_chat_inputs(tokenizer, p, r))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    batch = pad_and_stack(tokenized, pad_id=pad_id)
    summed = sequence_logprob(model, batch["input_ids"], batch["labels"],
                              batch["attention_mask"])
    shifted_labels = batch["labels"][:, 1:]
    n_tokens = (shifted_labels != -100).sum(dim=-1).clamp_min(1).float().to(summed.device)
    nll = -(summed / n_tokens).mean()
    return {"loss": nll, "components": {"coh_nll": float(nll.detach().cpu())}}
