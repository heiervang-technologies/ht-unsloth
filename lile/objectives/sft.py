"""Weighted SFT loss — T1.1 in the tier menu.

Per-sample weight supported (3×-5× for rewrite feedback is the plan-recommended
default). Loss gradient flows through the policy log-prob directly; Razin-safe.
"""
from __future__ import annotations

from typing import Any

import torch

from ._utils import build_chat_inputs, pad_and_stack, sequence_logprob


def sft_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
             **_: Any) -> dict[str, Any]:
    """Plain next-token CE on (prompt, response) samples.

    `samples` items: {"prompt": str, "response": str}
    """
    if not samples:
        raise ValueError("sft_loss requires at least one sample")
    tokenized = [build_chat_inputs(tokenizer, s["prompt"], s["response"]) for s in samples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    batch = pad_and_stack(tokenized, pad_id=pad_id)
    summed = sequence_logprob(model, batch["input_ids"], batch["labels"],
                              batch["attention_mask"])
    # Mean NLL per sample.
    shifted_labels = batch["labels"][:, 1:]
    n_tokens = (shifted_labels != -100).sum(dim=-1).clamp_min(1).float().to(summed.device)
    nll = -(summed / n_tokens).mean()
    return {"loss": nll, "components": {"sft_nll": float(nll.detach().cpu())}}


def weighted_sft_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
                      **_: Any) -> dict[str, Any]:
    """SFT but each sample carries a `weight` — used for rewrite upweighting."""
    if not samples:
        raise ValueError("weighted_sft_loss requires at least one sample")
    tokenized = [build_chat_inputs(tokenizer, s["prompt"], s["response"]) for s in samples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    batch = pad_and_stack(tokenized, pad_id=pad_id)
    summed = sequence_logprob(model, batch["input_ids"], batch["labels"],
                              batch["attention_mask"])
    shifted_labels = batch["labels"][:, 1:]
    n_tokens = (shifted_labels != -100).sum(dim=-1).clamp_min(1).float().to(summed.device)
    per_sample_nll = -(summed / n_tokens)
    weights = torch.tensor(
        [float(s.get("weight", 1.0)) for s in samples],
        device=per_sample_nll.device, dtype=per_sample_nll.dtype,
    )
    loss = (per_sample_nll * weights).sum() / weights.sum().clamp_min(1e-6)
    return {
        "loss": loss,
        "components": {
            "weighted_sft_nll": float(loss.detach().cpu()),
            "sum_weights": float(weights.sum().detach().cpu()),
        },
    }
