"""Hinge contrastive loss — T2.2, the Razin-safer fallback for `preferred`
feedback that doesn't require auxiliary sampling.

Loss = -log π(y⁺|x) + λ · max(0, log π(y⁻|x) - log π(y⁺|x) + margin)

The first term is standard SFT (EX-RM pathway — Razin-safe).
The second term has log-ratio character but is clipped by the hinge when the
margin is satisfied, which bounds how much surface-cue gradient accumulates.
"""
from __future__ import annotations

from typing import Any

import torch

from ._utils import build_chat_inputs, pad_and_stack, sequence_logprob


def hinge_contrastive_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
                            margin: float = 1.0, rejection_weight: float = 0.5,
                            **_: Any) -> dict[str, Any]:
    """
    `samples` items: {"prompt": str, "chosen": str, "rejected": str}
    """
    if not samples:
        raise ValueError("hinge_contrastive_loss requires at least one sample")

    chosen_tok = [build_chat_inputs(tokenizer, s["prompt"], s["chosen"]) for s in samples]
    rejected_tok = [build_chat_inputs(tokenizer, s["prompt"], s["rejected"]) for s in samples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    cbat = pad_and_stack(chosen_tok, pad_id=pad_id)
    rbat = pad_and_stack(rejected_tok, pad_id=pad_id)

    chosen_sum = sequence_logprob(model, cbat["input_ids"], cbat["labels"],
                                  cbat["attention_mask"])
    rejected_sum = sequence_logprob(model, rbat["input_ids"], rbat["labels"],
                                    rbat["attention_mask"])

    # Length-normalize both sides so length asymmetry doesn't masquerade as signal.
    c_n = (cbat["labels"][:, 1:] != -100).sum(-1).clamp_min(1).float().to(chosen_sum.device)
    r_n = (rbat["labels"][:, 1:] != -100).sum(-1).clamp_min(1).float().to(rejected_sum.device)
    c_lp = chosen_sum / c_n
    r_lp = rejected_sum / r_n

    sft_term = -c_lp.mean()
    hinge_term = torch.clamp(r_lp - c_lp + margin, min=0.0).mean()
    loss = sft_term + rejection_weight * hinge_term
    return {
        "loss": loss,
        "components": {
            "hinge_sft": float(sft_term.detach().cpu()),
            "hinge_rej": float(hinge_term.detach().cpu()),
            "hinge_total": float(loss.detach().cpu()),
        },
    }
