"""Next-token prediction on raw text — no chat template, no user/assistant frame.

Unlike the chat SFT objectives (which run apply_chat_template and mask
the user turn), NTP treats each sample as a single unbroken sequence and
computes CE over every token. Use when you want to imprint a raw passage
(domain doc, quote, stylistic sample) without forcing it into a
user→assistant frame.

Razin-safe (see lile/GLOSSARY.md): pure likelihood-up, no paired margin.

Sample shape: ``{"text": str}``.
"""
from __future__ import annotations

from typing import Any

import torch

from ._utils import (
    _to_int_list,
    extract_target_positions,
    pad_and_stack,
    sequence_logprob,
)


def ntp_loss(model: Any, tokenizer: Any, samples: list[dict[str, Any]],
             max_len: int = 2048, **_: Any) -> dict[str, Any]:
    if not samples:
        raise ValueError("ntp_loss requires at least one sample")

    tokenized = []
    for s in samples:
        text = s.get("text")
        if not text:
            raise ValueError("ntp sample missing 'text'")
        ids = _to_int_list(tokenizer(text=text, add_special_tokens=True).input_ids)[:max_len]
        t = torch.tensor(ids, dtype=torch.long)
        tokenized.append({
            "input_ids": t,
            "labels": t.clone(),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        })

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    batch = pad_and_stack(tokenized, pad_id=pad_id)
    summed = sequence_logprob(model, batch["input_ids"], batch["labels"],
                              batch["attention_mask"])
    shifted_labels = batch["labels"][:, 1:]
    n_tokens = (shifted_labels != -100).sum(dim=-1).clamp_min(1).float().to(summed.device)
    nll = -(summed / n_tokens).mean()
    positions, target_ids = extract_target_positions(batch["labels"])
    return {
        "loss": nll,
        "components": {"ntp_nll": float(nll.detach().cpu())},
        "target_positions": positions,
        "target_token_ids": target_ids,
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
