"""Shared helpers for objectives: tokenization, masked logprob, shape checks."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _to_int_list(x: Any) -> list[int]:
    """Coerce tokenizer output to list[int], regardless of version/shape.

    Transformers 5.x apply_chat_template returns diverse shapes depending on
    the tokenize/return_tensors combination — sometimes list[int], sometimes
    a tensor, sometimes a BatchEncoding with input_ids. Normalize here.
    """
    # BatchEncoding / dict-like → unwrap to input_ids first.
    if hasattr(x, "input_ids"):
        return _to_int_list(x.input_ids)
    if isinstance(x, dict):
        return _to_int_list(x["input_ids"])
    if isinstance(x, torch.Tensor):
        if x.dim() == 2:
            x = x[0]
        return [int(v) for v in x.tolist()]
    if isinstance(x, str):
        raise TypeError(f"expected token ids, got str: {x[:50]!r}...")
    # Nested list-of-list?
    if len(x) > 0 and isinstance(x[0], (list, tuple)):
        return _to_int_list(x[0])
    return [int(v) for v in x]


def build_chat_inputs(tokenizer: Any, prompt: str, response: str,
                      max_len: int = 2048) -> dict[str, torch.Tensor]:
    """Tokenize (prompt, response) into input_ids + labels with prompt masked out.

    Response tokens carry labels; prompt tokens are masked with -100 so loss
    does not penalize reproducing the prompt.
    """
    if getattr(tokenizer, "chat_template", None):
        messages_prompt = [{"role": "user", "content": prompt}]
        messages_full = messages_prompt + [{"role": "assistant", "content": response}]
        prompt_raw = tokenizer.apply_chat_template(
            messages_prompt, add_generation_prompt=True, tokenize=True,
            return_tensors=None,
        )
        full_raw = tokenizer.apply_chat_template(
            messages_full, add_generation_prompt=False, tokenize=True,
            return_tensors=None,
        )
        prompt_ids = _to_int_list(prompt_raw)
        full_ids = _to_int_list(full_raw)
    else:
        prompt_ids = _to_int_list(tokenizer(text=prompt, add_special_tokens=False).input_ids)
        full_ids = _to_int_list(tokenizer(text=prompt + response, add_special_tokens=False).input_ids)

    full_ids = full_ids[:max_len]
    if len(prompt_ids) > len(full_ids):
        prompt_ids = prompt_ids[:len(full_ids)]

    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    labels = labels[:len(full_ids)]
    if len(labels) < len(full_ids):
        labels.extend(full_ids[len(labels):])

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
    }


def pad_and_stack(tokenized: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(t["input_ids"].size(0) for t in tokenized)
    b = len(tokenized)
    ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    for i, t in enumerate(tokenized):
        n = t["input_ids"].size(0)
        ids[i, :n] = t["input_ids"]
        labels[i, :n] = t["labels"]
        attn[i, :n] = t["attention_mask"]
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def sequence_logprob(model: Any, input_ids: torch.Tensor, labels: torch.Tensor,
                     attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum log p(y_t | prefix) over tokens where labels != -100.

    Returns a (batch,) tensor of *summed* log-probs (no length norm). Divide by
    the number of non-mask tokens externally if you want a mean.
    """
    if attention_mask is None:
        attention_mask = (input_ids != 0).long()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits                                # (B, T, V)
    # Shift for next-token prediction.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != -100)
    # Replace -100 with 0 for safe gather, then mask.
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logprobs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~mask, 0.0)
    summed = token_logprobs.sum(dim=-1)
    return summed


def sequence_logprob_mean(model: Any, input_ids: torch.Tensor, labels: torch.Tensor,
                          attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Length-normalized log-prob — used by CCPD v2 (§5c.11)."""
    summed = sequence_logprob(model, input_ids, labels, attention_mask)
    with torch.no_grad():
        if labels.dim() == 2:
            # Account for label shift by one position.
            n_tokens = (labels[:, 1:] != -100).sum(dim=-1).clamp_min(1).float()
        else:
            n_tokens = (labels != -100).sum(dim=-1).clamp_min(1).float()
    return summed / n_tokens.to(summed.device)
