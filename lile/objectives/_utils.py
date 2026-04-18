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
                      max_len: int = 2048,
                      span_prefix: str | None = None) -> dict[str, torch.Tensor]:
    """Tokenize (prompt, response) into input_ids + labels with prompt masked out.

    Response tokens carry labels; prompt tokens are masked with -100 so loss
    does not penalize reproducing the prompt.

    ``span_prefix`` (T3.1 trace infilling, §T3.1 in the plan): if given, loss
    is computed *only* on tokens after ``span_prefix``. Interpretation: the
    assistant's response is ``span_prefix + suffix``; we mask the accepted
    prefix so gradient signal is surgically applied to the part that was
    regenerated. The token boundary is resolved by decoding progressively
    longer slices of the response and taking the shortest slice whose text
    ends with ``span_prefix``. This is robust to chat templates that inject
    hidden content between ``<|im_start|>assistant`` and the user-supplied
    text (Qwen3 auto-inserts a ``<think>..</think>`` block, for example),
    which naive string-concatenation tokenization misaligns against.
    """
    if getattr(tokenizer, "chat_template", None):
        # Multimodal processors (Qwen3.5, Gemma 4 E4B, …) crash on string
        # content because apply_chat_template pulls `image`/`video` entries
        # out of the list. Wrap as [{"type":"text","text":...}] when we
        # detect a Processor (has image_processor or video_processor).
        is_processor = (hasattr(tokenizer, "image_processor")
                        or hasattr(tokenizer, "video_processor"))
        user_content = ([{"type": "text", "text": prompt}] if is_processor
                        else prompt)
        asst_content = ([{"type": "text", "text": response}] if is_processor
                        else response)
        messages_prompt = [{"role": "user", "content": user_content}]
        messages_full = messages_prompt + [{"role": "assistant", "content": asst_content}]
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

    prefix_len = len(prompt_ids)
    if span_prefix:
        # Walk token-by-token past the prompt; the first slice whose decoded
        # text ends with span_prefix defines the infill boundary. Robust to
        # chat-template-inserted content (e.g. Qwen3's auto-think block).
        # O(N^2) in assistant tokens but N is small in practice.
        resolved = None
        n = len(prompt_ids)
        for pl in range(n, len(full_ids) + 1):
            decoded = tokenizer.decode(full_ids[n:pl])
            if decoded.endswith(span_prefix):
                resolved = pl
                break
        if resolved is not None:
            prefix_len = resolved

    labels = [-100] * prefix_len + list(full_ids[prefix_len:])
    labels = labels[:len(full_ids)]
    if len(labels) < len(full_ids):
        labels.extend(full_ids[len(labels):])

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
    }


def extract_target_positions(
    labels: torch.Tensor,
) -> tuple[list[list[int]], list[list[int]]]:
    """Derive (target_positions, target_token_ids) from a label tensor.

    ``labels`` is (B, T) with ``-100`` at positions the loss ignores.
    Returns two parallel ``list[list[int]]``:

    - ``positions[i]`` are the *logits* indices ``p`` (range ``0 .. T-2``)
      where the next-token prediction is supervised — i.e. where
      ``labels[i, p+1] != -100``.
    - ``token_ids[i]`` are the corresponding target token IDs.

    Keeps the safety_monitor primitive's contract pure-python: the main
    objective hands off these lists, the sidecar doesn't re-tokenize.
    """
    if labels.dim() != 2:
        raise ValueError(f"expected (B, T) labels, got shape {tuple(labels.shape)}")
    B = labels.size(0)
    positions: list[list[int]] = []
    token_ids: list[list[int]] = []
    shifted = labels[:, 1:]                                            # (B, T-1)
    for i in range(B):
        mask = shifted[i] != -100
        pos_i = mask.nonzero(as_tuple=True)[0].tolist()
        tok_i = shifted[i][mask].tolist()
        positions.append([int(p) for p in pos_i])
        token_ids.append([int(t) for t in tok_i])
    return positions, token_ids


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
