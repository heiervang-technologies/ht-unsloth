"""Inference path.

Uses Unsloth's fast_generate when available, falling back to HF generate.
Inference sees the live LoRA instantly — no sync is required because
training and inference share the same model weights (the single-process
invariant from DESIGN.md).
"""
from __future__ import annotations

import contextlib
import logging
from threading import Thread
from typing import Any, Iterator

import torch

log = logging.getLogger(__name__)


def _apply_template(tokenizer: Any, messages: list[dict[str, Any]],
                    enable_thinking: bool | None) -> str:
    """Render the chat template, forwarding ``enable_thinking`` only when
    the caller set it — templates that don't know the flag must not see it.
    """
    kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def generate_chat(model: Any, tokenizer: Any, messages: list[dict[str, str]],
                  max_new_tokens: int = 256, temperature: float = 0.7,
                  top_p: float = 0.95,
                  enable_thinking: bool | None = None,
                  mode_lock: Any = None) -> str:
    """OpenAI-style messages → generated assistant text.

    `mode_lock` is the ModelState.mode_lock; held across the Unsloth mode flip
    and the entire generate so concurrent training cannot tear down the
    per-layer temp buffers the fast-inference path relies on.
    """
    prompt_text = _apply_template(tokenizer, messages, enable_thinking)
    enc = tokenizer(text=prompt_text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    # Null-object pattern so callers that don't know about the lock still work.
    import contextlib
    lock_cm = mode_lock if mode_lock is not None else contextlib.nullcontext()

    with lock_cm:
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    gen = out[0, input_ids.size(-1):]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text


def generate_chat_stream(model: Any, tokenizer: Any, messages: list[dict[str, str]],
                         max_new_tokens: int = 256, temperature: float = 0.7,
                         top_p: float = 0.95,
                         enable_thinking: bool | None = None,
                         mode_lock: Any = None) -> Iterator[str]:
    """Yield decoded text chunks as the model generates them.

    Uses ``TextIteratorStreamer`` — generate() runs in a background thread,
    chunks are pushed onto a queue that this generator drains. The caller
    MUST fully drain the iterator; bailing early leaves the generate thread
    writing into an abandoned queue and keeps the mode_lock held until the
    thread finishes its full ``max_new_tokens``.
    """
    from transformers import TextIteratorStreamer

    prompt_text = _apply_template(tokenizer, messages, enable_thinking)
    enc = tokenizer(text=prompt_text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    lock_cm = mode_lock if mode_lock is not None else contextlib.nullcontext()

    with lock_cm:
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        def _run():
            with torch.no_grad():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    streamer=streamer,
                )

        thread = Thread(target=_run, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()
