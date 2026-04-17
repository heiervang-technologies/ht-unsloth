"""Inference engine — wraps the live model for chat-style generation.

Single-process, weight-sharing path per DESIGN §3 (we are *not* using a vLLM
sidecar in v0). The trade-off: training and inference share the same CUDA
context, so a training step blocks inference for its duration. This is fine on
a 1× 3090 — the alternative (vLLM sidecar with cross-process weight sync) is a
phase-6 item.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ChatMessage:
    role: str
    content: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatMessage":
        return cls(role=str(d["role"]), content=str(d["content"]))


@dataclass
class GenerationResult:
    response_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    finish_reason: str
    extra: dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """Wraps the live model for chat-style generation.

    Greedy + temperature sampling supported. ``response_id`` is a UUID4 string
    so feedback can later target a specific past response (the §5b.3 contract).
    """

    def __init__(self, state):
        self.state = state

    def _format_chat(
        self,
        messages: list[ChatMessage],
        *,
        enable_thinking: bool = False,
    ) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return self.state.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.state.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    @torch.no_grad()
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        seed: int | None = None,
        enable_thinking: bool = False,
    ) -> GenerationResult:
        # Guarantee inference mode without flipping the global state away from
        # training each call — the controller already manages that. We just
        # ensure no_grad here.
        prefix = self._format_chat(messages, enable_thinking=enable_thinking)
        device = self.state.device
        inputs = self.state.tokenizer(
            prefix, return_tensors="pt", add_special_tokens=False,
        ).to(device)
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.time()
        out = self.state.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.state.tokenizer.pad_token_id or self.state.tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0
        new_token_ids = out[0, inputs.input_ids.shape[1] :]
        text = self.state.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        finish_reason = (
            "stop" if new_token_ids[-1].item() == (self.state.tokenizer.eos_token_id or -1)
            else "length"
        )
        return GenerationResult(
            response_id=str(uuid.uuid4()),
            text=text,
            prompt_tokens=int(inputs.input_ids.shape[1]),
            completion_tokens=int(new_token_ids.shape[0]),
            elapsed_s=elapsed,
            finish_reason=finish_reason,
        )
