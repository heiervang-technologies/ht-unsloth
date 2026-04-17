"""Token-level utilities shared by every objective module.

The core operation: take a (model, tokenizer, prompt, completion) tuple and
produce the per-token log-probabilities of the completion. This is the building
block for SFT-shaped losses, KTO, CoH, hinge, and CCPD scoring.

All functions operate on a *single* (prompt, completion) pair to keep the API
simple — batched callers loop. This costs ~5% wall-time vs a true micro-batched
version but keeps the loss code obvious. Optimisation can come later if it
shows up in profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class CompletionLogProbs:
    """Result of :func:`completion_logprobs`.

    Fields:
        log_probs : 1D tensor of shape ``(n_completion,)`` with autograd attached.
        token_ids : 1D long tensor of the completion token IDs.
        prefix_len : token count of the prompt prefix.
    """

    log_probs: torch.Tensor
    token_ids: torch.Tensor
    prefix_len: int

    @property
    def n_tokens(self) -> int:
        return int(self.log_probs.numel())

    def sum(self) -> torch.Tensor:
        return self.log_probs.sum()

    def mean(self) -> torch.Tensor:
        return self.log_probs.mean() if self.n_tokens else self.log_probs.new_zeros(())


def _encode_pair(tokenizer, prefix: str, completion: str) -> tuple[torch.Tensor, int]:
    """Tokenize ``prefix + completion`` and return (full_ids[1, T], prefix_len).

    Note: we tokenize ``prefix`` and ``prefix + completion`` separately and use
    the length difference as the completion's token count. This is correct for
    most BPE/SentencePiece tokenizers as long as the prefix ends with a token
    boundary (the chat-template's "assistant\\n" generation prompt does).
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix + completion, return_tensors="pt", add_special_tokens=False).input_ids
    return full_ids, prefix_ids.shape[1]


def completion_logprobs(
    model,
    tokenizer,
    prefix: str,
    completion: str,
    *,
    requires_grad: bool = True,
    max_seq_length: int | None = None,
) -> CompletionLogProbs:
    """Compute per-token log-probability of ``completion`` given ``prefix``.

    If ``requires_grad`` is ``False`` the forward runs under ``no_grad``, so the
    result can be used as a detached scoring signal (CCPD's r_c).
    """
    device = next(model.parameters()).device
    full_ids, prefix_len = _encode_pair(tokenizer, prefix, completion)
    full_ids = full_ids.to(device)

    if max_seq_length is not None and full_ids.shape[1] > max_seq_length:
        # Truncate by dropping completion tokens past the limit. Caller should
        # really cap upstream; this is a safety net.
        full_ids = full_ids[:, :max_seq_length]

    n_completion = full_ids.shape[1] - prefix_len
    if n_completion <= 0:
        # Degenerate: empty completion. Return a zero with autograd attached
        # via a no-op multiplication so backward() is well-defined.
        zero = torch.zeros(0, device=device, dtype=torch.float32)
        if requires_grad:
            zero = zero + 0.0 * sum(p.sum() for p in model.parameters() if p.requires_grad)
        return CompletionLogProbs(
            log_probs=zero,
            token_ids=full_ids[0, 0:0],
            prefix_len=prefix_len,
        )

    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        out = model(full_ids).logits  # [1, T, V]
        # Logits at position t-1 predict token at position t, so the
        # "prediction window" is [prefix_len-1 : T-1].
        logits = out[0, prefix_len - 1 : -1, :].float()
        log_probs_dist = torch.log_softmax(logits, dim=-1)  # [n_completion, V]
        target = full_ids[0, prefix_len:]  # [n_completion]
        chosen = log_probs_dist.gather(1, target.unsqueeze(1)).squeeze(1)
    return CompletionLogProbs(log_probs=chosen, token_ids=target, prefix_len=prefix_len)


def kl_against_reference(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(student || teacher) over per-token distributions.

    Both ``*_logits`` are ``[B, T, V]``; ``mask`` is ``[B, T]`` (1 where valid).
    """
    student_logp = torch.log_softmax(student_logits, dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
    teacher_p = teacher_logp.exp()
    kl_per_token = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)  # [B, T]
    if mask is not None:
        kl_per_token = kl_per_token * mask
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(kl_per_token.numel(), device=kl_per_token.device)
    if reduction == "mean":
        return kl_per_token.sum() / denom
    if reduction == "sum":
        return kl_per_token.sum()
    return kl_per_token


def chat_prefix(tokenizer, prompt: str, system: str | None = None) -> str:
    """Apply the model's chat template, returning the assistant-cued prefix."""
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def average_logprob(model, tokenizer, prefix: str, completion: str, *, requires_grad: bool) -> torch.Tensor:
    """Length-normalised total log-prob — the basis for KTO and CCPD scores."""
    out = completion_logprobs(
        model, tokenizer, prefix, completion, requires_grad=requires_grad
    )
    if out.n_tokens == 0:
        return out.log_probs.new_zeros(())
    return out.log_probs.sum() / out.n_tokens


def stack_completions_logprob_sum(
    model,
    tokenizer,
    pairs: Iterable[tuple[str, str]],
    *,
    requires_grad: bool,
    length_normalise: bool,
) -> torch.Tensor:
    """Sum (or mean) of completion log-probs for a list of (prefix, completion) pairs.

    Returns a 1D tensor with one entry per pair; autograd attached if requested.
    """
    rows = []
    for prefix, completion in pairs:
        lp = completion_logprobs(model, tokenizer, prefix, completion, requires_grad=requires_grad)
        if lp.n_tokens == 0:
            rows.append(lp.log_probs.new_zeros(()))
        else:
            value = lp.log_probs.sum()
            if length_normalise:
                value = value / lp.n_tokens
            rows.append(value)
    return torch.stack(rows)
