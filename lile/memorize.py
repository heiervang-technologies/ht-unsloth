"""Greedy-memorize loop: SFT-train a single (prompt, response) pair until the
model greedily reproduces the response under next-token argmax.

Used by the chat UI's implicit-OK auto-SFT: on the user's next turn, the prior
assistant response is fed here so the model internalizes it without manual
feedback. The loop body lives in a module-level function so jurigged can hot-
patch it without a daemon bounce.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch

from .objectives._utils import build_chat_inputs, pad_and_stack

log = logging.getLogger(__name__)


@torch.no_grad()
def greedy_rank_fraction(model: Any, tokenizer: Any, prompt: str,
                         response: str) -> tuple[float, int, int]:
    """Fraction of response tokens that are argmax under the current model.

    Returns ``(fraction, matched, total)``. A fraction of 1.0 means greedy
    decoding from the prompt reproduces the response verbatim.

    Caller is responsible for holding ``ModelState.mode_lock`` — this runs a
    model forward against the live weights, so concurrent training must be
    excluded the same way streaming inference is.
    """
    tokenized = build_chat_inputs(tokenizer, prompt, response, span_prefix=None)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    batch = pad_and_stack([tokenized], pad_id=pad_id)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    model.eval()
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits  # (1, T, V)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    preds = shift_logits.argmax(dim=-1)  # (1, T-1)
    matched = int(((preds == shift_labels) & mask).sum().item())
    total = int(mask.sum().item())
    if total == 0:
        return (1.0, 0, 0)
    return (matched / total, matched, total)


async def iterate_memorize(
    controller: Any,
    prompt: str,
    response: str,
    max_steps: int = 30,
    threshold: float = 0.95,
    lr: float | None = None,
    weight: float = 1.0,
    plateau_patience: int = 3,
) -> dict[str, Any]:
    """Loop: SFT-step -> eval greedy rank -> repeat until threshold or max_steps.

    Runs on the asyncio event loop; delegates the train step to the compute
    queue (single writer) and the eval forward to ``run_in_executor`` so the
    loop stays responsive. Returns a summary dict with per-step history.
    """
    history: list[dict[str, Any]] = []
    loop_t0 = time.time()

    # Baseline rank BEFORE any training, so the caller sees the starting point.
    import asyncio

    def _eval_under_lock() -> tuple[float, int, int]:
        with controller.state.mode_lock:
            return greedy_rank_fraction(
                controller.state.model, controller.state.tokenizer,
                prompt, response,
            )

    ev_loop = asyncio.get_running_loop()
    rank, matched, total = await ev_loop.run_in_executor(None, _eval_under_lock)
    history.append({"step": 0, "rank": rank, "matched": matched, "total": total,
                    "loss": None})

    commit_token: int | None = None
    if rank >= threshold:
        log.info("memorize: already at %.3f (>= %.3f), skipping training",
                 rank, threshold)
        return {
            "steps": 0,
            "final_rank": rank,
            "matched": matched,
            "total": total,
            "commit_token": controller.queue.committed,
            "history": history,
            "reason": "already_memorized",
            "wall_s": time.time() - loop_t0,
        }

    # Plateau detection: if ``rank`` doesn't improve for this many consecutive
    # steps, give up. Caller-tunable — a small LR + small adapter can take
    # many steps before the first argmax flip lands, so the API exposes this.
    best_rank = rank
    steps_since_improve = 0

    for step in range(1, max_steps + 1):
        spec: dict[str, Any] = {
            "objective": "weighted_sft",
            "samples": [{"prompt": prompt, "response": response,
                         "weight": weight}],
            "chunk_size": 1,
        }
        if lr is not None:
            spec["kwargs"] = {"effective_lr": lr}
        res = await controller.submit_train(spec)
        commit_token = res["commit_token"]
        # Drain the in-flight task so the weights are visible to the next eval.
        try:
            await controller.queue.wait_for(commit_token, timeout=60.0)
        except Exception as exc:
            log.warning("memorize: wait_for(%s) failed: %s", commit_token, exc)

        rank, matched, total = await ev_loop.run_in_executor(None, _eval_under_lock)
        history.append({"step": step, "rank": rank, "matched": matched,
                        "total": total, "commit_token": commit_token})

        if rank >= threshold:
            return {
                "steps": step,
                "final_rank": rank,
                "matched": matched,
                "total": total,
                "commit_token": commit_token,
                "history": history,
                "reason": "threshold_reached",
                "wall_s": time.time() - loop_t0,
            }

        if rank > best_rank + 1e-6:
            best_rank = rank
            steps_since_improve = 0
        else:
            steps_since_improve += 1
            if steps_since_improve >= plateau_patience:
                return {
                    "steps": step,
                    "final_rank": rank,
                    "matched": matched,
                    "total": total,
                    "commit_token": commit_token,
                    "history": history,
                    "reason": "plateau",
                    "wall_s": time.time() - loop_t0,
                }

    return {
        "steps": max_steps,
        "final_rank": rank,
        "matched": matched,
        "total": total,
        "commit_token": commit_token,
        "history": history,
        "reason": "max_steps",
        "wall_s": time.time() - loop_t0,
    }
