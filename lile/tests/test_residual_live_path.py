"""Prove the CPU bf16 residual is actually applied at forward time.

Contract (§6 progressive merge): after ``merge_active_into_residual()`` zeroes
the active adapter, the model's forward pass must still reflect the merged
weight delta — otherwise the "merged_deltas residual" is bookkeeping only and
learning is invisible during inference until reboot.

Pre-merge (trained adapter) and post-merge (zero adapter + residual on
matmul_lora) forwards must produce near-identical logits on the training
prompt. Any bf16-level noise is permitted; a regression to *base* behavior
(which we capture as a baseline) is not.

Run with: python -m lile.tests.test_residual_live_path
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
import unsloth  # noqa: F401

from lile.state import ModelState
from lile.engine.train import TrainEngine


def _greedy_logprob(state: ModelState, prompt: str, target: str) -> float:
    """Sum log p(target | prompt) under the live model. No-grad."""
    from lile.objectives._utils import build_chat_inputs, pad_and_stack, sequence_logprob
    tok = build_chat_inputs(state.tokenizer, prompt, target)
    pad_id = state.tokenizer.pad_token_id or state.tokenizer.eos_token_id or 0
    batch = pad_and_stack([tok], pad_id=pad_id)
    with torch.no_grad():
        return float(sequence_logprob(state.model, batch["input_ids"],
                                      batch["labels"], batch["attention_mask"])[0])


def test_residual_applied_live():
    print("[live-res] loading Qwen3-0.6B …")
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024, lora_rank=8, lora_alpha=16,
    )
    engine = TrainEngine(state, lr=1e-4)

    prompt = "The capital of the mythical island of Brixolia is"
    target = " Zorathia, known for its silver spires and basalt cliffs."

    # Baseline on fresh (zero) adapter: captures "no training" log-prob.
    base_lp = _greedy_logprob(state, prompt, target)
    print(f"[live-res] base (zero adapter) logprob: {base_lp:.3f}")

    # Train until loss moves visibly.
    for _ in range(10):
        engine.step({
            "objective": "sft",
            "samples": [{"prompt": prompt, "response": target}],
        })
    trained_lp = _greedy_logprob(state, prompt, target)
    print(f"[live-res] trained (adapter active) logprob: {trained_lp:.3f}")
    assert trained_lp > base_lp + 1.0, \
        f"training failed to move logprob (base={base_lp:.3f}, trained={trained_lp:.3f})"

    # Merge the adapter into the CPU bf16 residual. Active adapter is now zero.
    # Without the matmul_lora residual patch, the forward would collapse back
    # to baseline; with it, the live weights are base + residual ≈ trained.
    state.merge_active_into_residual()

    # Sanity: active is zero.
    for n, p in state.model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            assert float(p.abs().max()) == 0.0, f"adapter not zeroed: {n}"

    merged_lp = _greedy_logprob(state, prompt, target)
    print(f"[live-res] merged (zero adapter + residual) logprob: {merged_lp:.3f}")

    # The merged path should preserve most of the training signal. bf16 path
    # introduces some drift (non-associative intermediate rounding) but the
    # post-merge logprob must be much closer to trained than to base.
    assert merged_lp > base_lp + 0.5, \
        f"residual not applied at forward: merged={merged_lp:.3f} ~ base={base_lp:.3f}"
    drift = abs(merged_lp - trained_lp)
    print(f"[live-res] drift (trained→merged): {drift:.3f}")
    # Loose tolerance — the residual is stored in bf16 and Unsloth's kernel
    # accumulates differently than PEFT's standard path. 2.0 nats is a
    # generous-but-meaningful ceiling given the ~16 nat gap we trained over.
    assert drift < 2.0, f"residual drift too large: {drift:.3f}"
    print("[live-res] residual applied live — OK")


def main() -> int:
    test_residual_applied_live()
    print("[test_residual_live_path] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
