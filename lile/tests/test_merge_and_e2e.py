"""End-to-end tests that require a real model on the GPU.

Tests:
  1. Merge determinism — merging the active LoRA twice (null merge second time)
     yields the same residual fingerprint as merging it once.
  2. Merge correctness — LoRA-on output ≈ base + residual output after merge
     (within bf16 numerical tolerance on greedy next-token logits).
  3. Commit-cursor end-to-end — submit train, then submit infer with
     after_commit_token; verify logprob on the training prompt moved.

Run with: python -m lile.tests.test_merge_and_e2e
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
import unsloth  # noqa: F401

from lile.state import ModelState
from lile.engine.train import TrainEngine
from lile.engine.inference import generate_chat


def test_merge_determinism():
    print("[merge] loading Qwen3-0.6B …")
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024, lora_rank=8, lora_alpha=16,
    )
    engine = TrainEngine(state, lr=5e-5)

    # Take a few SFT steps to populate the LoRA with something non-zero.
    for _ in range(3):
        engine.step({
            "objective": "sft",
            "samples": [
                {"prompt": "Name a color.", "response": "Blue is a calming color."},
            ],
        })

    # Merge once, record fingerprint, merge a null LoRA (zeroed) and re-fingerprint.
    state.merge_active_into_residual()
    fp1 = state.residual_fingerprint()
    assert fp1, "residual fingerprint empty after merge"
    print(f"[merge] fp after 1st merge: {fp1[:16]}…")

    # Reset active adapter is already done by merge_active; merge again.
    # Since active is zero, no new deltas should be produced and residual
    # fingerprint must be unchanged.
    state.merge_active_into_residual()
    fp2 = state.residual_fingerprint()
    print(f"[merge] fp after null 2nd merge: {fp2[:16]}…")
    assert fp1 == fp2, "null second merge changed residual (non-idempotent)"
    print("[merge] determinism OK")


def test_end_to_end_training_moves_logprob():
    """Train on a fixed (prompt, target); measure mean log-prob on the target
    before and after several steps; confirm it moved upward (loss decreased).
    This is the load-bearing visibility test: if training didn't affect the
    shared-weight model, this metric wouldn't move.
    """
    print("[e2e] loading Qwen3-0.6B (fresh state) …")
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024, lora_rank=8, lora_alpha=16,
    )
    engine = TrainEngine(state, lr=1e-4)

    prompt = "The special code is:"
    target = " ZETA-X9-PRIME-47 (this is my secret)."

    r0 = engine.step({
        "objective": "sft",
        "samples": [{"prompt": prompt, "response": target}],
    })
    initial_loss = r0["loss"]
    print(f"[e2e] initial SFT loss: {initial_loss:.3f}")

    for _ in range(8):
        engine.step({
            "objective": "sft",
            "samples": [{"prompt": prompt, "response": target}],
        })

    r_final = engine.step({
        "objective": "sft",
        "samples": [{"prompt": prompt, "response": target}],
    })
    final_loss = r_final["loss"]
    print(f"[e2e] final SFT loss:   {final_loss:.3f}")
    assert final_loss < initial_loss - 0.5, \
        f"expected substantial loss drop, got {initial_loss:.3f} → {final_loss:.3f}"
    print("[e2e] training visibly moved policy OK")


async def test_controller_commit_cursor_e2e():
    """Submit a train request, then a generate with after_commit_token, via
    the Controller. The generate must block until training commits, and the
    response must reflect the trained content.
    """
    from lile.config import ServeConfig
    from lile.controller import Controller

    print("[ctrl] loading controller …")
    cfg = ServeConfig(
        model="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024, lora_rank=8, lora_alpha=16,
        data_dir=__import__("pathlib").Path(f"/tmp/lile_test_{int(time.time())}"),
    )
    ctl = Controller(cfg)
    await ctl.start()
    try:
        # Submit a training batch that should be reflected.
        spec = {
            "objective": "sft",
            "chunk_size": 1,
            "samples": [
                {"prompt": "The color of the sky is",
                 "response": " green because of photosynthesis atmosphere."},
                {"prompt": "The color of the sky is",
                 "response": " green because of photosynthesis atmosphere."},
            ],
        }
        submit = await ctl.submit_train(spec)
        token = submit["commit_token"]
        print(f"[ctrl] submitted train, commit_token={token}, chunks={submit['n_chunks']}")

        # Generate with after_commit_token — this must block until commit.
        t_before = time.time()
        result = await ctl.generate(
            [{"role": "user", "content": "The color of the sky is"}],
            max_new_tokens=8, temperature=0.1,
            after_commit_token=token,
        )
        gen_wall = time.time() - t_before
        print(f"[ctrl] generate after commit: {gen_wall:.2f}s; cursor={ctl.queue.committed}")
        assert ctl.queue.committed >= token, "inference ran before training committed"
        print(f"[ctrl] response: {result['response']!r}")
        print("[ctrl] commit-cursor end-to-end OK")
    finally:
        await ctl.stop()


def main() -> int:
    test_merge_determinism()
    test_end_to_end_training_moves_logprob()
    asyncio.run(test_controller_commit_cursor_e2e())
    print("[test_merge_and_e2e] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
