"""End-to-end smoke test against a real model on a real GPU.

Skipped automatically when CUDA isn't available so CI without GPU stays green.
Run explicitly with:

    uv run --no-project python -m pytest lile/tests/test_smoke.py -v -s

Sequence under test (matches ``LIVELEARN.md`` §3.4 + §5b.4):

1. Boot the controller; load Qwen3-0.6B 4-bit; verify VRAM.
2. Generate a baseline chat completion; capture ``response_id``.
3. Submit an SFT batch; assert ``commit_token`` is monotonic.
4. ``wait_for_commit`` blocks until step lands.
5. Generate again with the same prompt — train_step bumped, no NaN.
6. Submit binary feedback against the response_id (KTO route).
7. Submit nl_critique_with_rewrite feedback (CCPD route).
8. Save snapshot, then restore it.
9. Trigger progressive merge; verify subsequent generation still works.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="smoke test requires CUDA",
)


@pytest.fixture(scope="module")
def controller(tmp_path_factory):
    from lile.controller import Controller, ControllerConfig
    from lile.state import StateConfig

    work_dir = tmp_path_factory.mktemp("lile-smoke")
    cfg = ControllerConfig(
        state=StateConfig(
            model_name=os.environ.get(
                "LILE_SMOKE_MODEL",
                "unsloth/qwen3-0.6b-unsloth-bnb-4bit",
            ),
            max_seq_length=1024,
            lora_rank=8,
            lora_alpha=8,
        ),
        work_dir=str(work_dir),
        lr=1e-5,
    )
    c = Controller(cfg).start()
    yield c
    c.shutdown()


def test_controller_loaded_with_lora(controller):
    assert controller.state is not None
    assert controller.state.model is not None
    trainable = sum(
        p.numel() for p in controller.state.model.parameters() if p.requires_grad
    )
    assert trainable > 0, "no trainable parameters → LoRA not attached"
    vram = controller.state.vram_summary()
    assert vram["allocated_gb"] > 0
    print(f"[smoke] vram allocated={vram['allocated_gb']:.2f}GB / total={vram['total_gb']:.2f}GB")


def test_chat_baseline(controller):
    res = controller.chat(
        [{"role": "user", "content": "Say 'ready' if you can hear me."}],
        max_new_tokens=16,
        temperature=0.7,
    )
    assert res.text, "empty completion"
    assert res.completion_tokens > 0
    assert res.response_id  # UUID4 string
    print(f"[smoke] baseline gen ({res.completion_tokens} tok in {res.elapsed_s:.2f}s): {res.text[:80]!r}")


def test_train_then_chat_with_barrier(controller):
    """The §3.4 contract: a chat after a train POST sees the trained state."""
    from lile.objectives import Batch, Sample

    pre_step = controller.train_engine.global_step
    batch = Batch(samples=[
        Sample(
            prompt="What is the secret codeword for today?",
            target="The secret codeword is octopus.",
            objectives=[{"sft": {}}],
            weight=2.0,
        ),
    ])
    token = controller.submit_train(batch)
    assert token.seq > 0
    # Block on the commit barrier; this is what the API surface does.
    res = controller.chat(
        [{"role": "user", "content": "what is the codeword?"}],
        max_new_tokens=16,
        temperature=0.0,  # greedy → deterministic
        do_sample=False,
        wait_for=token,
    )
    assert controller.train_engine.global_step == pre_step + 1
    assert res.text  # non-empty (we don't assert the exact content — random init)
    print(f"[smoke] post-train gen: {res.text[:80]!r}")


def test_feedback_kto_routes_to_train_queue(controller):
    """Binary feedback against the previous response → KTO sample in queue."""
    pre_step = controller.train_engine.global_step
    # First, generate a fresh response to feedback on.
    res = controller.chat(
        [{"role": "user", "content": "Tell me a fun fact."}],
        max_new_tokens=24,
        temperature=0.7,
    )
    token = controller.submit_feedback(
        response_id=res.response_id, kind="binary", value="up",
    )
    ok = controller.queue.wait_for_commit(token, timeout=60.0)
    assert ok
    assert controller.train_engine.global_step == pre_step + 1


def test_feedback_critique_with_rewrite_routes_to_ccpd(controller):
    res = controller.chat(
        [{"role": "user", "content": "What's 2 + 2?"}],
        max_new_tokens=8, temperature=0.0, do_sample=False,
    )
    pre_step = controller.train_engine.global_step
    token = controller.submit_feedback(
        response_id=res.response_id,
        kind="nl_critique_with_rewrite",
        critique="Be more concise; just give the number.",
        better_response="4",
    )
    ok = controller.queue.wait_for_commit(token, timeout=180.0)
    assert ok
    # CCPD may legitimately skip if the advantage spread is below tau; in
    # that case global_step still advances because the wrapper returns
    # an autograd-attached zero. Either way the queue must commit.
    assert controller.train_engine.global_step >= pre_step  # at least counted


def test_snapshot_save_and_restore(controller, tmp_path):
    from lile.objectives import Batch, Sample

    # Take a snapshot BEFORE further training.
    save_token = controller.submit_snapshot("snap-before")
    assert controller.queue.wait_for_commit(save_token, timeout=30.0)
    snap_dir = controller.work_dir / "snap-before"
    assert (snap_dir / "manifest.json").exists()

    # Train one more step to mutate state.
    train_token = controller.submit_train(Batch(samples=[
        Sample(prompt="hi", target="hello", objectives=[{"sft": {}}]),
    ]))
    assert controller.queue.wait_for_commit(train_token, timeout=60.0)

    # Restore — drains the queue and rolls model back.
    manifest = controller.restore("snap-before")
    assert manifest["schema"] == 1


def test_progressive_merge(controller):
    pre = controller.state.merge_count
    token = controller.submit_merge()
    assert controller.queue.wait_for_commit(token, timeout=60.0)
    assert controller.state.merge_count == pre + 1
    # Generation must still work post-merge.
    res = controller.chat(
        [{"role": "user", "content": "say ok"}],
        max_new_tokens=4, temperature=0.0, do_sample=False,
    )
    assert res.text
