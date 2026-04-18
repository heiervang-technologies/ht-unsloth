"""Concurrent-load invariant test.

Fires N interleaved /v1/chat and /v1/feedback calls against the in-memory
Controller and asserts the commit-cursor invariant holds under contention:

  - The cursor is strictly monotone across the whole run.
  - Every chat that carried an `after_commit_token` saw a cursor ≥ its token
    when it returned.
  - No deadlocks: the whole run completes within a wall-time bound.
  - The trajectory log contains every train_step and every inference event.

Run with: python -m lile.tests.test_concurrent_load
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import tempfile
import time

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import unsloth  # noqa: F401

from lile.config import ServeConfig
from lile.controller import Controller


async def _chat(ctl: Controller, i: int, token: int | None) -> tuple[int, int, float]:
    t0 = time.time()
    result = await ctl.generate(
        [{"role": "user", "content": f"Fact #{i}:"}],
        max_new_tokens=6, temperature=0.3,
        after_commit_token=token,
    )
    # Return the cursor at the moment the call completed.
    cursor = ctl.queue.committed
    assert result.get("response_id"), result
    return i, cursor, time.time() - t0


async def _train(ctl: Controller, i: int) -> int:
    spec = {
        "objective": "sft",
        "chunk_size": 1,
        "samples": [
            {"prompt": f"Fact #{i}:",
             "response": f" Answer_{i} is the canonical reply."},
        ],
    }
    submit = await ctl.submit_train(spec)
    return submit["commit_token"]


async def main_async() -> int:
    data_dir = pathlib.Path(tempfile.mkdtemp(prefix="lile_concurrent_"))
    cfg = ServeConfig(
        model="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024, lora_rank=8, lora_alpha=16,
        data_dir=data_dir,
    )
    ctl = Controller(cfg)
    print("[concurrent] loading controller …")
    await ctl.start()
    t_start = time.time()
    try:
        n_trains = 6
        n_chats = 10
        print(f"[concurrent] launching {n_trains} trains + {n_chats} chats interleaved")

        # Submit trains first so tokens exist to wait on.
        train_tokens: list[int] = []
        for i in range(n_trains):
            tok = await _train(ctl, i)
            train_tokens.append(tok)

        # Fire chats. Some carry an after_commit_token, some don't.
        chat_tasks: list[asyncio.Task] = []
        for j in range(n_chats):
            tok = train_tokens[j % n_trains] if j % 2 == 0 else None
            chat_tasks.append(asyncio.create_task(_chat(ctl, j, tok)))

        # And one more round of trains interleaved to stress the queue.
        extra_train_tasks = [
            asyncio.create_task(_train(ctl, i + n_trains))
            for i in range(4)
        ]

        chat_results = await asyncio.gather(*chat_tasks)
        extra_train_tokens = await asyncio.gather(*extra_train_tasks)
        train_tokens.extend(extra_train_tokens)

        wall = time.time() - t_start

        # Invariants.
        # 1) Strict monotone tokens.
        all_tokens = sorted(train_tokens)
        assert all_tokens == list(sorted(set(all_tokens))), \
            f"duplicate commit tokens: {train_tokens}"
        assert all_tokens == list(range(all_tokens[0], all_tokens[-1] + 1)), \
            f"non-contiguous tokens: {all_tokens}"

        # 2) Every chat with after_commit_token got a cursor >= that token.
        for j, cursor, latency in chat_results:
            if j % 2 == 0:
                want = train_tokens[j % n_trains]
                assert cursor >= want, \
                    f"chat {j} cursor={cursor} but needed >= {want}"

        # 3) Final cursor should have advanced to cover all submitted trains.
        assert ctl.queue.committed >= max(train_tokens), \
            f"final cursor {ctl.queue.committed} < max token {max(train_tokens)}"

        # 4) Wall time bound. 10 chats + 10 trains on Qwen3-0.6B should comfortably
        # finish in <180 s; allow 300 s for noisy CI.
        assert wall < 300.0, f"concurrent run took {wall:.1f}s — deadlock suspicion"

        # 5) Trajectory has every train + every inference.
        events = list(ctl.trajectory.iter_events())
        train_events = [e for e in events if e["kind"] == "train_step"]
        infer_events = [e for e in events if e["kind"] == "inference"]
        assert len(train_events) == len(train_tokens), \
            f"expected {len(train_tokens)} train_step events, got {len(train_events)}"
        assert len(infer_events) == n_chats, \
            f"expected {n_chats} inference events, got {len(infer_events)}"

        print(f"[concurrent] OK — {n_chats} chats + {len(train_tokens)} trains in {wall:.1f}s; "
              f"final cursor={ctl.queue.committed}")
        # Observed latencies (mostly useful as an FYI).
        max_lat = max(r[2] for r in chat_results)
        mean_lat = sum(r[2] for r in chat_results) / len(chat_results)
        print(f"[concurrent] chat latency max={max_lat:.2f}s mean={mean_lat:.2f}s")
        return 0
    finally:
        await ctl.stop()


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
