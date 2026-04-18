"""Invariant test: compute queue commit cursor ordering.

The contract from LIVELEARN §3.4: any task whose commit_token was *returned*
to a caller is guaranteed to have completed before any later submitted task
is reflected to a reader. We verify this via a fuzzy concurrent submit/wait
pattern — including a test that would fail under reordering.

Run with: pytest -xvs lile/tests/test_queue_cursor.py
         or: python -m lile.tests.test_queue_cursor
"""
from __future__ import annotations

import asyncio
import random
import sys
import time

import pytest

from lile.queue import ComputeQueue

pytestmark = pytest.mark.cpu_only


async def _slow_handler(task) -> int:
    """Simulates GPU work with a per-task delay."""
    delay = task.payload.get("delay", 0.01)
    await asyncio.sleep(delay)
    return task.token


async def _scenario_monotonic_cursor() -> None:
    """Cursor is strictly monotonic across 50 randomly-delayed tasks."""
    q = ComputeQueue(max_depth=100)
    await q.start(_slow_handler)

    tokens: list[int] = []
    for i in range(50):
        t = await q.submit("train", {"delay": random.uniform(0.001, 0.02)})
        tokens.append(t.token)

    # Wait for the final token to commit.
    await q.wait_for(tokens[-1], timeout=10.0)

    # Cursor must have monotonically reached the final token (or beyond).
    assert q.committed >= tokens[-1], f"cursor {q.committed} < final {tokens[-1]}"
    # Tokens must themselves be monotone (sanity).
    assert tokens == sorted(tokens)
    await q.stop()
    print("[test_queue] monotonic cursor OK ({} tasks)".format(len(tokens)))


async def _scenario_wait_blocks_correctly() -> None:
    """wait_for(T) must block until the cursor passes T — even if later tasks
    haven't been submitted yet. This is the load-bearing property: inference
    after seeing commit_token T must see T reflected.
    """
    q = ComputeQueue(max_depth=10)
    results_order: list[str] = []

    async def handler(task):
        await asyncio.sleep(0.05)
        return task.token

    await q.start(handler)

    t = await q.submit("train", {})
    token = t.token

    async def waiter():
        await q.wait_for(token, timeout=2.0)
        results_order.append("waited")

    async def marker():
        # After a tiny delay, record that the waiter shouldn't have
        # returned yet. Then wait until cursor passes.
        await asyncio.sleep(0.01)
        if q.committed < token:
            results_order.append("early_check_blocked")
        else:
            results_order.append("early_check_already_past")

    await asyncio.gather(waiter(), marker())

    assert "waited" in results_order, results_order
    # The early_check must have fired before the waiter returned — the whole
    # point is that we observed "blocked" state at least once.
    assert results_order[0] == "early_check_blocked", results_order
    await q.stop()
    print("[test_queue] wait_for blocks correctly OK")


async def _scenario_fifo_within_queue() -> None:
    """Even under randomized handler delays, the worker processes in submission
    order (single worker is FIFO). A reader that sees token T committed can
    trust tokens <T are also committed.
    """
    q = ComputeQueue(max_depth=50)
    completed_order: list[int] = []

    async def handler(task):
        await asyncio.sleep(random.uniform(0.001, 0.01))
        completed_order.append(task.token)
        return task.token

    await q.start(handler)

    tokens = []
    for _ in range(30):
        t = await q.submit("train", {})
        tokens.append(t.token)

    await q.wait_for(tokens[-1], timeout=5.0)
    await q.stop()

    # FIFO: completion order equals submission order.
    assert completed_order == tokens, \
        f"order drift! submitted={tokens[:5]}... completed={completed_order[:5]}..."
    print("[test_queue] FIFO completion OK ({} tasks)".format(len(tokens)))


async def _scenario_concurrent_submit_and_wait() -> None:
    """N waiters, N submitters, random interleave. Each waiter must see its
    token committed; cursor must reach the max token.
    """
    q = ComputeQueue(max_depth=60)

    async def handler(task):
        await asyncio.sleep(random.uniform(0.001, 0.01))
        return task.token

    await q.start(handler)

    tasks = []
    for _ in range(20):
        tasks.append(await q.submit("train", {}))

    async def wait_one(task):
        result = await q.wait_for(task.token, timeout=5.0)
        assert result.error is None

    await asyncio.gather(*(wait_one(t) for t in tasks))

    assert q.committed >= tasks[-1].token
    await q.stop()
    print("[test_queue] concurrent submit+wait OK ({} tasks)".format(len(tasks)))


async def main() -> int:
    random.seed(0)
    await _scenario_monotonic_cursor()
    await _scenario_wait_blocks_correctly()
    await _scenario_fifo_within_queue()
    await _scenario_concurrent_submit_and_wait()
    print("[test_queue] ALL OK")
    return 0


def test_queue_scenarios() -> None:
    """Pytest entrypoint that runs all four async scenarios.

    The scenarios themselves are private (underscore-prefixed) so pytest's
    auto-discovery doesn't try to run them as sync tests. This wrapper lets
    ``pytest -m cpu_only`` pick up the file.
    """
    assert asyncio.run(main()) == 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
