"""Tests for :mod:`lile.queue` — the §3.4 commit-cursor invariant.

Per ``DESIGN.md`` §8, this is the test the design doc explicitly committed to:
the cursor must advance monotonically and ``wait_for_commit`` must release in
strict order. If you ever switch the worker from single-thread to a pool, this
file is what fails first.
"""

from __future__ import annotations

import threading
import time

import pytest

from lile.queue import CommitToken, ComputeQueue, ComputeWorker


def test_enqueue_returns_monotonic_tokens():
    q = ComputeQueue()
    tokens = [q.enqueue(i)[0] for i in range(10)]
    seqs = [t.seq for t in tokens]
    assert seqs == list(range(1, 11))


def test_drain_one_advances_committed_seq():
    q = ComputeQueue()
    q.enqueue("a")
    q.enqueue("b")
    assert q.committed_seq == 0
    assert q.drain_one(handler=lambda x: x.upper()) is True
    assert q.committed_seq == 1
    assert q.drain_one(handler=lambda x: x.upper()) is True
    assert q.committed_seq == 2


def test_drain_one_preserves_fifo_under_worker():
    q = ComputeQueue()
    seen: list[int] = []
    handler = lambda x: seen.append(x)  # noqa: E731
    tokens = [q.enqueue(i)[0] for i in range(50)]
    worker = ComputeWorker(q, handler=handler)
    worker.start()
    try:
        # Wait for the last token to commit.
        assert q.wait_for_commit(tokens[-1], timeout=5.0)
    finally:
        worker.stop(timeout=2.0)
    assert seen == list(range(50)), "handler must see items in submission order"
    assert q.committed_seq == 50


def test_wait_for_commit_blocks_then_releases():
    q = ComputeQueue()
    handler_can_proceed = threading.Event()

    def slow_handler(payload):
        handler_can_proceed.wait(timeout=2.0)
        return payload

    token, fut = q.enqueue("x")
    worker = ComputeWorker(q, handler=slow_handler)
    worker.start()
    try:
        # The handler blocks → wait_for_commit must time out.
        assert q.wait_for_commit(token, timeout=0.2) is False
        # Release the handler → wait_for_commit returns True.
        handler_can_proceed.set()
        assert q.wait_for_commit(token, timeout=2.0) is True
        assert fut.result(timeout=1.0) == "x"
    finally:
        worker.stop(timeout=2.0)


def test_handler_exception_does_not_advance_cursor_for_that_seq():
    q = ComputeQueue()
    boom_token, boom_fut = q.enqueue("boom")
    ok_token, ok_fut = q.enqueue("ok")

    def handler(p):
        if p == "boom":
            raise ValueError("nope")
        return p.upper()

    # Drain both items synchronously.
    assert q.drain_one(handler) is True
    # The failed item's future must reflect the exception …
    with pytest.raises(ValueError, match="nope"):
        boom_fut.result(timeout=0.5)
    # … but the cursor must NOT have advanced for the failed seq.
    assert q.committed_seq == 0

    # The next item must commit its own seq, skipping the failed one in the cursor.
    assert q.drain_one(handler) is True
    assert q.committed_seq == ok_token.seq
    assert ok_fut.result(timeout=0.5) == "OK"


def test_wait_for_already_committed_returns_immediately():
    q = ComputeQueue()
    token, _ = q.enqueue("done")
    q.drain_one(handler=lambda x: x)
    # Already committed; should not block.
    t0 = time.monotonic()
    assert q.wait_for_commit(token, timeout=0.1) is True
    assert time.monotonic() - t0 < 0.05


def test_committed_seq_assert_on_out_of_order_would_fire():
    """White-box test: if a second worker tried to commit a stale seq, the
    assert in :meth:`ComputeQueue.drain_one` would fire. We construct the
    state manually rather than racing two threads (which is flaky in CI)."""
    q = ComputeQueue()
    q._committed_seq = 5  # simulate that seq 5 is already committed.

    # Now feed a "seq 3" item directly to drain_one. We bypass enqueue's seq
    # allocation by injecting into the underlying PriorityQueue.
    from lile.queue import _QueueItem

    q._q.put(_QueueItem(seq=3, payload="stale"))

    with pytest.raises(AssertionError, match="out-of-order commit"):
        q.drain_one(handler=lambda x: x)


def test_concurrent_producers_get_unique_seqs():
    """Many threads enqueue at once; every seq must be unique and contiguous."""
    q = ComputeQueue()
    n_threads = 16
    n_per_thread = 50
    tokens: list[CommitToken] = []
    lock = threading.Lock()

    def producer():
        local: list[CommitToken] = []
        for _ in range(n_per_thread):
            t, _ = q.enqueue(None)
            local.append(t)
        with lock:
            tokens.extend(local)

    threads = [threading.Thread(target=producer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    seqs = sorted(t.seq for t in tokens)
    assert seqs == list(range(1, n_threads * n_per_thread + 1))
