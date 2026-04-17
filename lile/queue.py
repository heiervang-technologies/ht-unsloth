"""Compute queue with monotonic commit cursor.

Implements the §3.4 contract: a training batch posted to ``/v1/train`` returns a
``commit_token``. Any inference request that arrives *after* the train POST returned
must observe the model state in which that batch is included. We achieve this by:

1. Each enqueue returns ``commit_token = next_seq``.
2. The training worker drains the queue in seq order and updates ``committed_seq``
   strictly after the model state has actually been mutated.
3. Inference dispatch records the current ``committed_seq`` as a barrier; if a
   client wants stronger guarantees ("see this commit_token") it can pass it as
   ``wait_for`` and the dispatch blocks until satisfied.

The queue is in-process; for multi-process variants, replace the ``threading``
primitives with ``multiprocessing`` equivalents. The interface stays identical.

Test invariant (see ``tests/test_queue.py``): if A is submitted before B, ``A.seq <
B.seq`` and ``committed_seq`` is monotonically non-decreasing.
"""

from __future__ import annotations

import queue as _q
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class _QueueItem:
    """Internal queue item; ``seq`` is the only sort key (FIFO by submission)."""

    seq: int
    payload: Any = field(compare=False)


@dataclass(frozen=True)
class CommitToken:
    """Returned to the client by ``enqueue``; opaque otherwise."""

    seq: int

    def __int__(self) -> int:
        return self.seq


class ComputeQueue:
    """Single-writer, multi-reader compute queue with a monotonic commit cursor.

    The contract:
        * ``enqueue(payload)`` returns immediately with a ``CommitToken``.
        * The single ``ComputeWorker`` (run on a dedicated thread) pulls items in
          submission order, runs ``handler(payload)``, and only after a successful
          handler return advances ``committed_seq`` to the item's seq.
        * ``wait_for_commit(token, timeout)`` blocks until ``committed_seq >= token.seq``.

    Exceptions in the handler do *not* advance the cursor — failed steps must be
    surfaced to the client via the future returned by ``enqueue``. (We expose the
    future on the returned token via ``_future`` for that purpose.)
    """

    def __init__(self) -> None:
        self._q: _q.PriorityQueue[_QueueItem] = _q.PriorityQueue()
        self._next_seq = 1  # commit tokens are 1-indexed; 0 == "no commits yet"
        self._committed_seq = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # Per-seq future for handler results (errors or values).
        self._futures: dict[int, "_Future"] = {}
        # Idle bookkeeping for the T4.1 replay scheduler. Set on each enqueue;
        # is_idle_for() reports True iff the queue is empty AND no enqueue has
        # landed in the last `seconds`.
        self._last_enqueue_ts = time.monotonic()

    # --- producer side -------------------------------------------------

    def enqueue(self, payload: Any) -> tuple[CommitToken, "_Future"]:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            fut = _Future()
            self._futures[seq] = fut
            self._last_enqueue_ts = time.monotonic()
        self._q.put(_QueueItem(seq=seq, payload=payload))
        return CommitToken(seq=seq), fut

    def is_idle_for(self, seconds: float) -> bool:
        """True iff queue is empty AND no enqueue in the last ``seconds``.

        Used by :class:`lile.engine.replay.IdleReplayScheduler` to fence its own
        enqueues against live traffic; when a chat or feedback POST arrives the
        scheduler immediately backs off.
        """
        with self._lock:
            if self._q.qsize() > 0:
                return False
            return (time.monotonic() - self._last_enqueue_ts) >= seconds

    # --- consumer side -------------------------------------------------

    def drain_one(self, handler: Callable[[Any], Any], timeout: float | None = None) -> bool:
        """Pop one item and run ``handler`` synchronously on the calling thread.

        Returns True if an item was processed, False on timeout.
        """
        try:
            item = self._q.get(timeout=timeout)
        except _q.Empty:
            return False
        try:
            result = handler(item.payload)
        except BaseException as e:  # noqa: BLE001  — propagate to the future
            with self._lock:
                fut = self._futures.pop(item.seq, None)
                # The handler failed; do *not* advance committed_seq for this seq.
                # Subsequent items can still commit; this item's failure will be
                # visible via its future.
                if fut is not None:
                    fut.set_exception(e)
                self._cond.notify_all()
            return True
        with self._lock:
            # Strictly increase; out-of-order items would only happen if multiple
            # workers raced, which we forbid by design (single worker thread).
            assert item.seq > self._committed_seq, (
                f"out-of-order commit: {item.seq=} <= {self._committed_seq=}"
            )
            self._committed_seq = item.seq
            fut = self._futures.pop(item.seq, None)
            if fut is not None:
                fut.set_result(result)
            self._cond.notify_all()
        return True

    # --- observers ------------------------------------------------------

    @property
    def committed_seq(self) -> int:
        with self._lock:
            return self._committed_seq

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def wait_for_commit(self, token: CommitToken, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._committed_seq < token.seq:
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=remaining)
            return True


class _Future:
    """Tiny single-use threading future. Avoids the asyncio import for sync code."""

    __slots__ = ("_lock", "_cond", "_done", "_result", "_exc")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._done = False
        self._result: Any = None
        self._exc: BaseException | None = None

    def set_result(self, value: Any) -> None:
        with self._cond:
            self._result = value
            self._done = True
            self._cond.notify_all()

    def set_exception(self, exc: BaseException) -> None:
        with self._cond:
            self._exc = exc
            self._done = True
            self._cond.notify_all()

    def result(self, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._done:
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("future not ready")
                    self._cond.wait(timeout=remaining)
            if self._exc is not None:
                raise self._exc
            return self._result


class ComputeWorker(threading.Thread):
    """Single dedicated thread draining the queue."""

    def __init__(self, q: ComputeQueue, handler: Callable[[Any], Any], name: str = "lile-compute"):
        super().__init__(name=name, daemon=True)
        self._q = q
        self._handler = handler
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self._q.drain_one(self._handler, timeout=0.1)

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        self.join(timeout=timeout)
