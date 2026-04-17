"""Compute queue with commit cursor.

The invariant from LIVELEARN §3.4:

  Any inference request that arrives *after* the server returned a commit_token
  for a training request must see that training reflected in the model.

Implementation: monotonic integer cursor, single writer (the training worker),
readers block on `wait_for(token)` until the worker's completed-cursor passes
their token. This is a semaphore, not a best-effort signal, and the test in
tests/test_queue_cursor.py is written to fail under reordering.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class QueueTask:
    token: int
    kind: str                          # "train" | "merge" | "snapshot" | "custom"
    payload: Any
    created_at: float = field(default_factory=time.time)
    # Set by the worker when done; inference waits on this.
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None
    result: Any = None

    # Opaque batch_id lets callers group tasks for a single /v1/train call and
    # get ONE commit_token that covers all sub-batches.
    batch_id: str = ""


class ComputeQueue:
    """Single-writer async queue with a monotonic commit cursor.

    The cursor advances *only when a task completes successfully*. Failures
    advance it too (the task is "done"), but the task's `error` is set and
    readers can inspect it — a failed train step still has a deterministic
    ordering with respect to subsequent inference.
    """

    def __init__(self, max_depth: int = 64) -> None:
        self._q: asyncio.Queue[QueueTask] = asyncio.Queue(maxsize=max_depth)
        self._next_token: int = 0
        self._completed_token: int = -1
        # Maps token -> completion event. Kept small by GC after wait_for.
        self._pending: dict[int, QueueTask] = {}
        self._cursor_advanced = asyncio.Condition()
        self._worker_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Monotonic timestamp of the last submit(). Read by is_idle_for().
        # Initialized far in the past so the very first poll after startup
        # reads as idle; the scheduler gates on "quiet for N seconds" anyway.
        self._last_enqueue_ts: float = 0.0

    # ------------------------------------------------------------------ enqueue
    async def submit(self, kind: str, payload: Any, batch_id: str = "") -> QueueTask:
        token = self._next_token
        self._next_token += 1
        task = QueueTask(token=token, kind=kind, payload=payload, batch_id=batch_id)
        self._pending[token] = task
        self._last_enqueue_ts = time.monotonic()
        await self._q.put(task)
        return task

    # ------------------------------------------------------------------ idleness
    def is_idle_for(self, seconds: float) -> bool:
        """True iff the queue is empty AND no submit happened in the last N seconds.

        Consulted by the idle replay scheduler (§T4.1) to avoid contending with
        live train/infer work. Single-reader pattern: the scheduler polls this
        at a coarse cadence, so we don't need a lock — `asyncio.Queue.empty()`
        is safe to call from any coroutine, and the monotonic float read is
        atomic on CPython.
        """
        if not self._q.empty():
            return False
        return (time.monotonic() - self._last_enqueue_ts) >= seconds

    # ------------------------------------------------------------------ read-side
    @property
    def committed(self) -> int:
        return self._completed_token

    async def wait_for(self, token: int, timeout: float | None = None) -> QueueTask:
        """Block until the committed cursor >= token."""
        task = self._pending.get(token)
        if task is None and token <= self._completed_token:
            # Already completed and GC'd; that's a valid "seen" state.
            return QueueTask(token=token, kind="n/a", payload=None)
        if task is None:
            raise KeyError(f"unknown token {token}")
        if timeout is None:
            await task.done.wait()
        else:
            await asyncio.wait_for(task.done.wait(), timeout=timeout)
        return task

    # ------------------------------------------------------------------ worker
    async def start(self, handler: Callable[[QueueTask], Any]) -> None:
        """Start the single worker task. `handler` is a coroutine-or-callable."""
        if self._worker_task is not None:
            raise RuntimeError("queue already started")
        self._worker_task = asyncio.create_task(self._run(handler))

    async def _run(self, handler: Callable[[QueueTask], Any]) -> None:
        log.info("compute queue worker started")
        while not self._stop.is_set():
            try:
                task = await asyncio.wait_for(self._q.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            try:
                result = handler(task)
                if asyncio.iscoroutine(result):
                    result = await result
                task.result = result
            except BaseException as e:
                log.exception("queue task %d (%s) failed", task.token, task.kind)
                task.error = e
            finally:
                async with self._cursor_advanced:
                    # Cursor strictly monotone; cover out-of-order completions
                    # defensively even though our worker is single-threaded.
                    if task.token > self._completed_token:
                        self._completed_token = task.token
                    task.done.set()
                    self._cursor_advanced.notify_all()
                    # GC: drop once cursor has passed.
                    for t in list(self._pending.keys()):
                        if t <= self._completed_token:
                            self._pending.pop(t, None)

    async def stop(self) -> None:
        self._stop.set()
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]
