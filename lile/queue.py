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

from .errors import ShutdownDroppedError, ShuttingDownError

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
        # Shutdown coordination (see ``graceful_drain``):
        #  - ``_accepting`` gates ``submit``; flipped False on drain entry.
        #  - ``_hard_stop`` tells the worker loop to exit WITHOUT pulling more
        #    tasks, even if the queue is non-empty. Set on deadline expiry.
        self._accepting: bool = True
        self._hard_stop = asyncio.Event()
        # Monotonic timestamp of the last submit(). Read by is_idle_for().
        # Initialized far in the past so the very first poll after startup
        # reads as idle; the scheduler gates on "quiet for N seconds" anyway.
        self._last_enqueue_ts: float = 0.0

    # ------------------------------------------------------------------ enqueue
    async def submit(self, kind: str, payload: Any, batch_id: str = "") -> QueueTask:
        if not self._accepting:
            raise ShuttingDownError(
                "queue is draining; new submits rejected until restart",
            )
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
        while True:
            # Drain-on-stop: exit only when stop is requested AND the queue
            # has been fully processed, OR a hard-stop is signalled (deadline
            # expired — stop pulling new tasks even if some are queued).
            if self._hard_stop.is_set():
                return
            if self._stop.is_set() and self._q.empty():
                return
            try:
                task = await asyncio.wait_for(self._q.get(), timeout=0.1)
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

    async def graceful_drain(self, deadline_s: float | None = None) -> dict[str, Any]:
        """Close the queue, drain pending, resolve remainders with ShutdownDropped.

        On entry: flip ``_accepting`` so any new ``submit`` raises
        :class:`ShuttingDownError`, then ask the worker to stop as soon as
        the queue is empty.

        While ``deadline_s`` has budget, the worker keeps pulling tasks and
        running them — the drain-on-stop semantics mean a sufficiently long
        budget completes every enqueued task naturally.

        If the budget expires with the queue non-empty, we set
        ``_hard_stop``: the worker exits after the currently-in-flight task
        returns (we never cancel mid-GPU-step — that would tear the LoRA),
        and every still-pending queue entry gets ``error =
        ShutdownDroppedError`` and ``done.set()`` so every ``wait_for``
        caller resolves deterministically.

        Idempotent: a second call with nothing left to do returns
        ``{"dropped": 0, "timed_out": False}``.
        """
        self._accepting = False
        self._stop.set()
        if self._worker_task is None or self._worker_task.done():
            return await self._reap_pending(timed_out=False)

        timed_out = False
        try:
            if deadline_s is not None:
                await asyncio.wait_for(
                    asyncio.shield(self._worker_task), timeout=deadline_s,
                )
            else:
                await self._worker_task
        except asyncio.TimeoutError:
            timed_out = True
            # Ask the worker to stop pulling more tasks; give the in-flight
            # one a bit more time to finish so its state is consistent.
            self._hard_stop.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._worker_task), timeout=30.0,
                )
            except asyncio.TimeoutError:
                # In-flight task is genuinely stuck; best we can do is move
                # on and let the process teardown handle it.
                pass
        self._worker_task = None
        return await self._reap_pending(timed_out=timed_out)

    async def _reap_pending(self, *, timed_out: bool) -> dict[str, Any]:
        """Fire ``done`` on every still-pending task with ShutdownDroppedError.

        Advances ``_completed_token`` over the dropped range so any reader
        that consults the cursor after drain sees monotone progress.
        """
        dropped = 0
        async with self._cursor_advanced:
            for task in list(self._pending.values()):
                if task.done.is_set():
                    continue
                task.error = ShutdownDroppedError(
                    f"queue task {task.token} ({task.kind}) dropped on shutdown",
                )
                if task.token > self._completed_token:
                    self._completed_token = task.token
                task.done.set()
                dropped += 1
            self._cursor_advanced.notify_all()
            # GC: we've resolved every pending task one way or another.
            for t in list(self._pending.keys()):
                if t <= self._completed_token:
                    self._pending.pop(t, None)
        return {"dropped": dropped, "timed_out": timed_out}


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]
