"""Commit-cursor event broadcaster for /v1/commits/stream.

Extracted from ``Controller`` so the SSE mechanics stay torchless-testable:
``lile.controller`` pulls in ``torch`` via ``TrainEngine``, but the fan-out
primitive itself is pure asyncio — the tests should not require a GPU.

One ``CommitBroadcaster`` instance per ``Controller``. Subscribers are
bounded ``asyncio.Queue`` instances — when a subscriber's consumer falls
behind its queue fills, and further events drop silently (counted in
``drops``). Training throughput must never back-pressure on a slow client.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any


def _iso_now_ms() -> str:
    """ISO 8601 UTC with millisecond precision and trailing Z.

    The spec pins the ``ts`` field to ISO 8601 UTC; ``datetime.isoformat()``
    emits microseconds with ``+00:00``, neither of which match the example
    in the spec. Slice to ms + ``Z`` so client parsers (including the RLAIF
    loop) can rely on a stable shape.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class CommitBroadcaster:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.drops: int = 0
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    # -- subscription lifecycle ------------------------------------------------
    def subscribe(self, *, maxsize: int = 256) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # -- fan-out ---------------------------------------------------------------
    def broadcast_commit(
        self, *, cursor: int, objective: str, loss: float,
        components: dict[str, Any], batch_size: int,
    ) -> None:
        """Fan-out a commit event. Must be called synchronously from the
        queue worker — ``put_nowait`` only *schedules* wakeups, so subscribers
        cannot resume until after the queue's finally block advances the
        commit cursor. That is what preserves the ``cursor=N in event ⇒
        committed >= N`` invariant.
        """
        if not self.enabled or not self._subscribers:
            return
        event = {
            "cursor": cursor,
            "ts": _iso_now_ms(),
            "objective": objective,
            "loss": loss,
            "components": components,
            "batch_size": batch_size,
        }
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self.drops += 1

    def broadcast_shutdown(self) -> None:
        """Fire the terminal sentinel to every live subscriber.

        Consumers recognise ``{"_shutdown": True}`` and emit the
        ``event: shutdown`` frame before closing their stream. Best-effort
        on ``QueueFull`` — the consumer is stuck anyway and the socket
        closes on lifespan teardown; no drop counter bump (shutdown isn't
        a steady-state loss signal).
        """
        for q in list(self._subscribers):
            try:
                q.put_nowait({"_shutdown": True})
            except asyncio.QueueFull:
                pass
