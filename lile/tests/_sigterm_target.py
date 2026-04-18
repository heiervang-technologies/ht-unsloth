"""Subprocess target for ``test_sigterm_lifespan.py``.

Runs a FastAPI app with the same lifespan wiring as :mod:`lile.server` but
with a stub Controller that never loads a model. When lifespan exits (SIGTERM
from the parent test), the stub writes a sentinel file so the parent can
assert that ``graceful_shutdown`` was invoked with the expected arguments.

Driven entirely by env vars so the parent test controls everything:

- ``LILE_SIGTERM_SENTINEL`` — path to write on graceful_shutdown
- ``LILE_SIGTERM_PORT`` — port to bind
- ``LILE_SIGTERM_DEADLINE`` — ``shutdown_deadline_s`` to pass (default 5.0)
- ``LILE_SIGTERM_GRACE`` — ``shutdown_hard_stop_grace_s`` to pass (default 5.0)
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI


class _StubController:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    async def start(self) -> None:
        pass

    async def graceful_shutdown(
        self,
        deadline_s: float = 30.0,
        *,
        hard_stop_grace_s: float = 30.0,
    ) -> dict:
        self.sentinel.write_text(
            f"graceful_shutdown deadline={deadline_s} grace={hard_stop_grace_s}"
        )
        return {
            "already_shut_down": False,
            "dropped": 0,
            "timed_out": False,
        }


def main() -> None:
    sentinel = Path(os.environ["LILE_SIGTERM_SENTINEL"])
    port = int(os.environ["LILE_SIGTERM_PORT"])
    deadline = float(os.environ.get("LILE_SIGTERM_DEADLINE", "5.0"))
    hard_stop_grace = float(os.environ.get("LILE_SIGTERM_GRACE", "5.0"))

    stub = _StubController(sentinel)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await stub.start()
        try:
            yield
        finally:
            await stub.graceful_shutdown(
                deadline_s=deadline,
                hard_stop_grace_s=hard_stop_grace,
            )

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
