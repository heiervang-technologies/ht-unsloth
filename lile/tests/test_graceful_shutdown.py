"""Controller-level graceful shutdown — pins the contract for #11.

Covers ``Controller.graceful_shutdown(deadline_s)``:

- flips ``_shutting_down`` (read by ``metrics.py`` for the ``lile_shutting_down``
  gauge),
- rejects new ``submit_train`` / ``submit_feedback`` / ``request_merge`` /
  ``request_snapshot_save`` / ``request_snapshot_load`` calls with
  :class:`ShuttingDownError`,
- stops the idle replay scheduler and closes the metrics logger,
- delegates queue drain to :meth:`ComputeQueue.graceful_drain`, whose
  contract is pinned in ``test_queue_graceful_drain.py``.

These tests build a Controller with the GPU-loading bits stubbed out — the
``ModelState`` is never loaded, the train engine is never instantiated — so
they stay cpu_only. The shutdown logic is all in the control-plane path.
"""
from __future__ import annotations

import asyncio
import pathlib
import tempfile
from typing import Any

import pytest

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------- fixtures


def _cfg(tmp_path: pathlib.Path):
    from lile.config import ServeConfig

    return ServeConfig(data_dir=tmp_path, max_queue_depth=8)


def _bare_controller(cfg):
    """Controller with queue wired but model/state stubbed — no GPU load."""
    from lile.controller import Controller

    c = Controller(cfg)
    # Short-circuit the GPU path: tests never call generate / train.
    c.state = None
    c.train_engine = None
    return c


# ---------------------------------------------------------------- tests


def test_shutting_down_flag_starts_false(tmp_path):
    with tempfile.TemporaryDirectory(dir=tmp_path) as d:
        cfg = _cfg(pathlib.Path(d))
        c = _bare_controller(cfg)
        assert c._shutting_down is False


def test_graceful_shutdown_flips_flag_and_drains_queue(tmp_path):
    async def main():
        from lile.queue import ComputeQueue

        with tempfile.TemporaryDirectory(dir=tmp_path) as d:
            cfg = _cfg(pathlib.Path(d))
            c = _bare_controller(cfg)

            # Start the queue with a no-op handler (bypass Controller._handle_task
            # which would need a real ModelState).
            async def _handler(task):
                return {"ok": True}

            await c.queue.start(_handler)
            t1 = await c.queue.submit("custom", {})
            t2 = await c.queue.submit("custom", {})

            await c.graceful_shutdown(deadline_s=2.0)

            assert c._shutting_down is True
            assert t1.done.is_set() and t1.error is None
            assert t2.done.is_set() and t2.error is None

    asyncio.run(main())


def test_submit_train_rejected_after_shutdown(tmp_path):
    async def main():
        from lile.errors import ShuttingDownError

        with tempfile.TemporaryDirectory(dir=tmp_path) as d:
            cfg = _cfg(pathlib.Path(d))
            c = _bare_controller(cfg)

            async def _handler(task):
                return {"ok": True}

            await c.queue.start(_handler)
            await c.graceful_shutdown(deadline_s=1.0)

            with pytest.raises(ShuttingDownError):
                await c.submit_train({"objective": "sft", "samples": [{"prompt": "x", "response": "y"}]})

    asyncio.run(main())


def test_submit_feedback_rejected_after_shutdown(tmp_path):
    async def main():
        from lile.errors import ShuttingDownError

        with tempfile.TemporaryDirectory(dir=tmp_path) as d:
            cfg = _cfg(pathlib.Path(d))
            c = _bare_controller(cfg)

            async def _handler(task):
                return {"ok": True}

            await c.queue.start(_handler)
            await c.graceful_shutdown(deadline_s=1.0)

            with pytest.raises(ShuttingDownError):
                await c.submit_feedback({
                    "kind": "rewrite", "prompt": "x", "response": "y",
                    "better_response": "z",
                })

    asyncio.run(main())


def test_state_ops_rejected_after_shutdown(tmp_path):
    async def main():
        from lile.errors import ShuttingDownError

        with tempfile.TemporaryDirectory(dir=tmp_path) as d:
            cfg = _cfg(pathlib.Path(d))
            c = _bare_controller(cfg)

            async def _handler(task):
                return {"ok": True}

            await c.queue.start(_handler)
            await c.graceful_shutdown(deadline_s=1.0)

            with pytest.raises(ShuttingDownError):
                await c.request_merge()
            with pytest.raises(ShuttingDownError):
                await c.request_snapshot_save("foo")
            with pytest.raises(ShuttingDownError):
                await c.request_snapshot_load("foo")

    asyncio.run(main())


def test_graceful_shutdown_is_idempotent(tmp_path):
    async def main():
        with tempfile.TemporaryDirectory(dir=tmp_path) as d:
            cfg = _cfg(pathlib.Path(d))
            c = _bare_controller(cfg)

            async def _handler(task):
                return {"ok": True}

            await c.queue.start(_handler)
            r1 = await c.graceful_shutdown(deadline_s=1.0)
            r2 = await c.graceful_shutdown(deadline_s=1.0)
            assert r1["already_shut_down"] is False
            assert r2["already_shut_down"] is True

    asyncio.run(main())


def test_in_flight_pending_tasks_resolve_with_shutdown_dropped(tmp_path):
    """Waiters on unpulled tasks get ``ShutdownDroppedError`` (not a hang)."""
    async def main():
        from lile.errors import ShutdownDroppedError

        with tempfile.TemporaryDirectory(dir=tmp_path) as d:
            cfg = _cfg(pathlib.Path(d))
            c = _bare_controller(cfg)

            async def _slow_handler(task):
                await asyncio.sleep(task.payload["sleep_s"])
                return {"ok": True}

            await c.queue.start(_slow_handler)
            tasks = [
                await c.queue.submit("custom", {"sleep_s": 0.5}) for _ in range(3)
            ]
            await c.graceful_shutdown(deadline_s=0.1)
            dropped = [t for t in tasks if isinstance(t.error, ShutdownDroppedError)]
            assert len(dropped) >= 1
            # Every task must be resolved — no pending done events.
            for t in tasks:
                assert t.done.is_set()

    asyncio.run(main())


# ---------------------------------------------------------------- server.py


def test_server_wires_lifespan(tmp_path):
    """Startup + shutdown are handled by a FastAPI lifespan context manager,
    which calls ``controller.graceful_shutdown`` on exit. We enumerate the
    router config without spinning up a Controller or a model (behavior is
    covered by the Controller-level tests above)."""
    from lile.config import ServeConfig
    from lile.server import create_app

    cfg = ServeConfig(data_dir=tmp_path)
    app = create_app(cfg)
    # Lifespan replaces the legacy on_startup/on_shutdown callbacks, so the
    # router's event-handler lists should be empty.
    assert app.router.on_startup == []
    assert app.router.on_shutdown == []
    # The lifespan context manager is bound on the router.
    assert app.router.lifespan_context is not None
