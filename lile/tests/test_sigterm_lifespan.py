"""End-to-end SIGTERM → lifespan → graceful_shutdown test.

Pins the signal-delivery contract flagged as missing in #28 review: uvicorn
must translate SIGTERM into a clean lifespan exit, and that exit must await
:meth:`Controller.graceful_shutdown` with the configured deadline + grace.

Uses a subprocess target (``_sigterm_target.py``) that stubs the Controller
so we don't load a GPU model in a cpu_only test. The target writes a sentinel
file when ``graceful_shutdown`` fires; the test asserts it ran with the
expected kwargs and that the subprocess exited cleanly (rc==0).
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.cpu_only


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthy(port: int, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.1)
    raise TimeoutError(
        f"daemon on port {port} did not become healthy (last: {last_err!r})"
    )


def test_sigterm_fires_graceful_shutdown(tmp_path):
    sentinel = tmp_path / "sentinel"
    port = _free_port()
    env = dict(
        os.environ,
        LILE_SIGTERM_SENTINEL=str(sentinel),
        LILE_SIGTERM_PORT=str(port),
        LILE_SIGTERM_DEADLINE="5.0",
        LILE_SIGTERM_GRACE="3.0",
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "lile.tests._sigterm_target"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_healthy(port)
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=15.0)
        out, err = proc.communicate(timeout=1.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    # The sentinel content is the real contract — it proves the lifespan ran
    # graceful_shutdown with the expected kwargs. uvicorn's default signal
    # handler propagates SIGTERM back to the process after the lifespan
    # completes (rc == -SIGTERM), so accept either disposition.
    assert rc in (0, -signal.SIGTERM), (
        f"subprocess exited with rc={rc}\n"
        f"stdout: {out.decode(errors='replace')}\n"
        f"stderr: {err.decode(errors='replace')}"
    )
    assert sentinel.exists(), "graceful_shutdown sentinel was not written"
    body = sentinel.read_text()
    assert "deadline=5.0" in body, body
    assert "grace=3.0" in body, body
