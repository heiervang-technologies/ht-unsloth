# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the lile capsule lifecycle + proxy routes."""

import json

import pytest
from fastapi.testclient import TestClient

from main import app  # type: ignore[import]


@pytest.fixture
def client():
    return TestClient(app)


def test_status_returns_offline_when_daemon_absent(client, monkeypatch):
    """Status probe returns running:false when lile /health is unreachable."""
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")  # nothing listening
    r = client.get("/api/lile/capsule/status")
    assert r.status_code == 200
    assert r.json() == {"running": False}


def test_status_returns_health_when_daemon_reachable(client, monkeypatch, respx_mock):
    """When lile /health responds 200, status mirrors the payload and url."""
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    body = {"ok": True, "model": "qwen3-0.6b", "queue_depth": 0,
            "commit_cursor": 7, "merges": 2}
    respx_mock.get("http://127.0.0.1:59999/health").respond(200, json=body)
    r = client.get("/api/lile/capsule/status")
    assert r.status_code == 200
    payload = r.json()
    assert payload["running"] is True
    assert payload["health"] == body
    assert payload["url"] == "http://127.0.0.1:59999"
    assert payload["externally_managed"] is True


def test_start_noop_when_already_running(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    respx_mock.get("http://127.0.0.1:59999/health").respond(200, json={"ok": True})
    r = client.post("/api/lile/capsule/start", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True
    assert body["externally_managed"] is True


def test_start_spawns_subprocess_when_absent(client, monkeypatch, respx_mock):
    from routes import lile as lile_mod
    spawned = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            spawned["argv"] = argv
            self.pid = 4242

    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59998")
    # Stub the initial /health gate as unreachable so the spawn branch runs.
    # Using respx keeps this deterministic — no real network probe on 59998.
    respx_mock.get("http://127.0.0.1:59998/health").mock(
        side_effect=__import__("httpx").ConnectError("refused")
    )
    monkeypatch.setattr(lile_mod.subprocess, "Popen", FakePopen)
    async def fake_probe():
        return {"ok": True, "model": "qwen3", "queue_depth": 0,
                "commit_cursor": 0, "merges": 0}
    monkeypatch.setattr(lile_mod, "_probe_health", fake_probe)

    r = client.post("/api/lile/capsule/start",
                    json={"model": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"})
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True
    assert body["externally_managed"] is False
    assert body["pid"] == 4242
    assert "lile.server" in " ".join(spawned["argv"])
    assert "--port" in spawned["argv"]


def test_start_closes_log_handle_when_popen_fails(client, monkeypatch, respx_mock):
    """If Popen raises, the daemon.log file handle must not leak."""
    import builtins
    from routes import lile as lile_mod

    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59998")
    respx_mock.get("http://127.0.0.1:59998/health").mock(
        side_effect=__import__("httpx").ConnectError("refused")
    )

    opened = []
    real_open = builtins.open

    class TrackingFH:
        def __init__(self, *a, **kw):
            self._real = real_open(*a, **kw)
            self.closed_ = False
            opened.append(self)
        def __getattr__(self, name):
            return getattr(self._real, name)
        def close(self):
            self.closed_ = True
            return self._real.close()

    def fake_open(path, *args, **kwargs):
        # Only track daemon.log, leave everything else alone so FastAPI /
        # logging internals keep working.
        if str(path).endswith("daemon.log"):
            return TrackingFH(path, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    def boom(*args, **kwargs):
        raise OSError("simulated Popen failure")
    monkeypatch.setattr(lile_mod.subprocess, "Popen", boom)

    # FastAPI will surface the OSError as a 500. What we really care about
    # is that the daemon.log fh we opened was closed.
    try:
        client.post("/api/lile/capsule/start", json={})
    except OSError:
        pass  # propagated by TestClient in some configs
    assert opened, "expected daemon.log to be opened"
    assert all(fh.closed_ for fh in opened), "fh leaked on Popen failure"


def test_stop_refuses_when_externally_managed(client, monkeypatch):
    from routes import lile as lile_mod
    monkeypatch.setattr(lile_mod, "_spawned_pid", None)
    r = client.post("/api/lile/capsule/stop")
    assert r.status_code == 200
    assert r.json() == {"stopped": False, "reason": "externally_managed"}


def test_stop_sends_signal_when_we_spawned(client, monkeypatch):
    from routes import lile as lile_mod
    killed = {}
    monkeypatch.setattr(lile_mod, "_spawned_pid", 4242)
    def fake_kill(pid, sig):
        killed["pid"] = pid; killed["sig"] = sig
    monkeypatch.setattr(lile_mod.os, "kill", fake_kill)
    r = client.post("/api/lile/capsule/stop")
    assert r.status_code == 200
    assert r.json()["stopped"] is True
    assert killed == {"pid": 4242, "sig": lile_mod.signal.SIGTERM}
    assert lile_mod._spawned_pid is None


def test_proxy_forwards_get(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    respx_mock.get("http://127.0.0.1:59999/v1/state/trajectory/tail")\
              .respond(200, json={"events": [], "next_offset": 0, "total_size": 0})
    r = client.get("/api/lile/v1/state/trajectory/tail")
    assert r.status_code == 200
    assert r.json()["total_size"] == 0


def test_proxy_forwards_post_with_body(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    route = respx_mock.post("http://127.0.0.1:59999/v1/train")\
                      .respond(200, json={"queued": True})
    r = client.post("/api/lile/v1/train",
                    json={"objective": "sft", "samples": []})
    assert r.status_code == 200
    assert route.called
    sent = json.loads(route.calls.last.request.content)
    assert sent["objective"] == "sft"


def test_proxy_502_on_upstream_down(client, monkeypatch):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59997")  # nothing listens
    r = client.get("/api/lile/v1/foo")
    assert r.status_code == 502
    assert "proxy upstream" in r.json()["error"]


def test_proxy_forwards_head(client, monkeypatch, respx_mock):
    """HEAD should be accepted by the proxy so health probes work."""
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    respx_mock.head("http://127.0.0.1:59999/healthz").respond(
        200, headers={"x-lile-probe": "ok"},
    )
    r = client.head("/api/lile/healthz")
    assert r.status_code == 200
    assert r.headers.get("x-lile-probe") == "ok"


def test_proxy_timeout_has_bounded_connect_unbounded_read():
    """Regression guard: connect must fail fast, read must be None for SSE."""
    from routes import lile as lile_mod
    t = lile_mod._PROXY_TIMEOUT
    assert t.connect is not None and t.connect <= 10.0
    assert t.read is None, "SSE requires unbounded read timeout"


def test_proxy_streams_sse_without_buffering(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")

    sse_body = (b"data: {\"delta\": \"hel\"}\n\n"
                b"data: {\"delta\": \"lo\"}\n\n"
                b"data: [DONE]\n\n")
    respx_mock.post("http://127.0.0.1:59999/v1/chat/completions").respond(
        200, content=sse_body,
        headers={"content-type": "text/event-stream",
                 "x-accel-buffering": "no"},
    )

    with client.stream("POST", "/api/lile/v1/chat/completions",
                       headers={"accept": "text/event-stream"},
                       json={"messages": [{"role": "user", "content": "hi"}],
                             "stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("x-accel-buffering") == "no"
        chunks = list(r.iter_bytes())
    assembled = b"".join(chunks)
    assert b"[DONE]" in assembled
