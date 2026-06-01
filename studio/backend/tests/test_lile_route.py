# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the lile capsule lifecycle + proxy routes.

Two modes covered:
- ``external`` — daemon is somewhere else; status/start probe ``LILE_DAEMON_URL``.
- ``spawned``  — Studio spawned the daemon as a subprocess; stop SIGTERMs it.
"""

import json

import pytest
from fastapi.testclient import TestClient

from main import app  # type: ignore[import]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_spawned_state():
    """Reset module-level _spawned between tests so they don't bleed."""
    from routes import lile as lile_mod
    lile_mod._spawned = None
    yield
    lile_mod._spawned = None


# ---------------------------------------------------------------- status
def test_status_returns_offline_when_daemon_absent(client, monkeypatch):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    r = client.get("/api/lile/capsule/status")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False
    assert body["mode"] == "external"


def test_status_unconfigured_when_no_env(client, monkeypatch):
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    monkeypatch.delenv("LILE_HOST", raising=False)
    monkeypatch.delenv("LILE_PORT", raising=False)
    r = client.get("/api/lile/capsule/status")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False
    assert body["mode"] == "unconfigured"


def test_status_returns_health_when_daemon_reachable(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    body = {"ok": True, "model": "qwen3-0.6b", "queue_depth": 0,
            "commit_cursor": 7, "merges": 2}
    respx_mock.get("http://127.0.0.1:59999/health").respond(200, json=body)
    r = client.get("/api/lile/capsule/status")
    assert r.status_code == 200
    payload = r.json()
    assert payload["running"] is True
    assert payload["mode"] == "external"
    assert payload["externally_managed"] is True
    assert payload["health"] == body
    assert payload["url"] == "http://127.0.0.1:59999"


def test_status_reports_spawned_mode_when_subprocess_alive(client, monkeypatch, respx_mock):
    """A tracked subprocess wins over env config in status reporting."""
    from routes import lile as lile_mod
    lile_mod._spawned = {"pid": 99999, "port": 59999,
                         "url": "http://127.0.0.1:59999",
                         "log_path": "/tmp/x", "started_at": 0}
    monkeypatch.setattr(lile_mod, "_is_alive", lambda pid: True)
    respx_mock.get("http://127.0.0.1:59999/health").respond(
        200, json={"ok": True, "model": "qwen3"})
    r = client.get("/api/lile/capsule/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "spawned"
    assert body["pid"] == 99999
    assert body["health"]["ok"] is True


def test_status_clears_dead_subprocess(client, monkeypatch):
    """If our tracked PID died, status clears it and falls through to external."""
    from routes import lile as lile_mod
    lile_mod._spawned = {"pid": 11111, "port": 59999,
                         "url": "http://127.0.0.1:59999",
                         "log_path": "/tmp/x", "started_at": 0}
    monkeypatch.setattr(lile_mod, "_is_alive", lambda pid: False)
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59998")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    r = client.get("/api/lile/capsule/status")
    assert lile_mod._spawned is None
    assert r.json()["mode"] == "external"


# ---------------------------------------------------------------- start
def test_start_noop_when_external_already_running(client, monkeypatch, respx_mock):
    """External reachable => no spawn attempt; mode=external."""
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    respx_mock.get("http://127.0.0.1:59999/health").respond(200, json={"ok": True})
    r = client.post("/api/lile/capsule/start", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True
    assert body["mode"] == "external"
    assert body["externally_managed"] is True


def test_start_falls_back_to_spawn_when_external_unreachable(client, monkeypatch):
    """If external probe fails and lile is importable, spawn it."""
    from routes import lile as lile_mod
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59996")  # nothing on this port
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    monkeypatch.setattr(lile_mod, "lile_available", lambda: True)

    spawn_info = {"pid": 4242, "port": 59996,
                  "url": "http://127.0.0.1:59996",
                  "log_path": "/tmp/lile-spawn-59996.log",
                  "started_at": 0}
    monkeypatch.setattr(lile_mod, "_spawn_lile", lambda req: spawn_info)

    async def _fake_wait(_url, deadline_s=60.0):  # noqa: ARG001
        return {"ok": True, "model": "qwen3", "queue_depth": 0}
    monkeypatch.setattr(lile_mod, "_wait_for_health", _fake_wait)

    r = client.post("/api/lile/capsule/start", json={"model": "qwen3"})
    body = r.json()
    assert body["running"] is True
    assert body["mode"] == "spawned"
    assert body["pid"] == 4242
    assert lile_mod._spawned == spawn_info


def test_start_reports_unreachable_when_no_spawn_available(client, monkeypatch):
    """Neither external nor spawnable => mode=external-unreachable with guidance."""
    from routes import lile as lile_mod
    monkeypatch.setenv("LILE_DAEMON_URL", "http://127.0.0.1:59998")
    monkeypatch.setattr(lile_mod, "lile_available", lambda: False)
    r = client.post("/api/lile/capsule/start", json={})
    body = r.json()
    assert body["running"] is False
    assert body["mode"] == "external-unreachable"
    assert "pip install lile" in body["error"]


def test_start_records_pid_even_when_health_times_out(client, monkeypatch):
    """A spawn that doesn't become healthy still tracks the PID so /stop cleans up."""
    from routes import lile as lile_mod
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59996")
    monkeypatch.setattr(lile_mod, "lile_available", lambda: True)
    info = {"pid": 5555, "port": 59996, "url": "http://127.0.0.1:59996",
            "log_path": "/tmp/lile-spawn-59996.log", "started_at": 0}
    monkeypatch.setattr(lile_mod, "_spawn_lile", lambda req: info)

    async def _fake_wait(_url, deadline_s=60.0):  # noqa: ARG001
        return None
    monkeypatch.setattr(lile_mod, "_wait_for_health", _fake_wait)

    r = client.post("/api/lile/capsule/start", json={})
    body = r.json()
    assert body["running"] is False
    assert body["mode"] == "spawned"
    assert body["pid"] == 5555
    assert "did not respond" in body["error"]
    assert lile_mod._spawned == info  # tracked for /stop


# ---------------------------------------------------------------- stop
def test_stop_is_noop_when_not_spawned(client):
    r = client.post("/api/lile/capsule/stop")
    assert r.status_code == 200
    assert r.json() == {"stopped": False, "reason": "externally_managed"}


def test_stop_sigterms_tracked_subprocess(client, monkeypatch):
    from routes import lile as lile_mod
    lile_mod._spawned = {"pid": 7777, "port": 59999,
                         "url": "http://127.0.0.1:59999",
                         "log_path": "/tmp/x", "started_at": 0}
    alive_calls = {"count": 0}
    # First _is_alive call: yes; second (after SIGTERM loop): no.
    def _alive(pid):
        alive_calls["count"] += 1
        return alive_calls["count"] <= 1
    monkeypatch.setattr(lile_mod, "_is_alive", _alive)

    killed: dict = {}
    def _killpg(pgid, sig):
        killed["pgid"] = pgid
        killed["sig"] = sig
    monkeypatch.setattr(lile_mod.os, "killpg", _killpg)
    monkeypatch.setattr(lile_mod.os, "getpgid", lambda pid: pid)

    r = client.post("/api/lile/capsule/stop")
    body = r.json()
    assert body["stopped"] is True
    assert body["reason"] == "subprocess_terminated"
    assert body["pid"] == 7777
    assert killed["pgid"] == 7777
    assert lile_mod._spawned is None


def test_stop_handles_already_exited(client, monkeypatch):
    from routes import lile as lile_mod
    lile_mod._spawned = {"pid": 8888, "port": 59999,
                         "url": "http://127.0.0.1:59999",
                         "log_path": "/tmp/x", "started_at": 0}
    monkeypatch.setattr(lile_mod, "_is_alive", lambda pid: False)
    r = client.post("/api/lile/capsule/stop")
    body = r.json()
    assert body == {"stopped": True, "reason": "already_exited"}
    assert lile_mod._spawned is None


# ---------------------------------------------------------------- _lile_base_url
def test_base_url_requires_configuration(monkeypatch):
    from routes import lile as lile_mod
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    monkeypatch.delenv("LILE_HOST", raising=False)
    monkeypatch.delenv("LILE_PORT", raising=False)
    with pytest.raises(RuntimeError, match="LILE_DAEMON_URL"):
        lile_mod._lile_base_url()


def test_base_url_prefers_daemon_url_over_host_port(monkeypatch):
    from routes import lile as lile_mod
    monkeypatch.setenv("LILE_DAEMON_URL", "http://lile.example:8080/")
    monkeypatch.setenv("LILE_HOST", "ignored")
    monkeypatch.setenv("LILE_PORT", "9999")
    assert lile_mod._lile_base_url() == "http://lile.example:8080"


# ---------------------------------------------------------------- proxy
def test_proxy_targets_spawned_when_alive(monkeypatch):
    """Spawned subprocess wins over env URL for proxy target."""
    from routes import lile as lile_mod
    lile_mod._spawned = {"pid": 1234, "port": 59999,
                         "url": "http://127.0.0.1:59999",
                         "log_path": "/tmp/x", "started_at": 0}
    monkeypatch.setattr(lile_mod, "_is_alive", lambda pid: True)
    monkeypatch.setenv("LILE_DAEMON_URL", "http://external.example:8080")
    assert lile_mod._proxy_target_url() == "http://127.0.0.1:59999"


def test_proxy_falls_back_to_env_when_no_spawn(monkeypatch):
    from routes import lile as lile_mod
    lile_mod._spawned = None
    monkeypatch.setenv("LILE_DAEMON_URL", "http://external.example:8080")
    assert lile_mod._proxy_target_url() == "http://external.example:8080"


def test_proxy_forwards_get(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    respx_mock.get("http://127.0.0.1:59999/v1/state/trajectory/tail")\
              .respond(200, json={"events": [], "next_offset": 0, "total_size": 0})
    r = client.get("/api/lile/v1/state/trajectory/tail")
    assert r.status_code == 200
    assert r.json()["total_size"] == 0


def test_proxy_forwards_post_with_body(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
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
    monkeypatch.setenv("LILE_PORT", "59997")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    r = client.get("/api/lile/v1/foo")
    assert r.status_code == 502
    assert "proxy upstream" in r.json()["error"]


def test_proxy_forwards_head(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)
    respx_mock.head("http://127.0.0.1:59999/healthz").respond(
        200, headers={"x-lile-probe": "ok"},
    )
    r = client.head("/api/lile/healthz")
    assert r.status_code == 200
    assert r.headers.get("x-lile-probe") == "ok"


def test_proxy_timeout_has_bounded_connect_unbounded_read():
    from routes import lile as lile_mod
    t = lile_mod._PROXY_TIMEOUT
    assert t.connect is not None and t.connect <= 10.0
    assert t.read is None, "SSE requires unbounded read timeout"


def test_proxy_streams_sse_without_buffering(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    monkeypatch.delenv("LILE_DAEMON_URL", raising=False)

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
