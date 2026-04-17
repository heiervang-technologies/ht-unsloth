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


def test_start_spawns_subprocess_when_absent(client, monkeypatch):
    from routes import lile as lile_mod
    spawned = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            spawned["argv"] = argv
            self.pid = 4242

    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59998")
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
