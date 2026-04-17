# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the lile capsule lifecycle + proxy routes."""

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
