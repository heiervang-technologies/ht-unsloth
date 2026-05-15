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


def test_start_reports_unreachable_when_daemon_absent(client, monkeypatch):
    """capsule/start no longer spawns; it just reports daemon reachability."""
    monkeypatch.setenv("LILE_DAEMON_URL", "http://127.0.0.1:59998")
    r = client.post("/api/lile/capsule/start", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False
    assert body["externally_managed"] is True
    assert "LILE_DAEMON_URL" in body["error"]


def test_stop_is_always_externally_managed(client):
    """capsule/stop is a no-op since lile is externally managed."""
    r = client.post("/api/lile/capsule/stop")
    assert r.status_code == 200
    assert r.json() == {"stopped": False, "reason": "externally_managed"}


def test_base_url_requires_configuration(monkeypatch):
    """_lile_base_url raises if no env hints are set, surfacing the move."""
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
