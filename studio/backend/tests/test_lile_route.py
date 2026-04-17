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
