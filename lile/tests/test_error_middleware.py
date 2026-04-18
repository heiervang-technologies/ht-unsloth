"""FastAPI-level tests for request-ID middleware + exception handlers.

These pin the wire contract for `/v1/*` error responses without loading a
model. We mount `register_error_handlers(app)` + `RequestIDMiddleware` on a
minimal app with stub routes that raise the relevant exceptions, then
probe it with TestClient.

Run with: pytest lile/tests/test_error_middleware.py
"""
from __future__ import annotations

import logging
import re

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

pytestmark = pytest.mark.cpu_only


REQ_ID_RE = re.compile(r"^req_[0-9a-f]{16}$")


def _build_app() -> FastAPI:
    from lile.errors import (
        InvalidInputError,
        QueueFullError,
        ShuttingDownError,
        UnknownResponseIdError,
    )
    from lile.middleware import RequestIDMiddleware, current_request_id
    from lile.server_errors import register_error_handlers

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    @app.get("/ok")
    async def ok() -> dict:
        return {"ok": True, "rid": current_request_id()}

    @app.get("/boom-value")
    async def boom_value() -> dict:
        raise ValueError("boom")

    @app.get("/boom-http-404")
    async def boom_http_404() -> dict:
        raise HTTPException(status_code=404, detail="not here")

    @app.get("/boom-http-500")
    async def boom_http_500() -> dict:
        raise HTTPException(status_code=500, detail="internal")

    @app.get("/boom-http-502")
    async def boom_http_502() -> dict:
        raise HTTPException(status_code=502, detail="bad gateway")

    @app.get("/boom-invalid")
    async def boom_invalid() -> dict:
        raise InvalidInputError("samples must be non-empty")

    @app.get("/boom-unknown-rid")
    async def boom_unknown_rid() -> dict:
        raise UnknownResponseIdError("r_missing")

    @app.get("/boom-queue-full")
    async def boom_queue_full() -> dict:
        raise QueueFullError("queue depth exceeded")

    @app.get("/boom-shutting-down")
    async def boom_shutting_down() -> dict:
        raise ShuttingDownError("daemon is shutting down")

    from pydantic import BaseModel

    class Payload(BaseModel):
        name: str

    @app.post("/needs-body")
    async def needs_body(p: Payload) -> dict:
        return {"name": p.name}

    return app


# ---------------------------------------------------------------- request ID middleware

def test_middleware_generates_request_id_when_header_missing():
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/ok")
    assert r.status_code == 200
    rid = r.headers.get("x-request-id")
    assert rid is not None
    assert REQ_ID_RE.match(rid), rid
    assert r.json()["rid"] == rid


def test_middleware_passes_through_incoming_x_request_id():
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/ok", headers={"X-Request-ID": "req_clientsupplied1"})
    assert r.headers["x-request-id"] == "req_clientsupplied1"
    assert r.json()["rid"] == "req_clientsupplied1"


def test_middleware_different_requests_get_different_ids():
    app = _build_app()
    with TestClient(app) as client:
        a = client.get("/ok").headers["x-request-id"]
        b = client.get("/ok").headers["x-request-id"]
    assert a != b


# ---------------------------------------------------------------- exception handlers

def _assert_envelope(body: dict, *, code: str, retryable: bool) -> dict:
    assert set(body.keys()) == {"error"}
    err = body["error"]
    assert err["code"] == code
    assert err["retryable"] is retryable
    assert isinstance(err["message"], str) and err["message"]
    assert REQ_ID_RE.match(err["request_id"]), err["request_id"]
    return err


def test_generic_exception_becomes_500_envelope():
    app = _build_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/boom-value")
    assert r.status_code == 500
    err = _assert_envelope(r.json(), code="internal", retryable=False)
    assert r.headers["x-request-id"] == err["request_id"]


def test_http_exception_becomes_envelope_with_matching_status():
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/boom-http-404")
    assert r.status_code == 404
    _assert_envelope(r.json(), code="not_found", retryable=False)


def test_invalid_input_error_becomes_400_envelope():
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/boom-invalid")
    assert r.status_code == 400
    err = _assert_envelope(r.json(), code="invalid_input", retryable=False)
    assert "samples must be non-empty" in err["message"]


def test_unknown_response_id_becomes_404_envelope():
    """PIN the /v1/feedback status-code bug fix (was 200 + {"error":...})."""
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/boom-unknown-rid")
    assert r.status_code == 404
    err = _assert_envelope(r.json(), code="unknown_response_id", retryable=False)
    assert "r_missing" in err["message"]


def test_queue_full_is_retryable_503():
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/boom-queue-full")
    assert r.status_code == 503
    _assert_envelope(r.json(), code="queue_full", retryable=True)


def test_shutting_down_is_retryable_503():
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/boom-shutting-down")
    assert r.status_code == 503
    _assert_envelope(r.json(), code="shutting_down", retryable=True)


def test_http_500_is_not_retryable():
    """Align with ``_fallback_handler`` — plain 500 is internal, not transient."""
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/boom-http-500")
    assert r.status_code == 500
    _assert_envelope(r.json(), code="internal", retryable=False)


def test_http_502_is_retryable():
    """Upstream gateway errors are the only 5xx statuses we flag retryable."""
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/boom-http-502")
    assert r.status_code == 502
    _assert_envelope(r.json(), code="internal", retryable=True)


def test_pydantic_validation_error_becomes_envelope_400():
    app = _build_app()
    with TestClient(app) as client:
        r = client.post("/needs-body", json={"wrong": "shape"})
    assert r.status_code == 400
    err = _assert_envelope(r.json(), code="invalid_input", retryable=False)
    # FastAPI's default error has `loc`/`msg`; the envelope just needs to be human-readable.
    assert err["message"]  # non-empty


# ---------------------------------------------------------------- trajectory contextvar

def test_trajectory_stamps_request_id_when_set(tmp_path):
    from lile.middleware import _REQUEST_ID_CTX, current_request_id, set_request_id
    from lile.trajectory import TrajectoryLog

    log = TrajectoryLog(tmp_path / "t.jsonl")

    token = set_request_id("req_trajstamp000001")
    try:
        log.log_event("test_event", {"k": 1})
        assert current_request_id() == "req_trajstamp000001"
    finally:
        _REQUEST_ID_CTX.reset(token)

    events = log.tail(1)
    assert events[0]["kind"] == "test_event"
    assert events[0]["request_id"] == "req_trajstamp000001"


def test_trajectory_omits_request_id_when_unset(tmp_path):
    from lile.trajectory import TrajectoryLog

    log = TrajectoryLog(tmp_path / "t.jsonl")
    log.log_event("test_event", {"k": 1})
    ev = log.tail(1)[0]
    assert "request_id" not in ev


# ---------------------------------------------------------------- logging filter

def test_log_records_carry_request_id(caplog):
    from lile.middleware import (
        RequestIdLogFilter,
        _REQUEST_ID_CTX,
        set_request_id,
    )

    logger = logging.getLogger("lile.test_rid")
    logger.setLevel(logging.INFO)
    filt = RequestIdLogFilter()
    logger.addFilter(filt)

    token = set_request_id("req_logtest0000001")
    try:
        with caplog.at_level(logging.INFO, logger="lile.test_rid"):
            logger.info("hello from request")
    finally:
        _REQUEST_ID_CTX.reset(token)

    assert caplog.records
    rec = caplog.records[-1]
    assert getattr(rec, "request_id", None) == "req_logtest0000001"
    logger.removeFilter(filt)
