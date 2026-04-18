"""Unit tests for `lile.errors` — envelope shape and code taxonomy.

The envelope is the contract every `/v1/*` response uses when something goes
wrong. Every follow-up (metrics, admission control, auth) reads `error.code`
and `error.retryable`, so the shape is load-bearing.

Run with: pytest lile/tests/test_errors.py
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_only


def test_envelope_shape_is_nested_under_error_key():
    from lile.errors import envelope_payload

    payload = envelope_payload(
        code="invalid_input",
        message="samples[0].prompt is required",
        retryable=False,
        request_id="req_deadbeefdeadbeef",
    )
    assert set(payload.keys()) == {"error"}
    err = payload["error"]
    assert err["code"] == "invalid_input"
    assert err["message"] == "samples[0].prompt is required"
    assert err["retryable"] is False
    assert err["request_id"] == "req_deadbeefdeadbeef"


def test_envelope_default_retryable_is_false():
    from lile.errors import envelope_payload

    payload = envelope_payload(code="internal", message="x", request_id="req_x")
    assert payload["error"]["retryable"] is False


def test_code_taxonomy_is_closed_set():
    from lile.errors import ERROR_CODES

    expected = {
        "invalid_input",
        "unknown_objective",
        "unknown_response_id",
        "not_found",
        "queue_full",
        "shutting_down",
        "shutdown_dropped",
        "timeout",
        "internal",
    }
    # Taxonomy must be at least this big. Adding new codes is fine; removing
    # is a breaking change for clients and requires an ADR.
    assert expected <= set(ERROR_CODES)


def test_lile_error_subclasses_carry_code_and_status():
    from lile.errors import (
        InvalidInputError,
        NotFoundError,
        QueueFullError,
        ShuttingDownError,
        TimeoutError as LileTimeoutError,
        UnknownResponseIdError,
    )

    assert InvalidInputError("x").code == "invalid_input"
    assert InvalidInputError("x").status_code == 400

    assert UnknownResponseIdError("r_deadbeef").code == "unknown_response_id"
    assert UnknownResponseIdError("r_deadbeef").status_code == 404

    assert NotFoundError("x").code == "not_found"
    assert NotFoundError("x").status_code == 404

    assert QueueFullError("x").code == "queue_full"
    assert QueueFullError("x").status_code == 503
    assert QueueFullError("x").retryable is True

    assert ShuttingDownError("x").code == "shutting_down"
    assert ShuttingDownError("x").status_code == 503
    assert ShuttingDownError("x").retryable is True

    assert LileTimeoutError("x").code == "timeout"
    assert LileTimeoutError("x").status_code == 504


def test_lile_error_default_message_goes_to_envelope():
    from lile.errors import InvalidInputError, envelope_payload

    exc = InvalidInputError("samples must be non-empty")
    payload = envelope_payload(
        code=exc.code,
        message=str(exc),
        retryable=exc.retryable,
        request_id="req_test",
    )
    assert payload["error"]["message"] == "samples must be non-empty"
    assert payload["error"]["code"] == "invalid_input"
