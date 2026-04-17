"""FastAPI exception handlers that emit the ``lile.errors`` envelope.

Call ``register_error_handlers(app)`` once during ``create_app`` to wire:

- ``LileError`` subclasses → envelope with their own ``status_code`` / ``code``
- ``HTTPException`` → envelope, preserving the HTTP status
- ``RequestValidationError`` → envelope with ``code="invalid_input"``, HTTP 400
- Everything else → envelope with ``code="internal"``, HTTP 500

All envelopes include a ``request_id`` read from the ``lile.middleware``
contextvar. An ``X-Request-ID`` response header is set to match, so clients
that only inspect headers still see the id.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from .errors import LileError, envelope_payload
from .middleware import REQUEST_ID_HEADER, current_request_id, new_request_id

log = logging.getLogger(__name__)


# HTTPException status → taxonomy code. Anything not here falls back to
# "internal" for 5xx and "not_found" / "invalid_input" heuristic for 4xx.
_HTTP_STATUS_TO_CODE: dict[int, str] = {
    400: "invalid_input",
    404: "not_found",
    503: "shutting_down",
    504: "timeout",
}


def _resolve_request_id() -> str:
    rid = current_request_id()
    return rid if rid is not None else new_request_id()


def _respond(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
) -> JSONResponse:
    rid = _resolve_request_id()
    body = envelope_payload(
        code=code, message=message, retryable=retryable, request_id=rid,
    )
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={REQUEST_ID_HEADER: rid},
    )


def _format_validation_message(errors: list[dict[str, Any]]) -> str:
    # FastAPI's default is a list of dicts with loc/msg/type; we flatten to
    # a single short message for humans. Full detail stays in logs.
    bits: list[str] = []
    for e in errors[:3]:  # cap the wire payload
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
        msg = e.get("msg", "invalid value")
        bits.append(f"{loc or 'body'}: {msg}" if loc else msg)
    if len(errors) > 3:
        bits.append(f"(+{len(errors) - 3} more)")
    return "; ".join(bits) or "invalid input"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LileError)
    async def _lile_error_handler(_req: Request, exc: LileError) -> JSONResponse:
        return _respond(
            status_code=exc.status_code,
            code=exc.code,
            message=str(exc) or exc.code,
            retryable=exc.retryable,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _req: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        return _respond(
            status_code=400,
            code="invalid_input",
            message=_format_validation_message(exc.errors()),
            retryable=False,
        )

    @app.exception_handler(HTTPException)
    async def _http_handler(_req: Request, exc: HTTPException) -> JSONResponse:
        status = exc.status_code
        code = _HTTP_STATUS_TO_CODE.get(
            status,
            "internal" if status >= 500 else "invalid_input",
        )
        # Only transient upstream statuses are retryable. 500 stays non-retryable
        # to match ``_fallback_handler`` (which also catches uncaught exceptions
        # and hands them back as 500 internal) — clients should not blind-retry
        # an internal server error.
        return _respond(
            status_code=status,
            code=code,
            message=str(exc.detail) if exc.detail else code,
            retryable=status in (502, 503, 504),
        )

    @app.exception_handler(Exception)
    async def _fallback_handler(_req: Request, exc: Exception) -> JSONResponse:
        # Never leak exception internals to clients, but do log.
        log.exception("unhandled exception in /v1/* route: %s", exc)
        return _respond(
            status_code=500,
            code="internal",
            message="internal server error",
            retryable=False,
        )
