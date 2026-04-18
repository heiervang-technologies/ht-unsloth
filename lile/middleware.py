"""Request-ID middleware, contextvar, and logging filter.

Every `/v1/*` request gets an ``X-Request-ID`` header on the response — either
echoed from the incoming header or freshly minted as ``req_<hex16>``. The id
is stashed on a ``contextvars.ContextVar`` so downstream code (exception
handlers, trajectory logging, log records) can pick it up without threading
it through every call.

See issue #12 for the design note.
"""
from __future__ import annotations

import contextvars
import logging
import secrets
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


_REQUEST_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "lile_request_id", default=None,
)

REQUEST_ID_HEADER = "X-Request-ID"


def new_request_id() -> str:
    """Return a fresh `req_<hex16>` identifier."""
    return "req_" + secrets.token_hex(8)


def current_request_id() -> Optional[str]:
    """Return the request id bound to the current async/thread context, if any."""
    return _REQUEST_ID_CTX.get()


def set_request_id(request_id: str) -> contextvars.Token:
    """Manually bind a request id. Returns a token for ``_REQUEST_ID_CTX.reset``.

    Prefer ``RequestIDMiddleware`` in FastAPI. This helper exists for tests
    and for background tasks that want to inherit an id.
    """
    return _REQUEST_ID_CTX.set(request_id)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Read/generate ``X-Request-ID``, bind to contextvar, echo on response."""

    def __init__(self, app: ASGIApp, header_name: str = REQUEST_ID_HEADER) -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(self.header_name)
        rid = incoming if incoming else new_request_id()
        token = _REQUEST_ID_CTX.set(rid)
        try:
            response: Response = await call_next(request)
        finally:
            _REQUEST_ID_CTX.reset(token)
        response.headers[self.header_name] = rid
        return response


class RequestIdLogFilter(logging.Filter):
    """Inject ``record.request_id`` from the contextvar (or ``"-"`` if unset).

    Pair with a formatter like ``%(request_id)s`` to see the id in log lines.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        rid = _REQUEST_ID_CTX.get()
        record.request_id = rid if rid is not None else "-"
        return True
