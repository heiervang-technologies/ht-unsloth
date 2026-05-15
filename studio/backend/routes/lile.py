# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lile capsule status + transparent proxy.

The lile daemon moved to heiervang-technologies/agi on 2026-05-15. Studio
no longer spawns it; it just proxies HTTP traffic at LILE_DAEMON_URL.
"""

from __future__ import annotations

import json
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/lile", tags=["lile"])


def _lile_base_url() -> str:
    url = os.environ.get("LILE_DAEMON_URL")
    if url:
        return url.rstrip("/")
    host = os.environ.get("LILE_HOST")
    port = os.environ.get("LILE_PORT")
    if host and port:
        return f"http://{host}:{port}"
    raise RuntimeError(
        "lile daemon location not configured: set LILE_DAEMON_URL "
        "(or legacy LILE_HOST + LILE_PORT). lile lives in "
        "heiervang-technologies/agi since 2026-05-15 and is no longer "
        "spawnable from this process."
    )


@router.get("/capsule/status")
async def capsule_status() -> dict:
    url = f"{_lile_base_url()}/health"
    try:
        async with httpx.AsyncClient(timeout=0.5) as c:
            r = await c.get(url)
        if r.status_code != 200:
            return {"running": False}
        return {
            "running": True,
            "externally_managed": True,
            "health": r.json(),
            "url": _lile_base_url(),
        }
    except httpx.HTTPError:
        return {"running": False}


class StartRequest(BaseModel):
    model: str | None = None
    max_seq_length: int | None = None
    lora_rank: int | None = None
    load_in_4bit: bool | None = None
    idle_replay: bool | None = None
    frozen_ref: bool | None = None


@router.post("/capsule/start")
async def capsule_start(req: StartRequest) -> dict:
    # lile lives in heiervang-technologies/agi since 2026-05-15 and is no
    # longer spawned by Studio. The endpoint stays so the frontend's
    # capsule lifecycle UI keeps working; it now just reports whether the
    # externally-managed daemon at LILE_DAEMON_URL is reachable.
    del req  # all spawn-time parameters are configured on the daemon side
    try:
        async with httpx.AsyncClient(timeout=0.5) as c:
            r = await c.get(f"{_lile_base_url()}/health")
        if r.status_code == 200:
            return {"running": True, "externally_managed": True,
                    "health": r.json(), "url": _lile_base_url()}
    except httpx.HTTPError:
        pass
    return {"running": False, "externally_managed": True,
            "url": _lile_base_url(),
            "error": "lile daemon not reachable at LILE_DAEMON_URL; start "
                     "it from the agi repo (python -m lile.console.launch)"}


@router.post("/capsule/stop")
async def capsule_stop() -> dict:
    # lile is externally managed since the move to heiervang-technologies/agi.
    return {"stopped": False, "reason": "externally_managed"}


_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}

# Fail fast on connect/pool, allow unbounded reads so SSE streams stay open
# for as long as upstream wants to push.
_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)


def _forward_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


async def _proxy_stream(method: str, url: str, headers: dict, body: bytes):
    client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)
    try:
        req = client.build_request(method, url, content=body, headers=headers)
        upstream = await client.send(req, stream=True)
    except BaseException:
        # send() raised before we could hand ownership to the generator —
        # close the client ourselves so it doesn't leak a connection pool.
        await client.aclose()
        raise

    async def gen():
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    resp_headers = _forward_headers(upstream.headers)
    resp_headers["x-accel-buffering"] = "no"
    resp_headers["cache-control"] = "no-cache"
    return StreamingResponse(
        gen(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


@router.api_route(
    "/{path:path}",
    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(path: str, request: Request):
    url = f"{_lile_base_url()}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    body = await request.body()
    headers = _forward_headers(request.headers)
    accept = request.headers.get("accept", "")
    is_sse = "text/event-stream" in accept.lower()

    try:
        if is_sse:
            return await _proxy_stream(request.method, url, headers, body)

        async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as c:
            upstream = await c.request(
                request.method, url, content=body, headers=headers,
                follow_redirects=False,
            )
        rh = _forward_headers(upstream.headers)
        return Response(content=upstream.content,
                        status_code=upstream.status_code,
                        headers=rh,
                        media_type=upstream.headers.get("content-type"))
    except httpx.HTTPError as exc:
        return Response(
            content=json.dumps({
                "error": "proxy upstream failure",
                "detail": f"{type(exc).__name__}: {exc}",
            }),
            status_code=502, media_type="application/json",
        )
