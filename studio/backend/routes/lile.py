# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lile capsule lifecycle + transparent proxy."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/lile", tags=["lile"])


def _lile_base_url() -> str:
    host = os.environ.get("LILE_HOST", "127.0.0.1")
    port = os.environ.get("LILE_PORT", "8765")
    return f"http://{host}:{port}"


# Module-level cell; flipped by /capsule/start when we spawn.
_spawned_pid: int | None = None


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
            "externally_managed": _spawned_pid is None,
            "health": r.json(),
            "url": _lile_base_url(),
        }
    except (httpx.ConnectError, httpx.TimeoutException):
        return {"running": False}


class StartRequest(BaseModel):
    model: str | None = None
    max_seq_length: int | None = None
    lora_rank: int | None = None
    load_in_4bit: bool | None = None
    idle_replay: bool | None = None
    frozen_ref: bool | None = None


async def _probe_health() -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=0.5) as c:
            r = await c.get(f"{_lile_base_url()}/health")
        if r.status_code == 200:
            return r.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return None


def _data_dir() -> Path:
    default = Path(__file__).resolve().parents[3] / "lile_data"
    d = Path(os.environ.get("LILE_DATA_DIR", str(default)))
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/capsule/start")
async def capsule_start(req: StartRequest) -> dict:
    global _spawned_pid
    # Initial gate: inline probe so tests can independently monkeypatch
    # `_probe_health` for the post-spawn readiness loop without short-
    # circuiting the spawn branch.
    try:
        async with httpx.AsyncClient(timeout=0.5) as c:
            r = await c.get(f"{_lile_base_url()}/health")
        if r.status_code == 200:
            return {"running": True, "externally_managed": True,
                    "health": r.json(), "url": _lile_base_url()}
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    port = os.environ.get("LILE_PORT", "8765")
    log_path = _data_dir() / "daemon.log"
    argv = [sys.executable, "-m", "lile.server", "--port", str(port)]
    if req.model:
        argv += ["--model", req.model]

    fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        argv, stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
    )
    _spawned_pid = proc.pid

    for _ in range(240):  # 240 * 0.5s = 120s
        health = await _probe_health()
        if health is not None:
            return {"running": True, "externally_managed": False,
                    "pid": proc.pid, "url": _lile_base_url(),
                    "health": health}
        await asyncio.sleep(0.5)

    return {"running": False, "error": "health-check timeout (120s)",
            "pid": proc.pid, "log": str(log_path)}


@router.post("/capsule/stop")
async def capsule_stop() -> dict:
    global _spawned_pid
    if _spawned_pid is None:
        return {"stopped": False, "reason": "externally_managed"}
    try:
        os.kill(_spawned_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    pid = _spawned_pid
    _spawned_pid = None
    return {"stopped": True, "pid": pid}


_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}


def _forward_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


async def _proxy_stream(method: str, url: str, headers: dict, body: bytes):
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(method, url, content=body, headers=headers)
    upstream = await client.send(req, stream=True)

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
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
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

        async with httpx.AsyncClient(timeout=None) as c:
            upstream = await c.request(
                request.method, url, content=body, headers=headers,
                follow_redirects=False,
            )
        rh = _forward_headers(upstream.headers)
        return Response(content=upstream.content,
                        status_code=upstream.status_code,
                        headers=rh,
                        media_type=upstream.headers.get("content-type"))
    except (httpx.ConnectError, httpx.ReadError):
        return Response(
            content=json.dumps({"error": "proxy upstream failure"}),
            status_code=502, media_type="application/json",
        )
