# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lile capsule lifecycle + transparent proxy.

lile (the LiveLearn daemon) lives in heiervang-technologies/agi since
2026-05-15. Studio talks to it in two modes:

- **External**  — daemon is already running somewhere else; we just probe
  ``LILE_DAEMON_URL`` (or legacy ``LILE_HOST`` + ``LILE_PORT``) and proxy.
  Stop is a no-op.

- **Spawned**   — if the ``lile`` package is importable in this Python
  interpreter (install it via ``pip install lile @ git+...agi``), we can
  start it as a subprocess and own its lifecycle. Stop sends SIGTERM.

``capsule/start`` tries external first (cheap reachability probe), then
falls back to spawn if available. ``capsule/status`` reports mode so the
frontend can switch Load/Stop affordances.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/lile", tags=["lile"])

# Module-level state for the locally-spawned daemon, if any. ``None``
# means we are in external-only mode (or have not yet spawned).
_spawned: dict[str, Any] | None = None


# ----------------------------------------------------------------- URL config
def _lile_base_url() -> str:
    """Resolve the daemon's base URL from env, preferring LILE_DAEMON_URL."""
    url = os.environ.get("LILE_DAEMON_URL")
    if url:
        return url.rstrip("/")
    host = os.environ.get("LILE_HOST")
    port = os.environ.get("LILE_PORT")
    if host and port:
        return f"http://{host}:{port}"
    raise RuntimeError(
        "lile daemon location not configured: set LILE_DAEMON_URL "
        "(or legacy LILE_HOST + LILE_PORT). Install the package "
        "(`pip install lile @ git+https://github.com/heiervang-technologies/agi`) "
        "to spawn locally instead."
    )


def _can_spawn() -> bool:
    """Is the lile package importable in this Python interpreter?

    Checked at call time (not import time) so the route module loads even
    when lile isn't installed. Studio's optional extra
    ``ht-unsloth-studio[lile]`` brings it in.

    ``find_spec`` raises ``ModuleNotFoundError`` when a *parent* package is
    missing (e.g. ``lile.console`` is asked for but ``lile.console`` itself
    isn't a package). We treat that the same as "missing" — the module is
    not importable in this interpreter.
    """
    try:
        return importlib.util.find_spec("lile.console.launch") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _spawn_port() -> int:
    """Port for the spawned daemon. Reuses LILE_PORT when set, default 8768."""
    raw = os.environ.get("LILE_PORT")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 8768


async def _probe(base_url: str, timeout: float = 0.5) -> dict | None:
    """Return /health JSON if reachable, else None."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/health")
        if r.status_code == 200:
            return r.json()
    except httpx.HTTPError:
        pass
    return None


# ----------------------------------------------------------------- spawn ops
def _spawn_lile(req: "StartRequest") -> dict[str, Any]:
    """Launch lile as a subprocess. Caller owns polling for readiness.

    Returns ``{"pid", "port", "url", "log_path", "started_at"}``. Caller
    sets ``_spawned`` to this dict once the daemon is health-probable so
    /capsule/status can report mode=spawned.
    """
    if not _can_spawn():
        raise RuntimeError(
            "lile package not importable; pip install lile @ git+...agi "
            "or set LILE_DAEMON_URL to point at a running daemon."
        )
    port = _spawn_port()
    log_path = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"lile-spawn-{port}.log",
    )
    env = os.environ.copy()
    env["LILE_HOST"] = "127.0.0.1"
    env["LILE_PORT"] = str(port)
    # Forward any caller-supplied overrides that launch.py honors via env.
    for src, dst in (("model", "LILE_MODEL"),
                     ("max_seq_length", "LILE_MAX_SEQ_LENGTH"),
                     ("lora_rank", "LILE_LORA_RANK")):
        v = getattr(req, src, None)
        if v is not None:
            env[dst] = str(v)
    fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, "-m", "lile.console.launch"],
        stdout=fh, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,  # keep child alive if Studio is restarted
    )
    return {
        "pid": proc.pid,
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "log_path": log_path,
        "started_at": time.time(),
    }


async def _wait_for_health(base_url: str, deadline_s: float = 60.0) -> dict | None:
    """Poll /health until 200 or deadline."""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        health = await _probe(base_url, timeout=0.5)
        if health:
            return health
        await asyncio.sleep(1.0)
    return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just not ours — treat as alive


# ----------------------------------------------------------------- routes
@router.get("/capsule/status")
async def capsule_status() -> dict:
    global _spawned
    # 1) we have a tracked subprocess — report that mode (regardless of env)
    if _spawned and _is_alive(_spawned["pid"]):
        health = await _probe(_spawned["url"])
        if health:
            return {"running": True, "mode": "spawned",
                    "url": _spawned["url"], "pid": _spawned["pid"],
                    "health": health}
        return {"running": False, "mode": "spawned",
                "url": _spawned["url"], "pid": _spawned["pid"],
                "error": "subprocess alive but /health unreachable"}
    # 2) tracked process died — clear it so a future start can re-spawn
    if _spawned and not _is_alive(_spawned["pid"]):
        log.info("lile spawned pid %s exited; clearing", _spawned["pid"])
        _spawned = None
    # 3) external mode
    try:
        base = _lile_base_url()
    except RuntimeError as exc:
        return {"running": False, "mode": "unconfigured", "error": str(exc)}
    health = await _probe(base)
    if health:
        return {"running": True, "mode": "external",
                "externally_managed": True, "url": base, "health": health}
    return {"running": False, "mode": "external", "url": base}


class StartRequest(BaseModel):
    model: str | None = None
    max_seq_length: int | None = None
    lora_rank: int | None = None
    load_in_4bit: bool | None = None
    idle_replay: bool | None = None
    frozen_ref: bool | None = None


@router.post("/capsule/start")
async def capsule_start(req: StartRequest) -> dict:
    """Two-stage start: probe external URL first, fall back to subprocess spawn."""
    global _spawned
    # Stage 1: external daemon already serving?
    try:
        base = _lile_base_url()
    except RuntimeError:
        base = None
    if base:
        health = await _probe(base)
        if health:
            return {"running": True, "mode": "external",
                    "externally_managed": True, "url": base, "health": health}

    # Stage 2: we already spawned one — return its current state
    if _spawned and _is_alive(_spawned["pid"]):
        health = await _probe(_spawned["url"])
        if health:
            return {"running": True, "mode": "spawned",
                    "url": _spawned["url"], "pid": _spawned["pid"],
                    "health": health}

    # Stage 3: try to spawn locally
    if _can_spawn():
        try:
            info = _spawn_lile(req)
        except Exception as exc:  # noqa: BLE001
            log.exception("lile spawn failed")
            return {"running": False, "mode": "spawn-failed", "error": str(exc)}
        health = await _wait_for_health(info["url"], deadline_s=60.0)
        if health:
            _spawned = info
            return {"running": True, "mode": "spawned",
                    "url": info["url"], "pid": info["pid"],
                    "health": health, "log_path": info["log_path"]}
        # Spawned but never became healthy — record the PID so /stop can
        # clean it up; tell the caller about the log file.
        _spawned = info
        return {"running": False, "mode": "spawned",
                "url": info["url"], "pid": info["pid"],
                "log_path": info["log_path"],
                "error": "subprocess started but /health did not respond within 60s"}

    # Stage 4: neither external reachable nor spawnable
    return {"running": False,
            "mode": "external-unreachable",
            "url": base,
            "error": "lile daemon not reachable at "
                     f"{base or 'LILE_DAEMON_URL'} and the lile package "
                     "is not installed (no spawn fallback)."
                     " Install via `pip install lile @ git+...agi`"
                     " or start the daemon externally."}


@router.post("/capsule/stop")
async def capsule_stop() -> dict:
    """Stop a spawned subprocess. No-op when externally managed."""
    global _spawned
    if not _spawned:
        return {"stopped": False, "reason": "externally_managed"}
    pid = _spawned["pid"]
    if not _is_alive(pid):
        _spawned = None
        return {"stopped": True, "reason": "already_exited"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        log.warning("SIGTERM to lile pid %s failed: %s", pid, exc)
    # Brief wait for graceful shutdown.
    for _ in range(30):
        if not _is_alive(pid):
            break
        await asyncio.sleep(0.5)
    if _is_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _spawned = None
    return {"stopped": True, "reason": "subprocess_terminated", "pid": pid}


# ---------------------------------------------------------------- proxy infra
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}

# Fail fast on connect/pool, allow unbounded reads so SSE streams stay open
# for as long as upstream wants to push.
_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)


def _forward_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _proxy_target_url() -> str:
    """Where /api/lile/{path} forwards to.

    Prefer the spawned subprocess (locally guaranteed; matches what we own),
    fall back to the externally-configured URL.
    """
    if _spawned and _is_alive(_spawned["pid"]):
        return _spawned["url"]
    return _lile_base_url()


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
    url = f"{_proxy_target_url()}/{path}"
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
