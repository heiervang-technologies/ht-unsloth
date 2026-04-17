# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lile capsule lifecycle + transparent proxy."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter

router = APIRouter(prefix="/api/lile", tags=["lile"])


def _lile_base_url() -> str:
    host = os.environ.get("LILE_HOST", "127.0.0.1")
    port = os.environ.get("LILE_PORT", "8765")
    return f"http://{host}:{port}"


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


# Module-level cell; flipped by /capsule/start when we spawn.
_spawned_pid: int | None = None
