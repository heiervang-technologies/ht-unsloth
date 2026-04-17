# Lile Frontend Capsule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a loadable "lile capsule" inside Unsloth Studio — new `/lile` route with live charts + train/trajectory/snapshot tabs, plus an optional "lile mode" toggle on the existing `/chat` route that routes chat through the lile daemon.

**Architecture:** Studio backend gains a `routes/lile.py` module that (a) detects or spawns the lile daemon as an independent subprocess, (b) transparently proxies `/api/lile/*` (including SSE) to it. Frontend gains `features/lile/` with a zustand store, an incremental trajectory poller, and components built on the existing shadcn/recharts chart primitives. `/chat` gets three surgical extensions: runtime baseURL switch, per-message feedback actions, `block-on-last-commit` toggle.

**Tech Stack:** FastAPI (studio backend) · httpx (async proxy) · React 18 + TypeScript + Vite · TanStack Router · Zustand · shadcn/ui · Recharts · `@assistant-ui/react` (chat runtime) · pytest + vitest.

**Spec:** `docs/superpowers/specs/2026-04-17-lile-frontend-capsule-design.md`

---

## File Structure

### Created

**Studio backend**
- `studio/backend/routes/lile.py` — lifecycle + proxy route module (one module, one responsibility: talk to the lile daemon).
- `studio/backend/tests/test_lile_route.py` — pytest covering status/start/stop/proxy.

**Studio frontend — data layer** (`studio/frontend/src/features/lile/`)
- `api/types.ts` — response/event shapes (`HealthReport`, `TrainStepEvent`, `TrajectoryTail`, `CapsuleStatus`, `ChatLileBlock`).
- `api/lile-client.ts` — thin `fetch` wrappers: `getStatus()`, `postStart(opts)`, `postStop()`, `getTrajectoryTail(sinceOffset)`, `postTrain(body)`, `postFeedback(body)`, `getSnapshots()`, `postSnapshot(name)`, `postMerge()`, `postWait()`. SSE helper `streamChatCompletions(body, onChunk)`.
- `stores/lile-capsule-store.ts` — zustand store (see §5.3 of spec).
- `hooks/use-lile-trajectory-poll.ts` — 2s poll driving the store; tolerates old shape.
- `hooks/use-lile-status-poll.ts` — 2s poll driving `status` in the store.

**Studio frontend — UI** (`studio/frontend/src/features/lile/`)
- `components/capsule-status-strip.tsx` — online/offline pill, model name, queue_depth, commit_cursor, merges, ping.
- `components/capsule-load-form.tsx` — form + Load/Stop button.
- `components/charts/loss-chart-card.tsx`
- `components/charts/grad-norm-chart-card.tsx`
- `components/charts/kl-divergence-chart-card.tsx`
- `components/charts/queue-depth-chart-card.tsx`
- `components/charts/components-chart-card.tsx` — auto-discovered secondary series.
- `components/charts/chart-utils.ts` — shared domain/tick helpers (consolidate what we reuse from `features/studio/sections/charts/utils.ts` without importing it; lile series are independent).
- `components/train-tab/chat-sft-card.tsx`
- `components/train-tab/ntp-card.tsx`
- `components/train-tab/reinforce-card.tsx`
- `components/train-tab/advanced-json-card.tsx`
- `components/snapshots-tab.tsx`
- `components/trajectory-tab.tsx`
- `components/lile-message-actions.tsx` — 👍 / 👎 / 💬 / ✎ action group + meta row. Used by `/chat` in lile mode.
- `components/feedback-modal.tsx` — shared modal for critique + rewrite.
- `lile-page.tsx` — grid layout: status strip → charts → tabs.
- `index.ts` — barrel.

**Studio frontend — route**
- `studio/frontend/src/app/routes/lile.tsx` — TanStack Route; register in `app/router.tsx`.

**Tests**
- `studio/frontend/src/features/lile/stores/lile-capsule-store.test.ts`
- `studio/frontend/src/features/lile/hooks/use-lile-trajectory-poll.test.ts`
- `studio/frontend/src/features/lile/components/capsule-status-strip.test.tsx`
- `studio/frontend/src/features/lile/components/train-tab/chat-sft-card.test.tsx`

### Modified

- `studio/backend/main.py` — register lile router.
- `studio/frontend/src/app/router.tsx` — import + add lile route.
- `studio/frontend/src/app/routes/__root.tsx` — add `/lile` nav link if a sidebar nav exists there (check during implementation; skip if nav lives elsewhere).
- `studio/frontend/src/features/chat/stores/chat-runtime-store.ts` — add `lileMode: boolean`, `lileBlockOnLastCommit: boolean`, `lileLastCommit: number | null` with actions + localStorage persistence.
- `studio/frontend/src/features/chat/chat-settings-sheet.tsx` — add two toggles.
- `studio/frontend/src/features/chat/api/chat-adapter.ts` — when `lileMode`, POST to `/api/lile/v1/chat/completions` instead of `/v1/chat/completions`; harvest `lile.*` block into message metadata.
- `studio/frontend/src/features/chat/components/**` — render `<LileMessageActions>` on assistant messages when in lile mode. Exact file located during implementation (component that renders assistant message footer / actions).

---

## Implementation Order

Five phases, each self-contained and testable:

1. **Studio backend route** (Tasks 1–6): isolated; peer and daemon don't need to change.
2. **Frontend data layer** (Tasks 7–10): store, polling, client.
3. **`/lile` page UI** (Tasks 11–17): components, page, route.
4. **`/chat` lile mode** (Tasks 18–21): settings toggles, adapter switch, message actions.
5. **E2E smoke** (Task 22): manual browser check.

After each task: run its tests, commit. Do not batch commits.

---

### Task 1: Scaffold `routes/lile.py` with `/capsule/status`

**Files:**
- Create: `studio/backend/routes/lile.py`
- Modify: `studio/backend/main.py` (register router)
- Test: `studio/backend/tests/test_lile_route.py`

- [ ] **Step 1: Write the failing test**

```python
# studio/backend/tests/test_lile_route.py
import json
import httpx
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd studio/backend && pytest tests/test_lile_route.py::test_status_returns_offline_when_daemon_absent -xvs`
Expected: FAIL with 404 (route does not exist yet).

- [ ] **Step 3: Create the route module (minimal)**

```python
# studio/backend/routes/lile.py
# SPDX-License-Identifier: AGPL-3.0-only
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
```

Register in `main.py` right after existing `app.include_router(...)` calls (around line 183):

```python
from routes.lile import router as lile_router  # new
app.include_router(lile_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd studio/backend && pytest tests/test_lile_route.py::test_status_returns_offline_when_daemon_absent -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/routes/lile.py studio/backend/main.py studio/backend/tests/test_lile_route.py
git commit -m "feat(studio): scaffold lile capsule status endpoint"
```

---

### Task 2: `/capsule/status` returns daemon health when reachable

**Files:**
- Modify: `studio/backend/routes/lile.py`
- Modify: `studio/backend/tests/test_lile_route.py`

- [ ] **Step 1: Write the failing test**

```python
def test_status_returns_health_when_daemon_reachable(client, monkeypatch, respx_mock):
    """When lile /health responds 200, status mirrors the payload and url."""
    import respx
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
```

Add to `conftest.py` if not there: `import respx; @pytest.fixture def respx_mock(): with respx.mock as m: yield m`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lile_route.py::test_status_returns_health_when_daemon_reachable -xvs`
Expected: PASS (already passes from Task 1's minimal implementation). If it fails, debug; otherwise the test is just locking in contract — commit and move on.

- [ ] **Step 3: Commit**

```bash
git add studio/backend/tests/test_lile_route.py studio/backend/tests/conftest.py
git commit -m "test(studio): lock lile capsule status contract"
```

---

### Task 3: `/capsule/start` — health-gated spawn

**Files:**
- Modify: `studio/backend/routes/lile.py`
- Modify: `studio/backend/tests/test_lile_route.py`

- [ ] **Step 1: Write failing tests**

```python
def test_start_noop_when_already_running(client, monkeypatch, respx_mock):
    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59999")
    respx_mock.get("http://127.0.0.1:59999/health").respond(200, json={"ok": True})
    r = client.post("/api/lile/capsule/start", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True
    assert body["externally_managed"] is True


def test_start_spawns_subprocess_when_absent(client, monkeypatch):
    from routes import lile as lile_mod
    spawned = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            spawned["argv"] = argv
            self.pid = 4242

    monkeypatch.setenv("LILE_HOST", "127.0.0.1")
    monkeypatch.setenv("LILE_PORT", "59998")
    monkeypatch.setattr(lile_mod.subprocess, "Popen", FakePopen)
    async def fake_probe():
        return {"ok": True, "model": "qwen3", "queue_depth": 0,
                "commit_cursor": 0, "merges": 0}
    monkeypatch.setattr(lile_mod, "_probe_health", fake_probe)

    r = client.post("/api/lile/capsule/start",
                    json={"model": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"})
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True
    assert body["externally_managed"] is False
    assert body["pid"] == 4242
    assert "lile.server" in " ".join(spawned["argv"])
    assert "--port" in spawned["argv"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lile_route.py -k "start_" -xvs`
Expected: FAIL (404 on POST).

- [ ] **Step 3: Implement `/capsule/start`**

```python
# add to routes/lile.py
import asyncio
import subprocess
import sys
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel


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
    health = await _probe_health()
    if health is not None:
        return {"running": True, "externally_managed": True,
                "health": health, "url": _lile_base_url()}

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lile_route.py -k "start_" -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/routes/lile.py studio/backend/tests/test_lile_route.py
git commit -m "feat(studio): lile capsule start with health-gated spawn"
```

---

### Task 4: `/capsule/stop` — only acts if we spawned

**Files:**
- Modify: `studio/backend/routes/lile.py`
- Modify: `studio/backend/tests/test_lile_route.py`

- [ ] **Step 1: Write failing tests**

```python
def test_stop_refuses_when_externally_managed(client, monkeypatch):
    from routes import lile as lile_mod
    monkeypatch.setattr(lile_mod, "_spawned_pid", None)
    r = client.post("/api/lile/capsule/stop")
    assert r.status_code == 200
    assert r.json() == {"stopped": False, "reason": "externally_managed"}


def test_stop_sends_signal_when_we_spawned(client, monkeypatch):
    from routes import lile as lile_mod
    killed = {}
    monkeypatch.setattr(lile_mod, "_spawned_pid", 4242)
    def fake_kill(pid, sig):
        killed["pid"] = pid; killed["sig"] = sig
    monkeypatch.setattr(lile_mod.os, "kill", fake_kill)
    r = client.post("/api/lile/capsule/stop")
    assert r.status_code == 200
    assert r.json()["stopped"] is True
    assert killed == {"pid": 4242, "sig": lile_mod.signal.SIGTERM}
    assert lile_mod._spawned_pid is None
```

- [ ] **Step 2: Run to verify fails; implement**

```python
# add to routes/lile.py
import signal


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
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_lile_route.py -k "stop" -xvs
git add studio/backend/routes/lile.py studio/backend/tests/test_lile_route.py
git commit -m "feat(studio): lile capsule stop signals only spawned daemons"
```

---

### Task 5: Transparent proxy `/api/lile/{path:path}` (non-streaming)

**Files:**
- Modify: `studio/backend/routes/lile.py`
- Modify: `studio/backend/tests/test_lile_route.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Implement non-streaming proxy**

```python
# add to routes/lile.py
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}


def _forward_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


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
```

Declare `_proxy_stream` as a placeholder that raises `NotImplementedError` for now — it gets implemented in Task 6.

```python
async def _proxy_stream(method: str, url: str, headers: dict, body: bytes):
    raise NotImplementedError  # Task 6
```

**Route registration note:** the `{path:path}` catch-all must be declared *after* `/capsule/*`. FastAPI matches in declaration order, and `/capsule/status` would be shadowed otherwise. Verify the order in the file.

- [ ] **Step 3: Run tests to verify PASS**

Run: `pytest tests/test_lile_route.py -k "proxy" -xvs`
Expected: PASS for the non-SSE cases.

- [ ] **Step 4: Commit**

```bash
git add studio/backend/routes/lile.py studio/backend/tests/test_lile_route.py
git commit -m "feat(studio): transparent proxy for /api/lile/* (non-streaming)"
```

---

### Task 6: SSE pass-through for `/v1/chat/completions`

**Files:**
- Modify: `studio/backend/routes/lile.py`
- Modify: `studio/backend/tests/test_lile_route.py`

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Implement `_proxy_stream`**

```python
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
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_lile_route.py -k "proxy_streams" -xvs
pytest tests/test_lile_route.py -xvs  # full route suite green
git add studio/backend/routes/lile.py studio/backend/tests/test_lile_route.py
git commit -m "feat(studio): SSE pass-through on lile proxy"
```

---

### Task 7: Frontend types + client wrappers

**Files:**
- Create: `studio/frontend/src/features/lile/api/types.ts`
- Create: `studio/frontend/src/features/lile/api/lile-client.ts`
- Create: `studio/frontend/src/features/lile/index.ts`

- [ ] **Step 1: Define types** (no test — types are compile-time only; vitest type-check covers this via other tests)

```ts
// features/lile/api/types.ts
export type HealthReport = {
  ok: boolean;
  model: string;
  queue_depth: number;
  commit_cursor: number;
  merges: number;
};

export type CapsuleStatus =
  | { running: false }
  | { running: true; externally_managed: boolean; health: HealthReport;
      url: string };

export type TrainStepEvent = {
  offset: number;
  kind: "train_step";
  batch_id: number;
  objective: string;
  loss: number;
  batch_size: number;
  commit_token?: number;
  grad_norm_total?: number;
  grad_clipped?: boolean;
  components?: Record<string, number>;
  ts?: number;
};

export type FeedbackEvent = {
  offset: number;
  kind: "feedback";
  response_id: string;
  feedback_kind: "binary" | "critique" | "rewrite";
  value?: unknown;
  ts?: number;
};

export type TrajectoryEvent = TrainStepEvent | FeedbackEvent |
  { offset: number; kind: string; [k: string]: unknown };

export type TrajectoryTail =
  | { events: TrajectoryEvent[]; next_offset: number; total_size: number }
  | { events: TrajectoryEvent[] };  // old shape, back-compat

export type ChatLileBlock = {
  response_id: string;
  commit_cursor: number;
  latency_s: number;
};

export type StartRequest = {
  model?: string;
  max_seq_length?: number;
  lora_rank?: number;
  load_in_4bit?: boolean;
  idle_replay?: boolean;
  frozen_ref?: boolean;
};
```

- [ ] **Step 2: Implement client**

```ts
// features/lile/api/lile-client.ts
import type { CapsuleStatus, StartRequest, TrajectoryTail } from "./types";

const BASE = "/api/lile";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`lile ${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const lileClient = {
  getStatus(): Promise<CapsuleStatus> {
    return fetch(`${BASE}/capsule/status`).then(json);
  },
  postStart(body: StartRequest) {
    return fetch(`${BASE}/capsule/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json);
  },
  postStop() {
    return fetch(`${BASE}/capsule/stop`, { method: "POST" }).then(json);
  },
  getTrajectoryTail(sinceOffset: number): Promise<TrajectoryTail> {
    const q = sinceOffset > 0 ? `?since_offset=${sinceOffset}` : "";
    return fetch(`${BASE}/v1/state/trajectory/tail${q}`).then(json);
  },
  postTrain(body: unknown) {
    return fetch(`${BASE}/v1/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json);
  },
  postFeedback(body: unknown) {
    return fetch(`${BASE}/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json);
  },
  getSnapshots() {
    return fetch(`${BASE}/v1/state/snapshot/list`).then(json);
  },
  postSnapshot(name: string) {
    return fetch(`${BASE}/v1/state/snapshot/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then(json);
  },
  postMerge() {
    return fetch(`${BASE}/v1/state/merge`, { method: "POST" }).then(json);
  },
};
```

- [ ] **Step 3: Barrel**

```ts
// features/lile/index.ts
export { lileClient } from "./api/lile-client";
export type * from "./api/types";
```

- [ ] **Step 4: Commit**

```bash
git add studio/frontend/src/features/lile/
git commit -m "feat(studio): lile api types and client wrapper"
```

---

### Task 8: `lile-capsule-store` (zustand)

**Files:**
- Create: `studio/frontend/src/features/lile/stores/lile-capsule-store.ts`
- Create: `studio/frontend/src/features/lile/stores/lile-capsule-store.test.ts`

- [ ] **Step 1: Write failing test**

```ts
// features/lile/stores/lile-capsule-store.test.ts
import { describe, it, expect, beforeEach } from "vitest";
import { useLileCapsuleStore } from "./lile-capsule-store";

describe("lile-capsule-store", () => {
  beforeEach(() => useLileCapsuleStore.getState().reset());

  it("mergeTail appends events and advances lastOffset", () => {
    const s = useLileCapsuleStore.getState();
    s.mergeTail({
      events: [{ offset: 0, kind: "train_step", loss: 1.0, batch_id: 0,
                 objective: "sft", batch_size: 1 }],
      next_offset: 1, total_size: 1,
    });
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(1);
    expect(useLileCapsuleStore.getState().lastOffset).toBe(1);
  });

  it("mergeTail deduplicates by offset", () => {
    const s = useLileCapsuleStore.getState();
    const ev = { offset: 0, kind: "train_step", loss: 1, batch_id: 0,
                 objective: "sft", batch_size: 1 };
    s.mergeTail({ events: [ev], next_offset: 1, total_size: 1 });
    s.mergeTail({ events: [ev], next_offset: 1, total_size: 1 });
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(1);
  });

  it("mergeTail handles old shape (no next_offset)", () => {
    const s = useLileCapsuleStore.getState();
    s.mergeTail({ events: [{ offset: 0, kind: "train_step", loss: 1,
                             batch_id: 0, objective: "sft", batch_size: 1 }] });
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(1);
    // old shape: lastOffset tracks max offset seen
    expect(useLileCapsuleStore.getState().lastOffset).toBe(1);
  });

  it("setStatus records last commit cursor", () => {
    const s = useLileCapsuleStore.getState();
    s.setStatus({ running: true, externally_managed: false,
                  health: { ok: true, model: "m", queue_depth: 0,
                            commit_cursor: 42, merges: 0 },
                  url: "http://x" });
    expect(useLileCapsuleStore.getState().lastCommitToken).toBe(42);
  });
});
```

- [ ] **Step 2: Run test — expect FAIL (module missing)**

Run: `cd studio/frontend && bun x vitest run src/features/lile/stores/lile-capsule-store.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement the store**

```ts
// features/lile/stores/lile-capsule-store.ts
import { create } from "zustand";
import type {
  CapsuleStatus, TrajectoryEvent, TrajectoryTail,
} from "../api/types";

type State = {
  status: CapsuleStatus | null;
  trajectory: TrajectoryEvent[];
  lastOffset: number;
  totalSize: number;
  lastCommitToken: number | null;

  setStatus(s: CapsuleStatus | null): void;
  mergeTail(t: TrajectoryTail): void;
  reset(): void;
};

const MAX_ROLLING = 2000;  // cap rolling window

export const useLileCapsuleStore = create<State>((set) => ({
  status: null,
  trajectory: [],
  lastOffset: 0,
  totalSize: 0,
  lastCommitToken: null,

  setStatus: (s) =>
    set(() => ({
      status: s,
      lastCommitToken:
        s && s.running ? s.health.commit_cursor : null,
    })),

  mergeTail: (t) =>
    set((prev) => {
      const seen = new Set(prev.trajectory.map((e) => e.offset));
      const fresh = t.events.filter((e) => !seen.has(e.offset));
      const combined = [...prev.trajectory, ...fresh].slice(-MAX_ROLLING);
      const hasNextOffset = "next_offset" in t;
      const nextOffset = hasNextOffset
        ? (t as { next_offset: number }).next_offset
        : combined.length > 0
          ? Math.max(prev.lastOffset, ...combined.map((e) => e.offset + 1))
          : prev.lastOffset;
      const totalSize =
        "total_size" in t && typeof t.total_size === "number"
          ? t.total_size
          : combined.length;
      return { trajectory: combined, lastOffset: nextOffset, totalSize };
    }),

  reset: () =>
    set({ status: null, trajectory: [], lastOffset: 0,
          totalSize: 0, lastCommitToken: null }),
}));
```

- [ ] **Step 4: Run — expect PASS. Commit**

```bash
bun x vitest run src/features/lile/stores/lile-capsule-store.test.ts
git add studio/frontend/src/features/lile/stores/
git commit -m "feat(studio): lile-capsule-store with rolling window + dedup"
```

---

### Task 9: `useLileTrajectoryPoll` hook

**Files:**
- Create: `studio/frontend/src/features/lile/hooks/use-lile-trajectory-poll.ts`
- Create: `studio/frontend/src/features/lile/hooks/use-lile-trajectory-poll.test.ts`

- [ ] **Step 1: Write failing test**

```ts
// features/lile/hooks/use-lile-trajectory-poll.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";
import { useLileTrajectoryPoll } from "./use-lile-trajectory-poll";
import { lileClient } from "../api/lile-client";

describe("useLileTrajectoryPoll", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useLileCapsuleStore.getState().reset();
  });
  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

  it("polls every 2s and merges with since_offset", async () => {
    const get = vi.spyOn(lileClient, "getTrajectoryTail")
      .mockResolvedValueOnce({ events: [
        { offset: 0, kind: "train_step", loss: 1, batch_id: 0,
          objective: "sft", batch_size: 1 },
      ], next_offset: 1, total_size: 1 })
      .mockResolvedValueOnce({ events: [
        { offset: 1, kind: "train_step", loss: 0.9, batch_id: 1,
          objective: "sft", batch_size: 1 },
      ], next_offset: 2, total_size: 2 });

    renderHook(() => useLileTrajectoryPoll({ enabled: true }));
    await act(async () => { await Promise.resolve(); });
    expect(get).toHaveBeenNthCalledWith(1, 0);

    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(get).toHaveBeenNthCalledWith(2, 1);
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(2);
  });

  it("tolerates old shape (no next_offset)", async () => {
    vi.spyOn(lileClient, "getTrajectoryTail").mockResolvedValue({
      events: [{ offset: 5, kind: "train_step", loss: 0.5, batch_id: 5,
                 objective: "sft", batch_size: 1 }],
    });
    renderHook(() => useLileTrajectoryPoll({ enabled: true }));
    await act(async () => { await Promise.resolve(); });
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run — expect FAIL; implement**

```ts
// features/lile/hooks/use-lile-trajectory-poll.ts
import { useEffect, useRef } from "react";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

export function useLileTrajectoryPoll(opts: { enabled: boolean }) {
  const enabledRef = useRef(opts.enabled);
  enabledRef.current = opts.enabled;

  useEffect(() => {
    if (!opts.enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function tick() {
      if (cancelled || !enabledRef.current) return;
      const lastOffset = useLileCapsuleStore.getState().lastOffset;
      try {
        const tail = await lileClient.getTrajectoryTail(lastOffset);
        if (!cancelled) useLileCapsuleStore.getState().mergeTail(tail);
      } catch {
        // network hiccup — backoff handled by status poll
      }
      if (!cancelled) timer = setTimeout(tick, 2000);
    }
    void tick();

    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [opts.enabled]);
}
```

- [ ] **Step 3: Commit**

```bash
bun x vitest run src/features/lile/hooks/use-lile-trajectory-poll.test.ts
git add studio/frontend/src/features/lile/hooks/
git commit -m "feat(studio): incremental trajectory poll hook"
```

---

### Task 10: `useLileStatusPoll` hook

**Files:**
- Create: `studio/frontend/src/features/lile/hooks/use-lile-status-poll.ts`

- [ ] **Step 1: Implement** (skip explicit test; covered by `CapsuleStatusStrip` test in Task 11)

```ts
// features/lile/hooks/use-lile-status-poll.ts
import { useEffect, useRef } from "react";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

export function useLileStatusPoll(opts: { enabled: boolean }) {
  const enabledRef = useRef(opts.enabled);
  enabledRef.current = opts.enabled;
  const backoffRef = useRef(2000);

  useEffect(() => {
    if (!opts.enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function tick() {
      if (cancelled || !enabledRef.current) return;
      try {
        const s = await lileClient.getStatus();
        if (!cancelled) {
          useLileCapsuleStore.getState().setStatus(s);
          backoffRef.current = 2000;
        }
      } catch {
        backoffRef.current = Math.min(backoffRef.current * 2, 5000);
        if (!cancelled) useLileCapsuleStore.getState().setStatus({ running: false });
      }
      if (!cancelled) timer = setTimeout(tick, backoffRef.current);
    }
    void tick();
    return () => { cancelled = true; if (timer !== null) clearTimeout(timer); };
  }, [opts.enabled]);
}
```

- [ ] **Step 2: Commit**

```bash
git add studio/frontend/src/features/lile/hooks/use-lile-status-poll.ts
git commit -m "feat(studio): lile status poll with 2→5s backoff"
```

---

### Task 11: `CapsuleStatusStrip` component

**Files:**
- Create: `studio/frontend/src/features/lile/components/capsule-status-strip.tsx`
- Create: `studio/frontend/src/features/lile/components/capsule-status-strip.test.tsx`

- [ ] **Step 1: Write failing test**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect, beforeEach } from "vitest";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";
import { CapsuleStatusStrip } from "./capsule-status-strip";

describe("CapsuleStatusStrip", () => {
  beforeEach(() => useLileCapsuleStore.getState().reset());

  it("shows offline when store has no status", () => {
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/offline/i)).toBeInTheDocument();
  });

  it("shows model name and commit when online", () => {
    useLileCapsuleStore.getState().setStatus({
      running: true, externally_managed: false,
      health: { ok: true, model: "qwen3-0.6b", queue_depth: 3,
                commit_cursor: 77, merges: 2 },
      url: "http://127.0.0.1:8765",
    });
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/qwen3-0.6b/)).toBeInTheDocument();
    expect(screen.getByText(/77/)).toBeInTheDocument();
    expect(screen.getByText(/3/)).toBeInTheDocument();  // queue_depth
  });
});
```

- [ ] **Step 2: Implement**

```tsx
// components/capsule-status-strip.tsx
import { Badge } from "@/components/ui/badge";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

export function CapsuleStatusStrip() {
  const status = useLileCapsuleStore((s) => s.status);
  if (!status || !status.running) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Badge variant="secondary">offline</Badge>
        <span>Lile daemon not reachable</span>
      </div>
    );
  }
  const h = status.health;
  return (
    <div className="flex items-center gap-4 text-sm">
      <Badge variant="default">online</Badge>
      <span className="font-mono">{h.model}</span>
      <span>queue {h.queue_depth}</span>
      <span>commit {h.commit_cursor}</span>
      <span>merges {h.merges}</span>
      {status.externally_managed && <Badge variant="outline">external</Badge>}
    </div>
  );
}
```

- [ ] **Step 3: Run + commit**

```bash
bun x vitest run src/features/lile/components/capsule-status-strip.test.tsx
git add studio/frontend/src/features/lile/components/capsule-status-strip.*
git commit -m "feat(studio): CapsuleStatusStrip with offline/online states"
```

---

### Task 12: `CapsuleLoadForm` component

**Files:**
- Create: `studio/frontend/src/features/lile/components/capsule-load-form.tsx`

- [ ] **Step 1: Implement** (no dedicated test — exercised by Task 22 E2E)

Form fields (shadcn `Input` / `Select` / `Switch`):
- Model (defaults to env-derived default, editable; suggestion list in component: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`, `unsloth/Llama-3.2-1B-unsloth-bnb-4bit`).
- `max_seq_length` (number, default 2048).
- `lora_rank` (number, default 16).
- `load_in_4bit` (switch, default on).
- `idle_replay` (switch, default off).
- `frozen_ref` (switch, default off).

On submit: disable form, call `lileClient.postStart(body)`, show progress ("Loading model… this can take 2 min"). When status poll observes `running: true`, collapse form into a "Stop capsule" button that calls `postStop`. Show any returned error message inline.

- [ ] **Step 2: Commit**

```bash
git add studio/frontend/src/features/lile/components/capsule-load-form.tsx
git commit -m "feat(studio): CapsuleLoadForm with Load/Stop lifecycle"
```

---

### Task 13: Chart primitives

**Files:**
- Create: `studio/frontend/src/features/lile/components/charts/chart-utils.ts`
- Create: `studio/frontend/src/features/lile/components/charts/loss-chart-card.tsx`
- Create: `studio/frontend/src/features/lile/components/charts/grad-norm-chart-card.tsx`
- Create: `studio/frontend/src/features/lile/components/charts/kl-divergence-chart-card.tsx`
- Create: `studio/frontend/src/features/lile/components/charts/queue-depth-chart-card.tsx`
- Create: `studio/frontend/src/features/lile/components/charts/components-chart-card.tsx`

- [ ] **Step 1: Shared selectors in `chart-utils.ts`**

Each card derives its series via a `useLileCapsuleStore` selector:

```ts
export function selectLossSeries(s) {
  return s.trajectory
    .filter((e) => e.kind === "train_step" && typeof e.loss === "number")
    .map((e) => ({ step: e.batch_id, value: e.loss }));
}
export function selectGradNormSeries(s) { /* e.grad_norm_total */ }
export function selectKlSeries(s) { /* e.components?.["batch.kl.loss"] ?? e.components?.kl ?? null */ }
export function selectQueueDepthSeries(s) { /* derived from status timeline — see Task 17 */ }
export function selectComponentsSeries(s) {
  const keys = new Set<string>();
  for (const e of s.trajectory) {
    if (e.kind !== "train_step" || !("components" in e)) continue;
    for (const k of Object.keys(e.components ?? {})) keys.add(k);
  }
  return Array.from(keys).map((k) => ({
    key: k,
    points: s.trajectory
      .filter((e) => e.kind === "train_step" && e.components?.[k] !== undefined)
      .map((e) => ({ step: e.batch_id, value: e.components![k] })),
  }));
}
```

- [ ] **Step 2: Implement cards**

Each card is a thin wrapper around shadcn's `Card` + Recharts' `LineChart` (mirror the pattern in `studio/frontend/src/features/studio/sections/charts/training-loss-chart-card.tsx`). Keep it minimal: single series, auto-fit Y domain, step X-axis.

Keep the primary series hardcoded: `loss`, `grad_norm_total`, `components.batch.kl.loss` (fallback to `components.kl` if present). `queue-depth` is a line over the short status history — store just the current `queue_depth` at the moment the status poll ticks (defer rolling history to `lile-capsule-store.ts` in Task 17 if not already there).

- [ ] **Step 3: Commit**

```bash
git add studio/frontend/src/features/lile/components/charts/
git commit -m "feat(studio): lile chart cards (loss/grad_norm/kl/queue/components)"
```

---

### Task 14: Train-tab cards

**Files:**
- Create: `studio/frontend/src/features/lile/components/train-tab/chat-sft-card.tsx`
- Create: `studio/frontend/src/features/lile/components/train-tab/ntp-card.tsx`
- Create: `studio/frontend/src/features/lile/components/train-tab/reinforce-card.tsx`
- Create: `studio/frontend/src/features/lile/components/train-tab/advanced-json-card.tsx`
- Create: `studio/frontend/src/features/lile/components/train-tab/chat-sft-card.test.tsx`

- [ ] **Step 1: Write failing test for ChatSftCard**

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect } from "vitest";
import { lileClient } from "../../api/lile-client";
import { ChatSftCard } from "./chat-sft-card";

describe("ChatSftCard", () => {
  it("POSTs /v1/train with {objective:'sft', samples:[{messages}]}", async () => {
    const spy = vi.spyOn(lileClient, "postTrain").mockResolvedValue({ queued: true });
    render(<ChatSftCard />);
    await userEvent.type(screen.getByLabelText(/user/i), "hi");
    await userEvent.type(screen.getByLabelText(/assistant/i), "hello");
    await userEvent.click(screen.getByRole("button", { name: /train/i }));
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({
      objective: "sft",
      samples: expect.arrayContaining([
        expect.objectContaining({
          messages: expect.arrayContaining([
            { role: "user", content: "hi" },
            { role: "assistant", content: "hello" },
          ]),
        }),
      ]),
    }));
  });
});
```

- [ ] **Step 2: Implement cards** (mirror demo.html UX)

- `chat-sft-card.tsx` — multi-turn editor (rows of role+content); "Train" sends `{objective: "sft", samples: [{messages}]}`.
- `ntp-card.tsx` — textarea; sends `{objective: "ntp", samples: [{text}]}`.
- `reinforce-card.tsx` — lists recent `feedback` events from the store; "replay" button re-POSTs via `postFeedback` (`fbPayloadFromTraj` logic from demo.html).
- `advanced-json-card.tsx` — raw JSON textarea → POST to `/v1/train` as-is (validated client-side as JSON before send).

- [ ] **Step 3: Run + commit**

```bash
bun x vitest run src/features/lile/components/train-tab/chat-sft-card.test.tsx
git add studio/frontend/src/features/lile/components/train-tab/
git commit -m "feat(studio): lile train-tab cards (sft/ntp/reinforce/advanced)"
```

---

### Task 15: Snapshots + Trajectory tabs

**Files:**
- Create: `studio/frontend/src/features/lile/components/snapshots-tab.tsx`
- Create: `studio/frontend/src/features/lile/components/trajectory-tab.tsx`

- [ ] **Step 1: Implement snapshots-tab**

`Table` + "New snapshot" input + Save/Merge buttons. On mount: `lileClient.getSnapshots()`. Save: `postSnapshot(name)` then refresh. Merge: `postMerge()` then re-poll status (peer-owned endpoint behavior).

- [ ] **Step 2: Implement trajectory-tab**

Virtualized (or capped at 200) list of `useLileCapsuleStore(s => s.trajectory)` reversed (newest first). Each row: kind pill, batch_id / offset, loss/grad summary, expandable JSON. For `feedback` kind add a "Replay" button that re-posts via `postFeedback`.

- [ ] **Step 3: Commit**

```bash
git add studio/frontend/src/features/lile/components/snapshots-tab.tsx studio/frontend/src/features/lile/components/trajectory-tab.tsx
git commit -m "feat(studio): lile snapshots and trajectory tabs"
```

---

### Task 16: `LileMessageActions` + `FeedbackModal` (shared by `/chat` mode)

**Files:**
- Create: `studio/frontend/src/features/lile/components/lile-message-actions.tsx`
- Create: `studio/frontend/src/features/lile/components/feedback-modal.tsx`

- [ ] **Step 1: Implement**

`LileMessageActions` props: `{ responseId: string; commitCursor: number; latencyS?: number }`.
Renders 👍 / 👎 / 💬 / ✎ + meta row. On click:
- 👍 / 👎 → `postFeedback({ response_id, kind: "binary", value: true|false })`.
- 💬 → open FeedbackModal in "critique" mode (textarea).
- ✎ → open FeedbackModal in "rewrite" mode (editable assistant text).
- Button flashes `ok` / `err` and settles.

`FeedbackModal` uses `Dialog` from shadcn.

- [ ] **Step 2: Commit**

```bash
git add studio/frontend/src/features/lile/components/lile-message-actions.tsx studio/frontend/src/features/lile/components/feedback-modal.tsx
git commit -m "feat(studio): LileMessageActions + FeedbackModal"
```

---

### Task 17: `lile-page.tsx` + route registration

**Files:**
- Create: `studio/frontend/src/features/lile/lile-page.tsx`
- Create: `studio/frontend/src/app/routes/lile.tsx`
- Modify: `studio/frontend/src/app/router.tsx`

- [ ] **Step 1: Implement `lile-page.tsx`**

Grid layout:
```
┌────────────────────────────────────────────────────────┐
│  CapsuleStatusStrip                                     │
├────────────────────────────────────────────────────────┤
│  [Loss]  [GradNorm]  [KL]  [Queue]                     │
│  [Components (auto-grid)]                              │
├────────────────────────────────────────────────────────┤
│  Tabs: Train | Trajectory | Snapshots                   │
│  ┌ Train tab body (cards stacked)                    ┐ │
│  │ ChatSftCard / NtpCard / ReinforceCard / AdvJson  │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

Enable `useLileStatusPoll` and `useLileTrajectoryPoll` while this page is mounted. If `status.running === false` render the page with the `CapsuleLoadForm` above the tabs and empty-state hints inside each card.

- [ ] **Step 2: Create route**

```tsx
// app/routes/lile.tsx
import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const LilePage = lazy(() =>
  import("@/features/lile/lile-page").then((m) => ({ default: m.LilePage }))
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/lile",
  beforeLoad: () => requireAuth(),
  component: LilePage,
});
```

- [ ] **Step 3: Register in `router.tsx`**

Add `import { Route as lileRoute } from "./routes/lile";` and include in `rootRoute.addChildren([...])` alongside `studioRoute`, `chatRoute`.

- [ ] **Step 4: Add nav link** (if the app has a sidebar; locate during task — grep `NavLink\|Link.*to="/studio"` to find the nav list component, and add a sibling `/lile` entry).

- [ ] **Step 5: Commit**

```bash
git add studio/frontend/src/features/lile/lile-page.tsx studio/frontend/src/app/routes/lile.tsx studio/frontend/src/app/router.tsx
git commit -m "feat(studio): /lile page and route wiring"
```

---

### Task 18: Chat runtime store additions

**Files:**
- Modify: `studio/frontend/src/features/chat/stores/chat-runtime-store.ts`

- [ ] **Step 1: Add state**

Add `lileMode: boolean`, `lileBlockOnLastCommit: boolean`, actions `setLileMode`, `setLileBlockOnLastCommit`. Persist via the existing localStorage pattern in that store (follow the file's existing persistence idiom; grep for similar `setItem` calls).

- [ ] **Step 2: Commit**

```bash
git add studio/frontend/src/features/chat/stores/chat-runtime-store.ts
git commit -m "feat(chat): lileMode and lileBlockOnLastCommit state"
```

---

### Task 19: Chat settings sheet toggles

**Files:**
- Modify: `studio/frontend/src/features/chat/chat-settings-sheet.tsx`

- [ ] **Step 1: Add toggles** in a new "Lile" section of the sheet:

- "Lile mode" → binds to `lileMode`.
- "Block on last commit" → binds to `lileBlockOnLastCommit` (disabled when `lileMode` is off).

- [ ] **Step 2: Commit**

```bash
git add studio/frontend/src/features/chat/chat-settings-sheet.tsx
git commit -m "feat(chat): settings sheet — Lile mode + block-on-last-commit"
```

---

### Task 20: Chat adapter routes to lile when `lileMode`

**Files:**
- Modify: `studio/frontend/src/features/chat/api/chat-adapter.ts`
- Modify: `studio/frontend/src/features/chat/api/chat-api.ts` (only if URL is constructed there)

- [ ] **Step 1: Switch base URL on lile mode**

Inside `createOpenAIStreamAdapter` (or whatever its caller is, see `chat-adapter.ts:13` `streamChatCompletions`), read `useChatRuntimeStore.getState().lileMode`. When true:

- URL `/api/lile/v1/chat/completions` instead of `/v1/chat/completions`.
- Include `after_commit_token: lileLastCommit` in the body when `lileBlockOnLastCommit` is on and `lileLastCommit !== null`.

Harvest `lile.*` block from the non-SSE response (`body.lile`) and the final SSE chunk into message metadata under `custom.lile = { response_id, commit_cursor, latency_s }`. Also update `useLileCapsuleStore.getState().setStatus(...)` optimistically when a new `commit_cursor` arrives (or just set `lastCommitToken`; simplest: trigger a `getStatus` poll tick on stream-complete).

- [ ] **Step 2: Commit**

```bash
git add studio/frontend/src/features/chat/api/
git commit -m "feat(chat): route to /api/lile when lileMode, harvest lile metadata"
```

---

### Task 21: Render `LileMessageActions` on assistant messages in lile mode

**Files:**
- Modify: whichever chat message component renders assistant action row (locate via `grep -r "MessageActions\|assistant.*actions" studio/frontend/src/features/chat/components`).

- [ ] **Step 1: Wrap the action row** so that when `useChatRuntimeStore(s => s.lileMode)` is true and the message has `metadata.custom.lile`, we render `<LileMessageActions {...message.metadata.custom.lile} />` adjacent to the existing action group.

- [ ] **Step 2: Commit**

```bash
git add studio/frontend/src/features/chat/
git commit -m "feat(chat): render LileMessageActions for lile-mode responses"
```

---

### Task 22: E2E smoke (manual, dev server)

**Prerequisites:**
- Studio backend running on `:8888` with `UNSLOTH_DISABLE_AUTH=1` (local dev) against a Python env that has `fastapi`, `httpx`, `uvicorn`, `structlog`. The worktree's `.venv` is empty — either use the repo's main `.venv` or install backend deps with `uv pip install -r studio/backend/requirements/base.txt` into a fresh venv.
- Lile daemon accessible (either the frontend launches it via `/capsule/start`, or you pre-start it).
- Frontend dev server: `cd studio/frontend && bun run dev`.

- [ ] **Step 1: `/lile` offline state**

Open `http://localhost:5173/lile` without a lile daemon running. Status strip shows "offline". `CapsuleLoadForm` visible. Charts and tabs render empty-state.

- [ ] **Step 2: Load capsule**

Submit the form with defaults. Watch `lile_data/daemon.log` (or the `tail -F` on `/tmp/lile-studio-backend.log`). Within 120 s the status strip flips to "online" showing model / queue / commit.

- [ ] **Step 3: Train a step**

Use `ChatSftCard`, enter `user: "what's 2+2"` and `assistant: "4"`, click Train. Within 2 s the trajectory tab shows a new `train_step` event with `loss`, `grad_norm_total`, and (post-peer-PR) `components.*`. Charts tick.

- [ ] **Step 4: Chat + feedback**

Go to `/chat`, open settings → enable Lile mode. Send "hello". Observe `lile.response_id` / `commit_cursor` in the inline meta row. Click 👎. Trajectory tab shows a `feedback` event of kind `binary`.

- [ ] **Step 5: Toggle block-on-last-commit**

In settings enable "Block on last commit". Send another message. Inspect the request body in DevTools Network — it contains `after_commit_token`.

- [ ] **Step 6: Stop capsule**

Click Stop in the status strip. Status flips to offline. Daemon process exits. `/lile` returns to empty-state.

If all six steps pass: tag merge-ready. Otherwise, file bugs and fix.

---

## Phase boundaries / parallelism notes

- Tasks 1–6 (backend) can run in parallel with 7–10 (frontend data layer) — they don't overlap.
- Task 11 onward depends on 7–10.
- Task 18–21 depend on 8 (store) and 16 (actions component); otherwise independent of `/lile` page.
- Tests: run the full pytest route suite after Task 6, full vitest suite after each frontend task.

## Dev-environment gotcha (surface early)

The worktree `.venv/` is empty. Before Task 1's test can execute, set up the backend env:

```bash
cd /home/me/ht/forks/ht-unsloth/.worktrees/lile-opus4.7
uv venv .venv
.venv/bin/uv pip install -r studio/backend/requirements/base.txt
# OR reuse the main repo's venv by setting VIRTUAL_ENV accordingly
```

If `uv pip install` tries to download Windows wheels (xformers), override the `requirements.txt` `platform_system` markers or install only the subset that studio backend tests need: `fastapi`, `httpx`, `uvicorn`, `pytest`, `pytest-asyncio`, `respx`, `structlog`, `pydantic`.

