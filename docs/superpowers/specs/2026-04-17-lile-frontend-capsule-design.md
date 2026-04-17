# Lile Frontend Capsule — Design

**Status:** Draft · **Date:** 2026-04-17 · **Branch:** `feat/lile-livelearn`

## 1. Context and motivation

`lile` is the in-process online fine-tuning daemon shipped in this fork — a FastAPI server that holds a single LoRA-adapted model, serializes all weight-mutating work through a compute queue, and exposes OpenAI-compatible chat plus `/v1/train`, `/v1/feedback`, `/v1/state/*`. Today the only UI is a standalone prototype at `/tmp/lile-trial/demo.html` fronted by a stdlib proxy (`/tmp/lile-trial/proxy.py`). There is no studio integration.

We want lile to be a first-class feature of **Unsloth Studio**: a loadable "capsule" you can boot inside the studio tab, talk to, teach, and watch its health and training metrics as it runs. The studio already has the chart, chat, and routing infrastructure needed — we're stitching lile into it, not reinventing.

Four metrics are explicitly in scope for v1 dashboards: **training loss**, **gradient norm**, **health metrics** (queue depth, commit cursor, merges, model fingerprint, online/offline), and **KL divergence** (from KL-anchor batch objective). These are the user's stated ask.

## 2. Goals and non-goals

**Goals**

- A new `/lile` route in studio frontend presenting: capsule status strip, live metric charts, training controls, trajectory tail, snapshot manager.
- An extended "lile mode" on the existing `/chat` route that routes chat through lile's `/v1/chat/completions` with per-message feedback buttons and commit-cursor metadata.
- Studio backend proxy + lifecycle route at `/api/lile/*` that detects a running lile daemon, launches one as an independent microservice if absent, and transparently forwards HTTP (including SSE) to it.
- Lile-side plumbing to surface per-step `grad_norm_total`, `grad_clipped`, and per-objective components (including KL) into the trajectory log. Owned by a peer agent in parallel (§ 6).

**Non-goals**

- Multi-GPU or multi-capsule simultaneous execution. API is multi-capsule-ready (capsule_id threaded through); v1 runtime is singleton on 1× RTX 3090.
- Remote-lile / production deployment concerns. v1 assumes the daemon runs locally on the studio host.
- GGUF export, Ollama bridging, or dataset ingestion flows for lile. Those live in the existing Unsloth studio training path; lile's contract is "prompt, response, feedback, watch."
- Changing the existing HF-style batch training view. It stays intact.

## 3. Architecture overview

```
┌─────────────────────────────────────── studio frontend (Vite, React) ───┐
│                                                                          │
│   /lile  (new)                      /chat  (existing, mode-extended)    │
│   ├─ capsule status strip           ├─ composer / thread sidebar         │
│   ├─ charts grid                    ├─ [lile mode toggle]                │
│   │   loss · grad_norm · KL · ...   └─ per-message action bar when on    │
│   └─ tabs: Train / Traj / Snap         (👍 / 👎 / critique / rewrite)    │
│                                                                          │
└────────────── /api/lile/* (Vite proxy → studio backend :8888) ──────────┘
                         │
┌────────────────────────▼─────────── studio backend (FastAPI, Python) ───┐
│                                                                          │
│   routes/lile.py                                                         │
│   ├─ GET  /api/lile/capsule/status   detect via lile /health            │
│   ├─ POST /api/lile/capsule/start    health-gated spawn of microservice │
│   ├─ POST /api/lile/capsule/stop                                         │
│   └─ ANY  /api/lile/v1/** and /api/lile/health → transparent proxy      │
│                                            (SSE-aware pass-through)     │
│                                                                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │   lile daemon   │   independent process (subprocess or
                │   :8000 (def.)  │   externally managed), not embedded.
                └─────────────────┘
```

Key invariants:

- **One daemon owner at a time.** `/capsule/start` hard-gates on `/health`; if lile is already reachable, we return `{running: true, externally_managed: true}` and do not spawn. Prevents double-binding the GPU.
- **Transparent proxy.** `/api/lile/*` strips the `/api/lile` prefix and forwards to the lile daemon verbatim, including `text/event-stream` bodies (no buffering, `x-accel-buffering: no`). Models the proven pass-through in `/tmp/lile-trial/proxy.py`.
- **No cross-origin.** Browser only talks to studio backend; studio backend talks to lile. CORS surface on lile is irrelevant.

## 4. Studio backend — `studio/backend/routes/lile.py`

Single new module. Lives alongside existing `training.py`, `inference.py`, etc. Registered in `studio/backend/main.py` the same way.

**Configuration** (env + sensible defaults):

| Var                     | Default                         | Purpose |
|-------------------------|---------------------------------|---------|
| `LILE_HOST`             | `127.0.0.1`                     | daemon host |
| `LILE_PORT`             | `8765`                          | daemon port (matches `/tmp/lile-trial` convention) |
| `LILE_MODEL`            | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` | default model on start |
| `LILE_DATA_DIR`         | `<repo>/lile_data`              | trajectory + snapshots |
| `LILE_LAUNCH_CMD`       | `python -m lile.server`         | override for custom launchers |

**Endpoints:**

- `GET /api/lile/capsule/status` — best-effort HTTP GET on `http://${LILE_HOST}:${LILE_PORT}/health` with 500 ms timeout.
  - Returns `{running: false}` on timeout or connection refused.
  - Returns `{running: true, externally_managed: bool, health: <health JSON>, url: "http://..."}` on success. `externally_managed` is `true` iff this studio process didn't spawn the daemon (tracked by a local `_spawned_pid` cell; persists across studio restarts as `false` — we don't try to adopt orphan processes).
- `POST /api/lile/capsule/start {model?, max_seq_length?, lora_rank?, load_in_4bit?, idle_replay?, frozen_ref?}`
  1. Call `/health` once. If reachable → return `{running: true, externally_managed: true}`. No spawn.
  2. Else `subprocess.Popen([python, "-m", "lile.server", "--port", str(LILE_PORT), ...])` as a **detached** process (new session, stdout → `${LILE_DATA_DIR}/daemon.log`). Track PID.
  3. Poll `/health` for up to 120 s (model load is slow).
  4. Return `{running: true, externally_managed: false, pid, url, health}` on success; `{running: false, error: "..."}` on timeout.
- `POST /api/lile/capsule/stop` — only acts if we spawned. Otherwise returns `{stopped: false, reason: "externally_managed"}`.
- `ANY /api/lile/{path:path}` — proxy. Uses `httpx.AsyncClient` with `follow_redirects=False` and explicit streaming for SSE (`Accept: text/event-stream`). Forwards method, headers (minus hop-by-hop), body. Streams response back chunked. Errors → 502 with `{error: "proxy upstream failure"}`.

**File sketch:**

```python
# studio/backend/routes/lile.py
router = APIRouter(prefix="/api/lile", tags=["lile"])

@router.get("/capsule/status") ...
@router.post("/capsule/start") ...
@router.post("/capsule/stop") ...

@router.api_route("/{path:path}",
                  methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"])
async def proxy(path: str, request: Request): ...
```

Included into `main.py`:

```python
from .routes.lile import router as lile_router
app.include_router(lile_router)
```

## 5. Studio frontend

### 5.1 New route `/lile`

New files under `studio/frontend/src/features/lile/`:

```
features/lile/
├─ index.ts                       barrel
├─ api/
│  ├─ lile-client.ts              fetch wrappers + SSE stream helpers
│  └─ types.ts                    response/event shapes
├─ stores/
│  └─ lile-capsule-store.ts       zustand: status, rolling trajectory
├─ hooks/
│  └─ use-lile-trajectory-poll.ts incremental poll with next_offset
├─ components/
│  ├─ capsule-status-strip.tsx    model / queue / commit / merges / ping
│  ├─ capsule-load-form.tsx       model + lora params + Load/Stop
│  ├─ charts/
│  │  ├─ loss-chart-card.tsx
│  │  ├─ grad-norm-chart-card.tsx
│  │  ├─ kl-divergence-chart-card.tsx
│  │  ├─ queue-depth-chart-card.tsx
│  │  └─ components-chart-card.tsx  auto-discovered secondary series
│  ├─ train-tab/
│  │  ├─ chat-sft-card.tsx         multi-turn editor, port of demo.html
│  │  ├─ ntp-card.tsx
│  │  ├─ reinforce-card.tsx        replay last / list feedback events
│  │  └─ advanced-json-card.tsx
│  ├─ snapshots-tab.tsx
│  └─ trajectory-tab.tsx           auto-tail, pills, replay inline
└─ lile-page.tsx                   grid: status → charts → tabs
```

Route registered in `app/routes/lile.tsx` using TanStack Router, matching the pattern of `chat.tsx`, `studio.tsx`.

**Existing pieces reused verbatim:**
- Chart card primitives under `features/studio/sections/charts/` provide the visual baseline. New lile charts are thin variants wiring different data streams into the same component.
- shadcn/ui components, Tailwind, the existing sidebar navigation layout.

### 5.2 `/chat` — lile mode

Surgical change. Three additions:

1. **Settings sheet toggle** ("Lile mode: on/off"). When on, persists to localStorage + `useChatRuntimeStore`.
2. **Runtime adapter switch.** `createOpenAIStreamAdapter` already targets an OpenAI-shaped endpoint; lile mode sets the base URL to `/api/lile` so requests go through the proxy.
3. **Per-message action bar.** Assistant messages in lile mode render an extra action group (wrapped in a `LileMessageActions` component) with 👍 / 👎 / 💬 critique / ✎ rewrite buttons plus a meta row (`response_id`, `commit_cursor`, `latency`). Critique + rewrite open the same modal ported from demo.html.
4. **"Block on last commit" toggle.** Also in the settings sheet. When on, chat requests include `after_commit_token: <lastCommit>`.

`response_id` and `commit_cursor` are harvested from lile's `lile.*` block in the chat-completion payload (both SSE and non-SSE paths — see `server.py`).

### 5.3 Data store & polling

`lile-capsule-store.ts` — Zustand store with:

```ts
type LileCapsuleState = {
  status: { running: boolean; externallyManaged?: boolean; health?: Health } | null;
  trajectory: TrainStepEvent[];           // rolling window keyed by offset
  lastOffset: number;                     // next_offset from last poll
  totalSize: number;                      // from tail response
  connect(): void;                        // start polling
  disconnect(): void;
  lastCommitToken: number | null;
};
```

Polling cadence: **2 s** for both `/capsule/status` and `/v1/state/trajectory/tail?since_offset=<lastOffset>`. We rely on lile's (peer-committed) incremental tail shape `{events, next_offset, total_size}`, but fall back gracefully if `next_offset` is missing (old shape) — poll full window every tick until the backend PR lands.

Charts subscribe to the store via selectors, mapping the rolling window to per-series arrays. Series discovery:
- **Primary series** are hardcoded: `loss`, `grad_norm_total`, `components.batch.kl.loss` (if present).
- **Secondary series** are auto-discovered: any `components.*` key is charted in the catch-all "Components" card. Keeps us forward-compatible as new objectives ship.

## 6. Lile backend changes (peer-owned)

These are in the peer's scope (pane %9, branch follow-up to PR#8 on `ht`):

1. Capture and return `grad_norm_total` from `torch.nn.utils.clip_grad_norm_` (currently the return value is dropped in `lile/engine/train.py:101`). Also expose `grad_clipped: bool`.
2. Extend `TrajectoryLog.log_train(...)` signature with optional `components: dict | None = None`. Emit the dict as extra keys on the `train_step` record. Default `None` preserves back-compat with existing callers and fixture tests.
3. `Controller._handle_task("train")` threads the `components` dict from the objective result into `log_train`.
4. Locked component keys (peer pre-committed): `kto_z0`, `kto_z0_source`, `kto_loss`, `ccpd_kl_mean`, `ccpd_mean_logprob`, `coh_margin`, `coh_good_logprob`, `coh_bad_logprob`, `sft_loss`, `ntp_loss`, plus `batch.kl.*` already emitted today.
5. `/v1/state/trajectory/tail` response becomes `{events, next_offset, total_size}` with an optional `since_offset` query param. Preserves old shape when `since_offset` is omitted (back-compat).

Frontend code tolerates both old and new shapes so we aren't blocked on merge order.

## 7. Failure modes & handling

| Failure | Handling |
|---|---|
| Lile daemon not running, user hits `/lile` | Status strip shows "offline" pill; `capsule-load-form` visible with "Load Capsule" button. All tabs show empty-state hints. |
| `/capsule/start` timeout (model load > 120s) | Return `{running: false, error}`; frontend keeps polling status; UI shows "still loading — check daemon.log". |
| Proxy upstream 5xx or connection drop | Status strip flips to "offline"; polling backs off to 5 s for 30 s then returns to 2 s. |
| Chat in lile mode, daemon goes away mid-stream | Adapter catches fetch error, surfaces as inline system message "lile daemon disconnected"; toggles "offline" pill; chat composer disabled until reconnect. |
| Feedback POST fails | Button flips to `err` state with error text in tooltip; user can retry. |
| Trajectory tail returns old shape (no `next_offset`) | Fall back to full-window poll; log a dev-mode warning once. |

## 8. Test plan

**Studio backend (pytest, `studio/backend/tests/test_lile_route.py`)**
- Status route returns `{running: false}` when nothing on `LILE_PORT`.
- Status route returns `{running: true, externally_managed: true}` against a mock lile `/health`.
- Start route is a no-op when lile already healthy.
- Start route spawns subprocess when lile absent (mock Popen, verify args).
- Proxy route forwards GET/POST with body preserved and status code mirrored.
- Proxy route streams `text/event-stream` without buffering (chunked response test).

**Frontend (vitest + Testing Library)**
- `lile-capsule-store` `connect()` polls status + trajectory; merges new events by offset; does not duplicate.
- `useLileTrajectoryPoll` tolerates old response shape (no `next_offset`).
- `CapsuleStatusStrip` renders offline when store is `running: false`.
- `ChatSftCard` submits `/v1/train` with correct `{objective, samples}` given a multi-turn state (mirrors demo.html logic).

**E2E smoke (manual, dev server)**
1. `bun run dev` + studio backend. Navigate to `/lile`. Expect offline state.
2. Click Load Capsule with default model. Watch daemon log; status strip flips online within 120s.
3. Send a train sample via Chat SFT card. Trajectory tab shows new `train_step` event with `loss`, `grad_norm_total`, `components.*`.
4. Charts render at least `loss` series; once peer PR lands, also `grad_norm` and `kl` series.
5. Navigate to `/chat`, toggle lile mode, send a message. Verify `lile.response_id` and `commit_cursor` in meta. Click 👎 — verify trajectory tab shows new `feedback` event of kind `binary`.

## 9. Scope boundaries & deferred items

- Multi-capsule UX (switcher, list). Backend routes already accept an optional `capsule_id` query param (single "default" capsule for now).
- Historical "past runs" for lile (the existing `HistoricalTrainingView` pattern). Lile is continuous, so the concept is fuzzier — deferred until we've lived with the live view.
- In-browser trajectory download / export.
- Auth. Studio auth is disabled by default on this fork; lile route inherits that behavior.

## 10. Open questions

None blocking v1. Two things worth revisiting after ship:

- Should `/capsule/start` accept LoRA target modules / alpha / dropout in the request, or lock to `ServeConfig` defaults? (Today: locked to defaults; if the capsule-load form grows, we'll extend.)
- Streaming trajectory (SSE from lile) vs. incremental poll. Polling at 2 s with `since_offset` is cheap; SSE is nicer but requires a new route in lile. Deferred unless polling becomes a visible latency issue.

## 11. Ownership

- **Frontend + studio backend proxy**: agent:claude-opus@%21 (this agent; branch `feat/lile-livelearn` via worktree `lile-opus4.7`).
- **Lile daemon changes** (§ 6): agent:claude-opus@%9 (follow-up PR against `ht` after PR#8 merges).
