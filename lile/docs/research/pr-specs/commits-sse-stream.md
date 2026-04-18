# /v1/commits/stream — SSE Primitive Spec

**Author:** claude-opus (live-learn-architect)
**Date:** 2026-04-18
**Status:** proposed, awaiting implementer (codex / lile-backend)
**Priority:** P0 — highest-priority backend work per Mei's push

---

## Why this primitive

Three concrete use cases converge on the same missing signal: a **per-commit event stream**.

1. **Byte-determinism live detection.** Under `do_sample=False` + deterministic algos, every A/B cursor pair should produce byte-identical inference outputs until a commit lands. Today we can only catch a flip count in offline compare-json (hours per eval). An SSE stream gives live drift detection the moment a flip occurs.

2. **RLAIF real-time feedback loop.** `lile/teach/rlaif/run.py` (in-flight, architect track) needs low-latency commit confirmation to issue the next tutor-as-critic batch. Polling `/v1/state/stats` is wrong-grained (coarse + expensive); SSE is the right primitive.

3. **Multi-client observability.** Any number of clients can attach (debugger, dashboard, another script) without adding load on the hot path. The server emits once, fans out. Enables us to leave a "watch loss" tail running in a terminal during manual testing.

---

## Contract

### Route

```
GET /v1/commits/stream  →  Content-Type: text/event-stream
```

No parameters. No replay buffer. Clients connect, receive all subsequent commit events in order, and handle their own backfill via `/v1/state/stats` or `/v1/state/snapshot` if they need history.

### Event shape

One event per commit cursor advance. Ordering matches the cursor monotonic:

```
event: commit
data: {"cursor": 42,
       "ts": "2026-04-18T15:32:01.123Z",
       "objective": "unlike",
       "loss": 0.2341,
       "components": {"unlike_ul": 0.18, "unlike_sft": 0.05, "unlike_triggered": 1, ...},
       "batch_size": 1}
```

Field notes:
- `cursor`: integer, the commit cursor value *after* the step landed. Strictly increasing.
- `ts`: ISO 8601 UTC, emitted at event-enqueue time (not at socket-write).
- `objective`: the primary objective name (from the train request).
- `loss`: scalar float, the committed step's loss.
- `components`: pass-through of the objective's components dict. Shape depends on objective.
- `batch_size`: number of samples in this commit. Informational; useful for loss normalization on the client side.

### Keepalive

Server emits a `: keepalive\n\n` comment line every 15s when no commit events fire. Keeps intermediate proxies from closing the connection.

### Error / shutdown

Server emits one terminal event on daemon shutdown and closes the stream:

```
event: shutdown
data: {"reason": "daemon_stop"}
```

Clients should reconnect with exponential backoff; no retry hint sent from server.

---

## Implementation sketch

### Producer side (Controller)

The single writer that advances the commit cursor already exists in `Controller`. Add a broadcast step *after* the cursor is advanced and *before* the train task's completion event is set:

```python
# somewhere in Controller.commit()
event = {
    "cursor": self._cursor,
    "ts": _iso_now(),
    "objective": spec.objective,
    "loss": float(result["loss"]),
    "components": result.get("components", {}),
    "batch_size": len(spec.samples),
}
for q in list(self._commit_subscribers):
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        # Drop-on-full: slow clients don't back-pressure training.
        self._commit_drop_counter += 1
```

`self._commit_subscribers: set[asyncio.Queue[dict]]`. Each subscriber queue is bounded (default 256) — when full, events drop silently to preserve the training hot path.

### Consumer side (FastAPI route)

```python
# routes/stream.py (new file)
from fastapi.responses import StreamingResponse

@app.get("/v1/commits/stream")
async def commits_stream(request: Request):
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=256)
    controller._commit_subscribers.add(queue)
    try:
        async def gen():
            keepalive_interval = 15.0
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event.get("_shutdown"):
                    yield f"event: shutdown\ndata: {json.dumps({'reason': 'daemon_stop'})}\n\n"
                    return
                yield f"event: commit\ndata: {json.dumps(event)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    finally:
        controller._commit_subscribers.discard(queue)
```

### Shutdown hook

Controller stop path enqueues `{"_shutdown": True}` on every subscriber queue before closing them.

---

## Non-goals (intentional)

- **No chat-event SSE.** Chat responses are already one-shot; streaming is a different primitive.
- **No trajectory replay.** The SSE endpoint is live-only. Historical commits live in the trajectory log.
- **No filter expressions.** Clients filter on the consumer side. Server-side filters drift toward per-workflow state, which violates primitives-as-contract.
- **No websocket variant.** SSE is simpler, survives HTTP/1.1 + Nginx + curl. Websocket would be appropriate only if we ever add server-push to chat (out of scope here).

---

## Test obligations

1. **Ordering invariant.** Subscribe two clients, enqueue a rapid burst of K commits, assert both clients receive cursor values `1..K` in strict order.
2. **Drop-on-full.** Subscribe one slow client (sleeps in its consumer), one fast client. Enqueue K > 256 commits. Fast client sees all K. Slow client sees some subset; dropped count surfaces in `/v1/state/stats` as `commit_sse_drops`. Training throughput unaffected (measure: wall time of K commits with vs without the slow subscriber attached — within 5%).
3. **Keepalive.** Connect a client, wait 30s with no commits, assert two `: keepalive` lines arrive.
4. **Shutdown clean.** Attach client, trigger daemon shutdown, assert client receives `event: shutdown` and socket closes cleanly.
5. **Ordering vs commit cursor semantics.** For every event with `cursor=N` the client sees, a subsequent `/v1/state/stats` call reports `cursor >= N`. Pinned because the current commit-cursor contract is the load-bearing invariant.

---

## Rollout

- Ship as a new route in a `lile/routes/stream.py`. No changes to existing routes, zero back-compat risk.
- Flag: `cfg.commits_sse_enabled` (default `True`). A disabled config short-circuits the subscriber set so the training path pays zero cost.
- Docs: add to `openapi.md` and link from `PLAN.md` §3.4 observability.

## Tracking

- Task #18 in the architect queue.
- Blocks: RLAIF loop real-time feedback (architect track), live byte-determinism detection (eval track).
- Blocked by: nothing. Can be worked in parallel with the kl_anchor scope extension (task #15) and unlike tiered preconditions (task #19).
