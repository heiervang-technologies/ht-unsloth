# `budget_exhausted` SSE event — session-level refuse signal

**Author:** claude-opus (live-learn-architect)
**Date:** 2026-04-18
**Status:** proposed
**Priority:** P2 — composes with Tier 5 (pending Cleo §7 K_session calibration)
**Task:** #22
**Depends on:** commits-sse-stream (#18, MERGED), unlike-trajectory-bound.md (Cleo, pending §7)
**Blocks:** nothing; landing early unblocks the Tier 5 wire-through the moment K_session lands

---

## Problem

Cleo's trajectory bound (A rev2 + trajectory-sketch) defines a per-session cumulative-drift functional `Φ_anc` and a refuse threshold `K_session`. Mei's Tier 5 (forthcoming in `unlike-tiered-preconditions.md`) gates unlike dispatch on `Φ_anc ≥ K_session`, refusing further steps until snapshot-restore.

A silent refuse is the wrong UX:

- A user returning from an idle period wouldn't know why their next correction was rejected.
- An RLAIF loop mid-training wouldn't know whether to retry, widen K, or load an earlier snapshot.
- The daemon-internal replay scheduler (IdleReplayScheduler) can tick the trajectory over `K_session` between user touches — the user needs the signal live, not on next-request.

Cleanest shape: a `budget_exhausted` event on the existing `/v1/commits/stream` SSE channel. Subscribers already get commit events; they should get this one on the same pipe with zero new infrastructure.

## Contract

### Event shape

Piggybacks on `lile/commit_stream.py::CommitBroadcaster`. Adds one new broadcast method + one new consumer branch.

```
event: budget_exhausted
data: {
  "cursor": <int>,                // commit cursor at the moment the budget tipped
  "ts": "2026-04-18T12:34:56.789Z",
  "phi_anc": <float>,             // observed cumulative drift functional
  "k_session": <float>,           // active threshold
  "k_warn": <float>,              // K_session / 2 per Cleo §5
  "last_safe_cursor": <int|null>, // cursor of the last snapshot with phi_anc < k_warn
  "reason": "tier_5_refuse"
}
```

`last_safe_cursor` may be `null` when no such snapshot exists in the current session (fresh start with immediate over-budget step — edge case but possible under adversarial inputs).

### Emission semantics

Fires **once per crossing**. If a subsequent step would also exceed `K_session` while the budget is still exhausted, no repeat event — emit only on the cursor that crosses the threshold, not on every subsequent refused step.

Re-arm on snapshot-restore: when `/v1/state/snapshot/load` runs successfully and the new state's `phi_anc < K_warn`, the broadcaster clears the exhausted flag. The next crossing fires a fresh event.

### Refuse semantics

Orthogonal to the event itself, but pinning here so the spec is self-contained:

- When exhausted, `TrainEngine.step` on any `unlike` sample returns `ValueError("session budget exhausted; load snapshot <=last_safe_cursor or widen K_session (see Tier 5)")`.
- Other objectives (`sft`, `coh`, `ntp`, `kl_anchor` solo) are **not** gated by Tier 5 — the trajectory bound applies to unlike only in the current Cleo framing. Extending to other SFT-family objectives is deferred pending theorem coverage.
- `IdleReplayScheduler` checks the flag before each enqueue; if exhausted, it sleeps without firing.

### Backward compatibility

Consumers that don't know about the event type should ignore it per SSE spec (unknown `event:` field). No payload-shape change to existing `event: commit` frames.

## Implementation

### `commit_stream.py` additions

```python
class CommitBroadcaster:
    def __init__(self, *, enabled: bool = True) -> None:
        # existing state ...
        self._exhausted: bool = False

    def broadcast_budget_exhausted(
        self, *, cursor: int, phi_anc: float, k_session: float,
        k_warn: float, last_safe_cursor: int | None,
    ) -> None:
        if self._exhausted:
            return  # idempotent; emit only on crossing
        self._exhausted = True
        event = {
            "cursor": cursor,
            "ts": _iso_now_ms(),
            "phi_anc": phi_anc,
            "k_session": k_session,
            "k_warn": k_warn,
            "last_safe_cursor": last_safe_cursor,
            "reason": "tier_5_refuse",
        }
        payload = {"_budget_exhausted": event}
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                self.drops += 1

    def clear_budget_exhausted(self) -> None:
        self._exhausted = False
```

### `server.py` consumer branch

In the `/v1/commits/stream` generator, after the existing `_shutdown` check:

```python
if event.get("_budget_exhausted"):
    yield (
        "event: budget_exhausted\n"
        f"data: {json.dumps(event['_budget_exhausted'])}\n\n"
    )
    continue
```

### `controller.py` wire-through

- `Controller` holds the `Φ_anc` accumulator (feeds from unlike-trajectory-bound per Cleo §3).
- On every unlike commit, after `broadcast_commit`, check threshold and fire `broadcast_budget_exhausted` if crossed.
- On successful `snapshot_load` with post-load `Φ_anc < K_warn`, call `clear_budget_exhausted`.

Tracking `Φ_anc` itself is Tier 5's scope, not this spec. This spec is **only** the plumbing.

## Tests

1. **Single crossing fires one event.** Push `Φ_anc` over `K_session` across two consecutive steps; subscriber sees exactly one `event: budget_exhausted` on the first crossing, nothing on the second.
2. **Re-arm after clear.** Exhausted → `clear_budget_exhausted` → push over again → second event fires.
3. **Event payload shape pinned.** All required fields (`cursor`, `ts`, `phi_anc`, `k_session`, `k_warn`, `last_safe_cursor`, `reason`) present; `last_safe_cursor` nullable.
4. **Drop-on-full preserved.** Slow consumer with a full queue increments `drops`; fast consumer still sees the event. Same semantics as `commit` events.
5. **Shutdown beats budget_exhausted.** If `broadcast_shutdown` fires while exhausted flag is set, the shutdown event still emits (consumer branch order: shutdown → budget_exhausted → commit).
6. **Subscriber-count observability.** `/health` already exposes `commit_sse_subscribers` + `commit_sse_drops`; no new field needed for this event — drops aggregate across event types.

All tests cpu_only; exercise the broadcaster primitive directly + a FastAPI inline harness matching the pattern in `test_commits_sse_stream.py`.

## Rollout

Single PR on top of Tier 5. Keep the commits split:
1. `commit_stream.py` + tests (plumbing only, no Tier 5 semantics).
2. `controller.py` wire-through + integration test (requires Tier 5's `Φ_anc` accumulator).

Allows (1) to ship the moment K_session lands and the accumulator follows in (2).

## Non-goals

- **No `budget_warn` event** for the `K_warn` threshold crossing. Tier 5 can add later if the UX needs it; this spec is refuse-only.
- **No per-user/session scoping.** The daemon is single-tenant from a trajectory perspective; multi-user budgets are out of scope for lile v2.
- **No automatic snapshot-restore on exhausted.** The event surfaces; the orchestrator decides (could be the RLAIF loop, the user, or a retry script). Auto-restore couples two concerns and is easier to layer later than to unwind.

## Downstream uses

- **RLAIF tutor-as-critic loop.** On `budget_exhausted`, the critic can pick: reload a `last_safe_cursor` snapshot and replay corrections at a safer η, or pause the loop.
- **Household AI UX.** A returning user sees "the model refused a correction because it's drifted too far from last safe state; load snapshot N?" — concrete, actionable.
- **Calibration sweeps (#17).** The sweep harness can count `budget_exhausted` events per configuration; a sweep row that exhausts the budget is by definition in the unsafe regime.
