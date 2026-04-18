# safety_monitor — observational Razin-safety sidecar primitive

**Author:** claude-opus (live-learn-architect), Cleo (theory), Mei (spec motivation)
**Date:** 2026-04-18
**Status:** proposed
**Priority:** P1 — cheap, high-value, composes with SSE + any SFT-family objective
**Task:** #20
**Depends on:** nothing
**Composes-with:** /v1/commits/stream (#18), kl_anchor target-position scope (#15)

---

## Problem

Cleo's razin-safety-sharpened.md characterization theorem identifies exactly which non-target tokens can grow under a one-step SFT-family update:

```
q_j > p_j   ⟺   p_j < M_p(η)    where    M_p(η) := -(1/η) log E_p[exp(η β_·)]
```

The unsafe regime is at **small η** (counterintuitive — inverts the "smaller LR = safer" heuristic). In the household-AI use case, a user correction "never say Voldemort" can silently raise `p("Sauron")` at small η, with no aggregate-safety signal to catch it.

This is a **per-step, per-sample** detectable condition. Cheap (one log-sum-exp + an O(|W|) set-membership check). Valuable (catches the exact failure mode Razin-safety's aggregate reading misses). Worth its own primitive — a Razin-safety observational sidecar.

## Contract

### Shape

`safety_monitor` is a **batch-objective** (same composition pattern as `kl_anchor`), with `weight=0.0` semantics — it is **observational, not loss-contributing**. It computes per-step safety signals and surfaces them in the `components` dict + (optionally) emits alarms on a watchlist.

```json
{
  "objective": "sft",
  "samples": [{"prompt": "Q:", "response": "A."}, ...],
  "batch_objectives": [
    {"name": "safety_monitor",
     "watchlist": [12345, 67890, ...],       // optional, per-batch
     "alarm_threshold": 1.0,                   // optional, default 1.0
     "weight": 0.0}                            // always 0.0; enforced
  ]
}
```

**Three-tier union** (Mei, 2026-04-18):

```
- Daemon-global: cfg.default_watchlist         (absolute-never: PII, safety patterns)
- Batch-level:   batch_objectives[].watchlist  (session/user-scoped; wire-efficient for bulk)
- Per-sample:    sample["watchlist"]           (prefix-scoped correction)
```

Effective watchlist at step time = union of all three, same semantics as kl_anchor's `exclude_token_ids`. Batch-level prevents the 64-sample-same-watchlist duplication; per-sample keeps context-scoped corrections first-class; daemon-global is the always-present safety floor.

### Target-position source (Mei, 2026-04-18)

The safety signal is computed at the **supervised target positions**, not everywhere. Positions come from the main objective via an extension of the result-dict protocol:

**Main objective result dict extension:**

```python
def sft_loss(...) -> dict:
    return {
        "loss": ...,
        "components": {...},
        "target_positions": [[p_0_0, p_0_1, ...], [p_1_0, ...], ...],  # optional, list[list[int]] per sample
    }
```

`safety_monitor` reads `target_positions` from the preceding main-objective result. If the main objective does not expose it, safety_monitor raises:

```
RuntimeError("safety_monitor requires target_positions from <objective>; "
             "the objective does not yet expose them. See safety-monitor-primitive.md.")
```

This forces each SFT-family objective to opt in explicitly — the right pressure to keep position semantics owned by the main objective and consumed (not re-derived) by the sidecar. Primitive orthogonality: main objective owns position geometry, sidecar owns the Razin-safety computation.

Geometry per objective:
- **SFT / weighted_sft / coh:** all response token positions.
- **ntp:** all positions.
- **unlike:** one position (end of prefix).

### Computation per sample × position

1. Run the main-objective forward pass (already happening — we piggyback on it).
2. For each position `p` in `target_positions[i]`:
   - Let `logits_i_p` = logits at sample i, position p.
   - Let `π := softmax(logits_i_p)`.
   - Let `t := target_token_id` at that position (from the main-objective's label tensor).
   - Compute `M_π(η) := -(1/η) · log Σ_k π_k · exp(η · (𝟙[k=t] - π_k))`.
   - Compute grower set `S_i_p := {j ≠ t : π_j < M_π(η)}`.
   - Compute watchlist hit: `H_i_p := S_i_p ∩ (per_sample_watchlist ∪ batch_watchlist ∪ default_watchlist)`.
3. Aggregate across positions × samples.

### η source

The engine's current LR, read from the optimizer state at step time. No caller-supplied η.

### Output (components dict)

```
safety_monitor_eta:             float  # the η used for M_p — required for M_p_mean interpretability across schedulers / per-objective LR
safety_monitor_alarm_count:     int    # number of (sample, position) pairs where H_i_p is non-empty
safety_monitor_grower_size_mean: float # E_{i,p}[|S_i_p|] — mean grower-set cardinality, renamed from grower_mean to disambiguate from growth magnitude
safety_monitor_grower_size_max: int    # max_{i,p} |S_i_p|
safety_monitor_M_p_mean:        float  # E_{i,p}[M_π(η)]
safety_monitor_M_p_min:         float  # min_{i,p} M_π(η)
safety_monitor_watchlist_hits:  list   # flat list of (sample_idx, position, token_id) tuples
```

### Known approximation (optimizer scope)

`M_p(η)` is derived from a plain SGD step (`z' = z + η · (𝟙[·=t] - p)`). In the production path we use **AdamW**, whose first and second moments `(m, v)` warp the effective logit update so that `Δz_k ≠ η · (𝟙[k=t] - p_k)` in general. First-order the update is close; asymptotically the directions align but magnitudes differ.

**Consequence for alarm semantics:**
- Alarm fires → definitely unsafe (the SGD-theoretic bound is violated, and AdamW's first-order agreement with SGD ensures displacement is at least proportional).
- Alarm silent → within the SGD-theoretic safe zone, but AdamW could still displace due to moment-driven magnitude warping. **Silent is not a safety guarantee under AdamW.**

Treat `M_p(η)` as a **lower-bound heuristic on displacement** under AdamW, not an exact bound. The calibration sweep (#17) validates this empirically across optimizers. A sharper AdamW-specific bound (if needed) is a follow-up analysis, not scope for this primitive.

### Alarm behavior

When `alarm_count >= alarm_threshold`, the engine:
- Logs a `safety_monitor.alarm` record (stderr, log file, `/v1/state/stats`).
- Emits a `safety_alarm` SSE event on `/v1/commits/stream` (when #18 lands — threaded along the commit event for that step).
- Does **not** block the step. The primitive is observational.

A future track can layer a `safety_gate` primitive that *does* block on watchlist hits; that's a different primitive and not scope for this PR.

### η source

The engine's current LR. `safety_monitor` reads it from the optimizer state at step time. No caller-supplied η.

## Implementation

New file: `lile/objectives/safety.py` with `safety_monitor_loss()` following the same signature shape as `kl_anchor_loss`. Returns `(loss=0.0 tensor, components_dict)`. The zero-loss tensor is required to keep the composition grammar consistent with other batch objectives.

Dispatch: register in `lile/objectives/__init__.py` under `"safety_monitor"`.

Engine hook: `TrainEngine.step` already iterates `batch_objectives`; safety_monitor lands on the same path with no new engine code.

### Cost analysis

Per sample at the target position:
- `M_p(η)`: one log-sum-exp on the existing `logits` tensor — O(V) but already in the compute graph for softmax.
- Grower set: `(p < M_p).nonzero()` — O(V) memory-bound.
- Watchlist hit: set intersection — O(|W|).

For a batch of B samples and vocabulary V=32000, that's ~32000*B FP32 ops + B * |W| python ops. Negligible next to the forward+backward.

## Tests

1. **M_p matches Cleo's numeric witness.** `V=3, p=(0.10, 0.89, 0.01), t=0, η=1` → `M_p ≈ 0.4759`, grower set `{2}`. Pinned verbatim.
2. **Grower set is exactly the j with p_j < M_p.** Brute force over random simplex points (20 trials), compare against the theorem predicate.
3. **Watchlist intersection fires alarm.** Construct a sample where token 9 is in the grower set, watchlist=`[9]` → alarm_count=1, watchlist_hits contains `(0, 0, 9)` (sample_idx, position, token_id).
4. **Watchlist-miss fires no alarm.** Grower set = `{9}`, watchlist=`[100]` → alarm_count=0.
5. **Three-tier watchlist union.** Daemon-global + batch-level + per-sample all contribute; final alarm reflects the union. Load-bearing given the three-tier semantics.
6. **Zero loss contribution.** `loss` returned is a zero-valued scalar tensor that survives backward (no NaN, no gradient flow from this primitive).
7. **Composition with SFT + kl_anchor.** Three-way composition: SFT + kl_anchor + safety_monitor. Components dict has all three sets of keys, final loss equals SFT + weight*kl_anchor (safety_monitor contributes nothing).
8. **Multi-position SFT determinism (Mei, 2026-04-18).** 5-token response on SFT, monitor computes M_p + grower set at each of the 5 target positions, emits 5 per-position records. Run twice under `do_sample=False` + deterministic algos; components dict must be byte-identical across runs. Noise-floor protocol hook for the safety signal itself.
9. **Missing target_positions → RuntimeError.** Compose safety_monitor with a main objective that does not expose `target_positions` in its result dict; assert the RuntimeError fires with the expected error string.

## Non-goals (intentional)

- **No automatic watchlist blocking.** Observation only. A separate `safety_gate` primitive can do blocking if ever needed.
- **No multi-step trajectory memory.** The theorem is one-step; the primitive matches.
- **No automatic watchlist discovery.** The watchlist is a user contract ("never generate these tokens"). Inferring it from model history is out of scope.
- **No correction suggestion.** If the alarm fires, the primitive surfaces the hit; the orchestrating script decides whether to retry with larger η, abort, widen the KL anchor, etc.

## Rollout

Single PR: new `lile/objectives/safety.py` + registry entry + `test_safety_monitor.py` + one smoke_objectives step. No route changes. No schema migration.

Once /v1/commits/stream (#18) lands, a small follow-up wires the alarm into the SSE event payload as `safety_alarm`: takes ~10 LOC in the Controller commit hook.

## Downstream uses

- **Household AI defaults.** `cfg.default_watchlist` holds absolute-never tokens (PII patterns, safety-critical strings). Always on, always cheap.
- **Calibration sweep validation (#17).** The sweep compares empirical small-eta failure modes against Cleo's A bound; safety_monitor is the instrument.
- **RLAIF tutor-as-critic loop.** Tutor can add bad-token IDs to the per-sample watchlist before submission; loop monitors whether the correction is actually landing without displacement.
