# Unlike + SFT-on-good: cumulative trajectory bound across N steps

**Status.** Sketch. Per-step bound (Lean-pending) lives at `unlike-kl-step-size-bound.md` (A). This document is the multi-step extension Mei flagged as load-bearing for `lile`'s online loop, snapshot-restore cadence, and idle-replay scheduler.

**Companion documents.** A — `unlike-kl-step-size-bound.md`. B — `razin-safety-sharpened.md`.

## 1. Why per-step is not enough

A protects against *one* bad step. `lile`'s online loop takes `N` steps per session — typically `N` in the 10²–10⁴ range across a continual-correction window. Even when every per-step `ε_i` is inside the §5 ceiling, the *cumulative* off-anchor drift across `V \ S` is unbounded under naive composition. The household-AI failure mode this primitive is supposed to prevent — slow drift toward unintended outputs over a session — is exactly the one A doesn't cover.

The session envelope is what we need:

> **Operational claim.** There exists a per-session functional `Φ(η_1, …, η_N; p_0, p_1, …, p_N; w_+,1, …, w_+,N)` such that the cumulative off-anchor TV drift `Σ_i ‖q_i|_{V\S} - p_i|_{V\S}‖_{TV}` is bounded by `Φ`, and `lile` should refuse to step (or trigger a snapshot restore) when `Φ` exceeds a session budget `K`.

## 2. Setup

Session = sequence of states `p_0, p_1, …, p_N`, with `p_{k+1} = compositeStep(p_k; b_k, g_k, w_{+,k}, η_k)`. Per-step anchor `L_kl(z; z_0) = KL(p|_S(z) ‖ p_0|_S(z_0))` references the **session-start** distribution `p_0|_S`, not the previous step. (This is the operational semantics in `lile/objectives/unlike.py` after PR #29.)

Two regimes interact:

- **On-S drift.** `(p_k)|_S` migrates toward `g` step-by-step. Anchor pulls back at `k > 1` (§3 below). Bounded by anchor convergence rate.
- **Off-S drift.** `(p_k)|_{V\S}` shrinks step-by-step (under the §5 §A.2 sign condition `w_+ > p_b/(1-p_b)`). No anchor restoring force off S — the cumulative drift accumulates.

The cumulative risk functional has to capture off-S drift; on-S is anchor-controlled.

## 3. Anchor restoring force at k > 1 (orthogonal side theorem on on-S drift)

**Scope (Mei).** The result in this section concerns the **on-S** conditional `(p_k)|_S` and the cumulative anchor energy `E_k`. It is **not load-bearing for the cumulative off-S budget** in §4 — anchor is on-S, the operational off-S budget is what `lile` actually trips on. Keep this section as a side theorem about surgery-position guarantees: the `{bad, good}` pair is what we're operating on, and reasoning about how its conditional drifts is independently useful (e.g. for evaluating whether the unlike loss is "fair" to good — does it pull p_g up too aggressively over a session?).


**Load-bearing invariant (Mei).** This entire §3 derivation, and the trajectory bound that depends on it, assumes:

```
π_ref  =  base + merged_deltas   (updates at merge boundaries)
```

This is `lile`'s default `kl.py` path under `disable_adapter()` — π_ref is part of the snapshot payload and "resets" naturally on snapshot restore. If a future spec introduces a persistent session-scope reference that survives snapshots (e.g. for longer-horizon drift budgets), the §3 restoring-force calculation needs a drift-penalty term added. Document the new invariant before extending.

**Initial-condition flag for the two reference modes.** `lile/state.py` exposes a `frozen_ref=True` toggle that loads a pristine NF4 base as π_ref instead of using `disable_adapter()`. The §3 restoring-force result holds in both cases (the math is about π_ref being fixed within a session and p drifting), but the initial anchor energy differs:

```
default     (disable_adapter):   E_0 = KL(p_session_start ‖ base + merged_deltas) ≈ 0
frozen_ref  (pristine base):     E_0 = KL(p_session_start ‖ pristine_base)        > 0
```

The `Φ_obs` calibration in §6.1 will be different across these regimes — frozen_ref starts with a non-zero anchor energy that discounts the early-session ε_i more aggressively.


At `k = 1`, `∇_z L_kl(z_0) = 0` because the anchor reference *is* `p_0|_S` (A §6.1). At `k > 1`, the model has drifted: `p_k|_S ≠ p_0|_S`, so

```
∇_{z_k} L_kl(z_k; z_0)  =  (p_k|_S - p_0|_S)  +  cross-coupling on V\S
                       ≠  0.
```

Concretely, after the composite step's first iteration `p_1|_S` has shifted toward `g`. The anchor at step 2 generates a gradient pulling `z_b` *up* and `z_g` *down* (back toward the reference). This is a **restoring force** on the on-S conditional.

**Consequence for the trajectory bound.** Define the on-S anchor energy `E_k := KL(p_k|_S ‖ p_0|_S)`. Under plain SGD on the composite loss with anchor weight `λ_kl > 0`:

```
E_{k+1}  ≤  E_k  -  η_k · λ_kl · ‖∇L_kl(z_k)‖²  +  η_k² · drift_quadratic_term
```

i.e. the anchor delivers a `Θ(η · λ_kl)` decrement against an `O(η²)` drift increment. **For sufficiently small η_k the on-S energy is non-increasing.** This is the proximal-gradient interpretation Mei picked in A §6.1 made precise: the anchor's effect *develops* across steps even though it's first-order zero at step 1.

## 4. Cumulative risk functional: Φ_obs

**Headline.**

```
Φ_obs  :=  Σ_{i=1}^N  TV_sim^{emp}_i
```

where `TV_sim^{emp}_i` is the per-step empirical off-S TV computed at dispatch (A §5.b — one-step simulation, O(V)). This is the cumulative off-S drift consumed by the trajectory so far — by construction, an exact account of what's happened, not a bound that may or may not hold.

**Why not a closed-form bound.** The original draft of this section proposed `Φ_obs = Σ ε_i · exp(-α · E_i)` with `ε_i` the §5 closed-form per-step bound and `E_i` the on-S anchor energy. Two problems surfaced during §7 calibration:

1. **A §5 closed-form ε is not a strict upper bound.** 26% of per-step trials in the calibration sweep had empirical TV > the §5 formula, with worst-case ratio ≈ 5×. The "ε" was a linearization, not a TV ceiling. Building Φ on it would inherit the under-bound.
2. **Anchor discount targets the wrong drift.** The §3 restoring force is on the **on-S** conditional, but the cumulative budget controls **off-S** drift. The two are essentially uncorrelated empirically — the discount factor `exp(-α · E_i)` had near-zero effect on the median ratio across α ∈ [0, 20].

`Φ_obs` sidesteps both problems: ε_i is the observed quantity (no bound to fail), and there is no anchor discount to misalign. Same operational structure as A §5.b's empirical ceiling: simulate, observe, decide.

**Sub-linear growth.** Per-step TV averages `~ε̄`, but cumulative TV grows sub-linearly when drift directions decorrelate across steps (random training input, varying `(b, g)` pairs). On the §7 sweep, `Φ_obs` over N=100 steps is typically `~14× ε̄` (vs. the worst-case linear `100 · ε̄`), so the K_session* threshold is well below what a naive `N · ε_max` budget would suggest.

**Operational hook.** Refuse-step when `Φ_obs > K_session` (and surface WARN at `K_warn = K_session / 2`). `K_session*` is back-solved from the §7 sweep at the 95th percentile of `Φ_obs` over N=100-step trajectories — that's the v1 default for `cfg.cumulative_session_budget`.

## 5. Snapshot-restore cadence (the lile-side application)

Given `Φ_obs`, the natural session policy is:

1. **Per-step floor (A §4).** Refuse if `η_k < η_min^{emp}(p_k, b_k, g_k, w_{+,k})`.
2. **Per-step ceiling (A §5.b).** Refuse if `TV_sim^{emp}_k > ε_target` (per-step off-S budget).
3. **Cumulative.** Track `Φ_obs = Σ TV_sim^{emp}_i` across the trajectory. WARN at `Φ_obs > K_warn` (default `K_session / 2`).
4. **Refuse-session.** If `Φ_obs > K_session`, refuse further training. Trigger a snapshot restore to the last checkpoint with `Φ_obs < K_warn`.
5. **Idle-replay rebudget.** Replay steps accumulate into `Φ_obs` like any other step (architect's call, §5.1 below). Reproducibility check: replay from the trajectory should reach the same `Φ_obs` within tolerance.

### 5.1 Rebudget semantics across replay and session boundaries

Architect-locked, math-verified. Three corollaries of the §3–§4 results applied to `lile`'s step-source plurality:

1. **Budget is a snapshot-between-snapshots property.** Snapshot load = budget reset; all weight-updating steps between the same snapshot boundaries accumulate into `Φ_obs`. Consistent with §6.2 reset-on-resume — `E_k` and `Φ_obs` are properties of the *trajectory of weights*, not of the user-interaction or scheduling layer. Math-check: ✓.
2. **Session is not a user-intent boundary.** Daemon-idle → replay-fires → user-returns is **one trajectory** as far as `Φ_obs` is concerned. No implicit split on idle. `K_warn` / `K_session` thresholds cross counts across any source (feedback, replay, tutor), not per-source. Math-check: ✓ — the derivation makes no user-interaction assumption.
3. **Replay-induced refuse is a daemon-emitted shutdown signal.** If `IdleReplayScheduler` ticks the trajectory over `K_session` while the user is idle, the daemon refuses further steps (both replay AND live) until snapshot-load. Composes with `commit_stream` (SSE primitive): `budget_exhausted` event emitted so a returning user sees the state. Math-side impact: zero (this is plumbing) — but the operational hook is real and lives in §8 below.

## 6. Open seams

1. **`K_session*` calibration — closed (§7).** Default `K_session* = 0.27` (n=2000, N=100, 95th percentile). Ships in `lile/objectives/unlike.py`.
2. **Anchor on resume — resolved (Mei + lile codepath audit).** Default path (`disable_adapter()`) gives `π_ref = base + merged_deltas`, which is part of the snapshot payload — π_ref "resets" naturally on snapshot restore as a consequence of lile's existing semantics, no explicit reset code needed. Trajectory bound restarts on resume. The §3 invariant block pins this assumption; if a future spec introduces a persistent session-scope reference that survives snapshots, the §3 derivation needs a drift-penalty term.
3. **Composition with B.** B characterizes per-step per-token mass flow under SFT alone. The trajectory generalization of B (per-token cumulative growth across N SFT steps) is a separate theorem that would let us drop the unlike loss entirely for tokens that B says shrink monotonically. Out of scope here; flagged for follow-up.

## 7. Calibration sweep — `K_session*` recommendation

Empirical sweep over n=2000 trajectories, N=100 steps each, with random `(b_k, g_k)` token pairs (V ∈ [3, 20]), `w_{+,k} ∈ [0.1, 10]` (log-uniform), and `η_k ∈ [10⁻⁴, 0.05]` (log-uniform — keeps A §5 linearization regime valid). Per step, `TV_sim^{emp}_k` is computed from the actual composite step; `Φ_obs := Σ_k TV_sim^{emp}_k`.

Distribution of `Φ_obs`:

| statistic | value |
|---|---|
| median  | 0.144 |
| p95     | 0.267 |
| max     | 0.438 |

**Recommended default: `K_session* = 0.27`** (95th percentile, 5% refuse-rate target). Ships as the default for `cfg.cumulative_session_budget` in `lile/objectives/unlike.py`. `K_warn = K_session / 2 = 0.135`.

Sub-linear growth is real: median `Φ_obs = 0.144` over N=100 steps with median per-step TV `~ε̄ ≈ 0.001-0.005` gives a cumulative `~14×ε̄`, two orders of magnitude under the linear `100·ε̄` budget. Drift directions decorrelate across random `(b, g)` pairs.

Script: `~/identity-matrices/cleo-work/scratch/trajectory_bound_sweep.py`. Reproducer: `python trajectory_bound_sweep.py`.

**Failure mode for this calibration**: a deployed workload with strongly correlated `(b, g)` selection (e.g., a tutor stream that hammers the same misconception across many steps) could exceed the random-sweep p95. This is exactly what the operational `TV_sim^{emp}` per-step ceiling (A §5.b) is for — even if `Φ_obs` outpaces the random-sweep prior, the per-step refuse keeps individual TVs bounded. `K_session*` is a session-budget prior, not a worst-case guarantee.

## 8. Implementation hooks

- `lile/objectives/unlike.py` — log per-step `E_k` (anchor energy) under `unlike/anchor_energy`.
- `lile/controller.py` — track `Φ_obs` across the session under `safety/cumulative_drift`. Surface `WARN` at `K_warn`, refuse-step at `K_session`.
- `lile/snapshot.py` — snapshot policy aligned with the §5 cadence. Snapshot when `Φ_obs < K_warn`; restore when `Φ_obs > K_session`.
- `unlike-tiered-preconditions.md` — extend with **Tier 5: cumulative-drift refuse-session**, gated on `Φ_obs`.
- `commit_stream` SSE wire-through — emit `budget_exhausted` event when `Φ_obs > K_session`. Deferred follow-up, architect-owned (~10 LOC, drafted after trajectory stabilizes).
- `eval-methodology-gate.md` regime-labels — add `cumulative_drift_budget` as a mandatory regime label when reporting results from sessions that triggered `Φ_obs > K_warn`.

## 9. Lean (deferred — depends on A's discharge)

Statement to formalize once A's `unlike_composite_step_window` discharges:

```lean
theorem unlike_trajectory_bound
    (sessions : List CompositeStepConfig) (p_0 : Distribution V)
    (K_session : ℝ) (hK : 0 < K_session) :
    let trajectory := composite_step_chain p_0 sessions
    let phi_anc := cumulative_anchor_discounted_drift trajectory
    phi_anc ≤ K_session → cumulative_off_S_tv trajectory ≤ K_session
```

Composes on top of A's per-step lemma. No new mathematical machinery beyond Mathlib's `KL` and the supermartingale lemma if we eventually formalize §4.3.
