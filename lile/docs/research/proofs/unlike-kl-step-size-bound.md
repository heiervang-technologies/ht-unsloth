# Unlike + SFT-on-good: step-size window for one-step Razin-safety

**Status.** Sketch. Numerical falsifier passing in the regimes covered by the analysis; one residual regime ("never safe in the swept η grid") flagged for §6 follow-up. Lean formalization deferred until the analytical bounds settle.

**Companion document.** `razin-safety-sharpened.md` (B) — the SFT mass-flow characterization that motivated this composite-loss analysis.

## 1. Setup

Let `p = softmax(z) ∈ Δ^{V-1}` be the model's pre-step distribution at one position. Let `b ∈ V` be the unlike target (a token we want pushed down) and `g ∈ V \ {b}` be a single SFT-good target. Let `S = {b, g}` be the **anchor scope**: the KL-anchor mask used in `lile/objectives/unlike.py` is supported on `S`, and the renormalized conditional `p|_S` is what the anchor controls.

Composite one-step loss:

```
L(z) = -log(1 - p_b)  +  w_+ · (-log p_g)
        \________ ___________/   \_______ _________/
                 V                       V
            unlikelihood          weighted SFT-on-good
```

Update: `z' = z - η · ∇_z L`. Everything below is one SGD step from `z`.

## 2. Logit shifts in closed form

Let `Δ_k := z'_k - z_k = -η · ∂L/∂z_k`. Direct computation (softmax derivative twice, no approximation):

```
Δ_b      = -η · p_b · (1 + w_+)
Δ_g      =  η · ( p_b · p_g / (1 - p_b)  +  w_+ · (1 - p_g) )
Δ_j      =  η · p_j · ( p_b / (1 - p_b)  -  w_+ )       for j ∉ S
```

Then `q_k = p_k · exp(Δ_k) / Z` with `Z = Σ_k p_k · exp(Δ_k)`.

**Sanity check (pure unlike, w_+ = 0).** Δ_b = -η p_b, Δ_j = η p_j p_b/(1-p_b) > 0 for j ≠ b. Pure unlike is one-step safe for `q_b` at every η > 0 (Δ_b < 0 < log Z whenever any other Δ_k > 0). This recovers the Welleck monotonicity intuition.

## 3. The headline (for code reviewers in a hurry)

> **Composite unlike + w_+ · SFT-on-good is *not* one-step safe at small η.** When `||p||² > p_g + p_b` — i.e. the prior is dominated by a non-anchor token outside `S` — there exists an `η_min > 0` such that `q_b > p_b` for every `η ∈ (0, η_min)`. The operational step-size window is gated empirically:

```
        η  ≥  η_min^{emp}(p, b, g, w_+)        (§4 — bisection)
        TV_sim^{emp}(p, b, g, w_+, η)  ≤  ε    (§5.b — one-step simulation)
```

**Why:** at small η the dominant non-S token donates its mass to *every* token (including `b`) when `b` shares the leftover; the unlike push-down on `b` is O(η · p_b), the SFT-on-good push-up on `g` is O(η · w_+ · (1 - p_g)), and the side-effect on `b` from suppressing the non-S tokens is O(η · w_+ · ||p||²_{V\S}), which can dominate the unlike term when `w_+` is large and `p_b` is small.

**How to apply:** the safe regime sits between two bounds. `lile/objectives/unlike.py` should expose `η_min` as a configurable floor. Treat any prescribed step `η < η_min(p, w_+)` as a code-smell that the unlike loss is being run in its unsafe regime.

## 4. η_min: smallest step where q_b ≤ p_b (one-step)

`q_b ≤ p_b` ⟺ `Δ_b ≤ log Z` ⟺ `μ(η) := log E_p[exp(Δ_·)] ≥ Δ_b = -η p_b (1 + w_+)`.

Define the threshold:

```
M_composite(η)  :=  -(1/η) · log E_p[exp(Δ_·(η))]
```

Then **`q_b ≤ p_b` iff `p_b · (1 + w_+) ≥ M_composite(η)`** (mirror of the SFT characterization in B §3).

**First-order expansion (η → 0+).** Let `δ_k := Δ_k/η` (the η-independent shift coefficient). At first order, `q_b ≤ p_b` reduces to `E_p[δ] + p_b(1 + w_+) ≥ 0`. Carrying out the expectation and collecting (using `Σ_{j∉S} p_j² = ||p||² - p_b² - p_g²` and `(1-p_b)² + (||p||² - p_b²) = 1 - 2p_b + ||p||²`):

```
E_p[δ]  +  p_b · (1 + w_+)  =  w_+ · ( p_g + p_b - ||p||² )  +  R(p, b)
```

where the **g-independent** remainder is

```
R(p, b)  :=  p_b · (1 - 2p_b + ||p||²) / (1 - p_b).
```

`R(p, b) ≥ p_b · (1 - p_b)` (since `1 - 2p_b + ||p||² ≥ (1-p_b)²`), with equality on the binary support `||p||² = p_b² + (1-p_b)²`. Uniformly non-negative; uniform in V and in g (see §6.3).

The sign of the `w_+`-term partitions the regime:

- **Safe regime (composite is monotone-safe at all η > 0):** `||p||² ≤ p_g + p_b`. The `w_+`-coefficient is non-negative, R is non-negative, so the linearization is non-negative and the threshold inequality holds at every η.
- **Unsafe regime (η_min > 0 needed):** `||p||² > p_g + p_b` *and* `w_+ · (||p||² - p_g - p_b) > R(p, b)`. There is a dominant non-anchor token *and* the SFT-on-good weight is large enough to overpower the non-w_+ remainder. At small η the linearization fails; η_min is the smallest η where the O(η²) terms restore the threshold inequality.

**Closed form for η_min (linearization).** No clean closed form for the exact crossing in general. The linearization gives:

```
η_min^{lin}  :=  ( w_+ · (||p||² - p_g - p_b) - R(p, b) )_+ / [ p_b · (1 + w_+) ]
```

**Calibration (important for the operational recommendation):** the linearization is *conservative but loose*. On the §7 adversarial case (`p = (0.10, 0.10, 0.70, 0.10)`, `w_+ = 10`) it predicts `η_min^{lin} ≈ 2.78` against an empirical crossing of `≈ 0.17` — a ~17× overestimate, because the O(η²) terms become non-negligible once `η · w_+ · (1 - p_g)` is O(1) and `log E_p[exp(η δ)]` exceeds its first-order surrogate by a wide margin. The direction is what matters: `η_min^{lin} ≥ η_min^{true}` in every case checked (1657/2000 strict-consistent under "linearization predicts → empirical agrees within 10×"; remaining 343 are predicted-positive cases where the empirical crossing lies below the linearization, i.e. linearization conservative). Operational implication: a refuse-to-step guard using `η_min^{lin}` produces *false positives* (refuses some safe steps) but never *false negatives* — exactly the failure direction `lile`'s safety-monitor can tolerate.

(linearization around 0; tighter than the true crossing, conservative as a recommendation). This is the formula `lile/objectives/unlike.py` should compute when it needs a recommended step-size floor.

## 5. η_max: KL-collateral ceiling on V \ S

The KL anchor in `lile/objectives/unlike.py` is currently a complement-vocab anchor at the target position with mask `{bad, good}` and renormalized conditional. At one SGD step on the composite loss, the anchor's gradient w.r.t. `z` evaluated at `z_old` is **zero** (the reference distribution is `p|_S` itself, by construction), so the anchor does not contribute a first-order pull at the very first step under plain SGD.

What the anchor *does* control is the off-S collateral: we want `Σ_{j ∉ S} |q_j - p_j| ≤ ε`. From the Δ_j formula:

```
q_j / p_j  =  exp(Δ_j) / Z  =  exp( η · p_j · (p_b/(1-p_b) - w_+) ) / Z
```

For `w_+ > p_b/(1-p_b)` (the operational regime — w_+ is large, p_b is small) the per-token Δ_j is negative, so off-S mass uniformly *decreases*. Two ceilings on η, parallel to §4's two floors:

### 5.a η_max^{lin}(p, w_+, ε) — linearization (sanity metric only)

Linearizing `q_j/p_j ≈ 1 + η · δ_j` and dropping the `(1 + O(η))` correction:

```
η_max^{lin}  :=  ε / [ w_+ · (||p||² - p_b² - p_g²) ]
```

**Calibration (5× under-bound, false-negative side — important).** This formula is **not a strict upper bound**. On 200k random per-step trials (η ∈ [1e-4, 0.05], w_+ ∈ [0.1, 10], V ∈ [3, 20]), the per-step empirical TV exceeds the linearization in 51871/200000 = 26% of steps, with worst-case ratio ≈ 5×. Direction is the dangerous one: `η_max^{lin}` *under-bounds* TV, meaning a refuse-step guard built on `η ≤ η_max^{lin}` permits steps whose actual off-S drift exceeds `ε`. Mirror of §4's calibration paragraph but pointing the opposite way: §4's `η_min^{lin}` was conservative (false-positive only), §5's `η_max^{lin}` is **non-conservative** (false-negative only). Operational implication: never use `η_max^{lin}` as the operational ceiling — log it as a compile-time sanity metric only.

### 5.b TV_sim^{emp}(p, w_+, η) — operational ceiling

At dispatch we already have `(p_b, p_g, ||p||², w_+, η)` and the forward tensor in hand. One step-simulation gives the exact per-step off-S TV:

```
Δ_k  ←  η · compositeDelta(p, b, g, w_+)         # O(V)
q    ←  softmax-renormalize p · exp(Δ)           # O(V)
TV_sim^{emp}  ←  ½ · Σ_{j ∉ S} |q_j - p_j|       # O(V)
```

Refuse-to-step if `TV_sim^{emp} > ε_target`. Same dispatch-site cost order as `η_min^{emp}` (~20 bisection steps × O(V), already accepted as the operational floor in §4).

**Composability requirement (operational form).** The composite is one-step safe iff:

```
η  ≥  η_min^{emp}(p, b, g, w_+)        (§4 — bisection)
TV_sim^{emp}(p, b, g, w_+, η)  ≤  ε    (§5.b — one-step simulation)
```

When either fails the run refuses to step. The closed-form `η_max^{lin}` is logged but does not gate.

## 6. Open seams (push back here)

1. **Anchor as post-step audit (proximal-gradient reading).** Under plain SGD the anchor does not gradient-pull at step one — `∇L_kl(z_old) = 0` since the reference distribution is `p|_S` itself. The §5 ε-bound is therefore a *post-step audit*: compute (η_min, η_max, ε_*) at dispatch, refuse-to-step if the window is empty, measure the actual collateral after the step, alarm if the anchor's ε-ball is exceeded. This matches `lile`'s existing observational-sidecar semantics (safety_monitor weight=0.0; tier-4 unlike-tiered-preconditions dispatch-time warn). The natural-gradient reading is rejected — `lile` ships AdamW, and a bound that holds only under Fisher-metric steps would be operationally wrong. The trajectory bound — N small steps inside ε per-step but cumulative drift outside ε — is the right *next* theorem; it composes on top of (a) and is load-bearing for snapshot-restore and idle-replay. Deferred to `unlike-trajectory-bound.md` (to-be-drafted). The Lean statement in §9 stays per-step.
2. **"Never safe" residue resolved.** Re-ran the §7 sweep on η ∈ [1e-8, 1e4] (vs. the original [1e-4, 10]); residue collapsed from 44/5000 to 0/5000. The originally-flagged trials had η_min < 1e-4 (very small p_b regime, where the unlike push-down on b is O(η · p_b) and needs sub-1e-4 η to flip the linearization). No "genuinely never safe" configs in the random sweep — the analytical η_min from §4 is consistent across extremes. The operational implication: when `lile`'s configured step-size is below the per-sample η_min, the refuse-to-step guard is what saves us, not a wider grid.
3. **V → ∞ behavior of R(p, b) — closed.** After the §4 simplification, R is a function of (p_b, ||p||²) only — independent of g and of V. Lower bound `R ≥ p_b(1 - p_b)` is uniform in V. As V → ∞ with (p_b, p_g) fixed, ||p||² can take any value in `[p_b² + p_g² + (1-p_b-p_g)²/(V-2), 1 - 2(p_b·p_g + ...) ]`; the lower limit converges to `p_b² + p_g²`, giving `R_∞^{lower}(p_b, p_g) = p_b · ((1-p_b)² + p_g²)/(1-p_b) ≥ p_b(1-p_b)`. Same uniform bound holds. The safe-regime claim is therefore uniform in V. Closed.
4. **Composition with B.** B's characterization gives a *per-token* growth predicate at any p; here we have a *per-target* (b only) safety predicate. The natural unification: extend B's M_p threshold to composite losses and read off η_min as the smallest η where `p_b ≥ M_composite(η)` flips for the unlike target.

## 7. Falsifier protocol

**Methodology note.** A sign-flip in Δ_j (j ∉ S) caught during the initial sweep run produced a spurious all-safe result. The corrected formula in §2 (Δ_j = η · p_j · (p_b/(1-p_b) - w_+), unlike contribution positive on off-S logits, SFT contribution negative) is what the sweep below uses. Falsifier triggers on this class of sign error: any "all configs safe" outcome under the adversarial demo regime is a flag.

Two sweeps; both pass on the analysis above, and the §6.2 residue is resolved on the wider grid:

- **Adversarial demo** — `~/identity-matrices/cleo-work/scratch/unlike_eta_min_sweep.py::adversarial_demo`. p = (0.10, 0.10, 0.70, 0.10), b=0, g=1, w_+=10. Empirical η_min ≈ 0.166; q_b > p_b for η ≤ 0.15, q_b < p_b for η ≥ 0.17.
- **Random sweep** — same script, `random_sweep(n=5000)`. Result: 3421/5000 safe at smallest η, 1535/5000 need a positive η_min consistent with the §4 linearization, 44/5000 fall in the §6.2 residue.

Reproduce with `python unlike_eta_min_sweep.py`. Determinism: `random.seed(42)` at sweep entry. Falsifier triggers if any trial's empirical η_min disagrees with the linearization by more than the expected O(η) correction.

## 8. Implementation hooks (what `lile` should do)

- `lile/objectives/unlike.py` — log four metrics, refuse on the two empirical ones:
  - `η_min^{lin}` (§4 closed form) under `unlike/eta_floor_lin` — compile-time sanity, conservative (false-positive only, ~17× on adversarial).
  - `η_min^{emp}` (§4 bisection on `q_b ≤ p_b`, ~20 steps × O(V)) under `unlike/eta_floor_emp` — **operational floor**.
  - `η_max^{lin}` (§5.a closed form) under `unlike/eta_ceiling_lin` — compile-time sanity, **non-conservative** (false-negative only, ~5× under-bound on sweep). Never gates.
  - `TV_sim^{emp}` (§5.b one-step simulation, O(V)) under `unlike/predicted_step_tv` — **operational ceiling**.
- Refuse-to-step guard: refuse if `η < η_min^{emp}` OR `TV_sim^{emp} > ε_target`. Fallback: skip the step, or fall back to `w_+ = 0` (pure unlike, always safe and §5 trivially satisfied).
- **Per-sample dispatch cost.** At V=128k: bisection (~2.5M ops) + step-sim (~0.5M ops) = ~3M fp ops per sample, ~190M per batch-of-64 — sub-ms on GPU. If watched: cache by `(p_b, p_g, ||p||², w_+, η)` fingerprint; fallback to `η_min^{lin}` only if conservatism budget can absorb the 17× overshoot. Out of scope for v1.
- KL anchor — keep the complement-vocab `{bad, good}` mask scope; the §5 ε-bound is what the anchor delivers, document as "post-step audit" not "in-step constraint" (§6.1).
- GLOSSARY — add `composite-safe`: a one-step composite loss is composite-safe at (p, w_+, ε, η) iff η ∈ [η_min(p, w_+), η_max(p, w_+, ε)] and the window is non-empty.

## 9. Lean

Lake project: `lile/docs/research/proofs/lean/RazinSafety/` (Mathlib v4.30.0-rc1).

**Headline theorem (per-step floor only).**

```lean
theorem unlike_composite_step_window
    (p : Distribution V) (b g : Fin V) (hbg : b ≠ g)
    (wPlus : ℝ) (hw : 0 ≤ wPlus) (η : ℝ) (hη : 0 < η) (hV : 0 < V)
    (hMin : etaMinLin p b g wPlus ≤ η) :
    (compositeStep p b g wPlus η hV).prob b ≤ p.prob b
```

`etaMinLin`, `compositeDelta`, `compositeStep` per §2 and §4. Discharge modulo one labeled `taylor-remainder` sorry on the analytical lemma `η · δ_b ≤ log Z` (uniform `O(η²)` Taylor-remainder bound on `M_composite`); the structural reduction (q_b ≤ p_b ⟺ exp(η δ_b) ≤ Z ⟺ η δ_b ≤ log Z) is fully proved.

**Intentionally out-of-scope: the TV conjunct.** The original sketch claimed `η ≤ η_max → tv_off_S p q ≤ ε`. After the §5.a calibration found `η_max^{lin}` non-conservative (26% empirical violation, max 5× ratio) and §5.b moved the operational ceiling to `TV_sim^{emp}` (one-step simulation, not an analytical bound), formalizing the TV side would either (a) embed an empirical calibration constant inside the proof, making the "theorem" a function of which sweep we ran, or (b) axiomatize a definition (the operational TV ceiling is computational, not propositional). Both rejected. The TV ceiling is validated by the falsifier scripts in §7 and the `unlike_eta_min_sweep.py` regression suite, not by Lean. This boundary — analytical content goes to Lean, operational content to falsifiers — is deliberate.
