# Razin-safety, sharpened: which non-target tokens can grow under SFT

Status: **sketch**, pre-Lean. Owner: cleo (`@%23`). Validator: mei (`@%9`).
Date: 2026-04-18.

Parent claim being sharpened: `GLOSSARY.md` classifies `sft`, `weighted_sft`,
`ntp`, `coh` as Razin-safe because they can "only push mass toward the
target." As a statement about *aggregate* non-target mass, that holds. As a
statement about *every individual* non-target token, it does not — a small
non-target can gain absolute probability mass when the shrinkage falls
mostly on a dominant non-target. This document characterizes exactly which
non-targets grow.

---

## 1. Setup

Fix a vocabulary of size `V`. Let `z ∈ ℝ^V` be pre-step logits with
`p := softmax(z) ∈ Δ^{V−1}` strictly positive (`p_j > 0 ∀ j`). Fix a
target token `t` with `p_t ∈ (0, 1)` and a step size `η > 0`.

SFT cross-entropy: `L(z) = −log p_t`.  Its logit-gradient is
`∂L/∂z_j = p_j − 𝟙[j = t]`, so one SGD step yields

```
z'_j = z_j + η · (𝟙[j = t] − p_j).
```

Let `q := softmax(z')`. Writing `β_k := 𝟙[k = t] − p_k`, we have

```
q_j = p_j · exp(η β_j) / Z,
```

with partition-function ratio

```
Z := E_p[exp(η β_·)] = p_t · exp(η (1 − p_t)) + Σ_{k≠t} p_k · exp(−η p_k).
```

## 2. The characterization

Define

```
M_p(η) := −(1/η) · log Z
       = −(1/η) · log E_p[exp(η · (𝟙[·=t] − p_·))].
```

**Theorem (mass-flow characterization).** For every `j ≠ t`,

```
q_j > p_j   ⟺   p_j < M_p(η).
```

**Proof.** For `j ≠ t`, `β_j = −p_j`, so `q_j = p_j · exp(−η p_j) / Z`.
Thus `q_j > p_j ⟺ exp(−η p_j) > Z ⟺ −η p_j > log Z ⟺ p_j < −(log Z)/η = M_p(η)`. ∎

All non-target tokens with prior mass **strictly below** a single scalar
threshold `M_p(η)` grow under one SFT step; all above it shrink; any at
the threshold are fixed points (measure-zero).

## 3. Where the threshold lives

| regime | `M_p(η)` | meaning |
|---|---|---|
| `η → 0⁺` | `‖p‖² − p_t = p_t² − p_t + Σ_{j≠t} p_j²` (= `Σ_{j≠t} p_j² − p_t(1 − p_t)`) | small-step limit. |
| `η → ∞` | `p_t − 1 < 0` | aggressive step; threshold negative → no non-target grows. |
| monotonicity | `M_p(η)` is continuous and non-increasing in `η` (proof: differentiate `log Z` using `Z = E_p[exp(η β_·)]` and Jensen). | the unsafe-growth regime shrinks as `η` increases. |

### Headline: the unsafe regime is at *small* η, not large

This inverts the standard tuning intuition. Practitioners reach for a
small learning rate when they want a "safe" or "gentle" fine-tuning step.
For surgical SFT-style corrections of specific tokens, that intuition is
backwards. Large η consumes the vocabulary into the target and
collapses every non-target. Small η leaves a single dominant non-target
under-compressed, and the redistribution falls disproportionately on the
tail — small-prior tokens can gain absolute mass while aggregate
non-target mass shrinks. The adversarial case is the gentle case.

This result has a direct operational consequence for the daemon's
`unlike` objective with positive teacher (analyzed in companion doc
`unlike-kl-step-size-bound.md`, Problem A): under tiny LR the
SFT-on-good component can push the *bad* token *up* faster than the
unlike term pushes it down. The "small η is gentler" intuition fails
both for B's mechanism and A's composition.

**Household-AI corollary (Mei's framing).** Push toward
"he who shall not be named" while "Voldemort" dominates and "Sauron" is
a 1% rounding error: SFT can raise `p("Sauron")` in absolute terms. The
small token grows because the dominant non-target drops fast enough that
the normalizing constant `Z < 1`, i.e. `M_p(η) > 0`.

## 4. Aggregate properties (already-known, restated for completeness)

**Proposition (target strictly grows).** `q_t > p_t`. *Proof.* `β_t = 1 − p_t > 0`,
so `q_t = p_t · exp(η(1 − p_t)) / Z`. Since `Z = E_p[exp(η β_·)]` is a
mixture of `exp(η β_k)` values including `exp(η(1 − p_t))` but also
`exp(−η p_k) < exp(η(1 − p_t))` for every `k ≠ t`, we have
`Z < exp(η(1 − p_t))`, hence `q_t > p_t`. ∎

**Proposition (non-target mass strictly shrinks).** `Σ_{j≠t} q_j < Σ_{j≠t} p_j`,
equivalently `q_t > p_t`. Follows from the previous proposition. ∎

**Razin-safety, in precise form.** SFT is aggregate-Razin-safe (no mass leaves
the target as a group) but **not** pointwise-Razin-safe (individual
non-target tokens can gain absolute mass, as §2 characterizes).

## 5. Existence (what was already a calculation)

The set `S(p, t, η) := {j ≠ t : q_j > p_j}` is non-empty iff

```
min_{j ≠ t} p_j  <  M_p(η).
```

Mei's numerical witness, for the record: `V=3`, `p=(0.10, 0.89, 0.01)`,
`t=0`, `η=1`. Then `Z ≈ 0.6213`, `M_p(1) ≈ 0.4759`, so the grower set is
`{2}` (p_2 = 0.01 < M; p_1 = 0.89 > M). Post-step
`q ≈ (0.3959, 0.5882, 0.0159)` — the 0.01 token grew to 0.016 in absolute
terms while aggregate non-target mass dropped from 0.90 to 0.60.

## 6. Proposed Lean statement

```
-- V : Type with [Fintype V] [DecidableEq V]
-- p : V → ℝ, simplex, all-positive
-- t : V
-- η : ℝ, η > 0

theorem sft_mass_flow_iff
    (p : V → ℝ) (hp_pos : ∀ j, 0 < p j) (hp_sum : ∑ j, p j = 1)
    (t : V) (ht : p t < 1)
    (η : ℝ) (hη : 0 < η)
    (j : V) (hjt : j ≠ t) :
    (q p t η) j > p j ↔ p j < M p t η
```

with `q p t η := softmax (log ∘ p + η • (δ_t − p))` and
`M p t η := −(1/η) * log (∑ k, p k * exp (η * (indicator (k = t) 1 − p k)))`.

**Dependencies in Mathlib I intend to lean on:**

- `Mathlib.Analysis.SpecialFunctions.Log.Basic` — `Real.log`, `Real.exp`, their
  monotonicity.
- `Mathlib.Probability.Distributions.Simplex` (or manual on-`Finset`).
- No measure theory needed — everything is a finite sum.

**Proof plan in Lean.** The characterization is a direct algebraic manipulation:
(`q_j > p_j`) ↔ (`exp(−η p_j) > Z`) ↔ (`−η p_j > log Z`) ↔ (`p_j < M`).
Each step is one `rw` or `div_lt_div_iff` / `Real.log_lt_log_iff` application.
The non-trivial lemma is `Z > 0` (needed to turn `q_j > p_j` into `exp(...) > Z`
without sign-flipping the inequality), which follows from `p` being in the
open simplex.

Estimated Lean effort: one afternoon for the characterization + aggregate
corollaries, once the Lake project is set up and `Finset.sum` tactics are
loaded.

## 7. Falsifier protocol

### 7a. Existence — done

Mei's `V=3` instance is a numerical witness; folded into §5.

### 7b. Characterization — falsifier sweep

Protocol: draw random simplex points and confirm the predicate
`p_j < M_p(η)` partitions growers from shrinkers perfectly.

- **Sample space.** `V ~ Uniform{2, …, 50}`, `p ~ Dirichlet(𝟙_V)` (via
  `exponential(1)` normalisation), `t ~ Uniform{0, …, V−1}`, `η` log-uniform
  in `[0.01, 10]`.
- **Trials.** 10 000.
- **Pass criterion.** For every `(trial, j ≠ t)`: `(p_j < M_p(η)) ⇔ (q_j > p_j)`.
  Budget: any mismatch within `|p_j − M_p(η)| < 10⁻⁹` is a floating-point tie
  and not counted; all others count. Expected non-numerical mismatch count: **0**.

### 7c. Result

Ran (2026-04-18, `~/identity-matrices/cleo-work/scratch/razin_characterization_sweep.py`):

```
Total trials: 10 000
Non-numerical mismatches: 0
Near-threshold ties: 0
```

Characterization holds across the sweep. Mei's 5% error budget is met
(budget was 5%, observed was 0%).

## 8. From theorem to operational safety signal

The characterization isn't only a static property of the loss surface;
it generates a cheap, in-loop alarm. This section promotes that
instrumentation from side-note to load-bearing — at Mei's request.

### 8a. GLOSSARY column

Proposed addition to the Razin-safety table:

| objective | aggregate-safe | pointwise-safe | notes |
|---|:---:|:---:|---|
| `sft` | ✓ | ✗ | tail non-targets with `p_j < M_p(η)` grow; adversarial if a dominant non-target absorbs shrinkage |
| `weighted_sft` | ✓ | ✗ | same mechanism, weight scales `η` |
| `ntp` | ✓ | ✗ | same |
| `coh` | ✓ | ✗ | same, applied per hindsight token |
| `kto` | ✗ | ✗ | pre-existing |
| `hinge` | ✗ | ✗ | pre-existing |
| `unlike` (needs its own row) | depends | depends | analysis in `unlike-kl-step-size-bound.md` |

### 8b. Watchlist alarm — the per-step instrument

The bare characterization talks about `min_{j ≠ t} p_j` vs `M_p(η)`. For
the household-AI use case the daemon doesn't care about an *arbitrary*
tail token gaining mass; it cares about *specific* unwanted tokens — a
"don't-say-this" watchlist `W ⊂ V`. The alarm generalizes cleanly:

```
For each SFT-family training step at target t with step size η:
    Compute  M_p(η)  and  growing := { w ∈ W : p_w < M_p(η) }.
    If growing ≠ ∅, log a "pointwise-unsafe-step" event with payload (t, η, growing).
```

Cost: `O(|W| + V)` per step (`O(V)` for `M_p(η)` via one log-sum-exp
already in the forward; `O(|W|)` for the watchlist scan). Trivial
relative to the forward pass.

This is the operational meaning of "Razin-safe at this step" — concrete,
cheap, interpretable, and sharper than any aggregate KL number.

### 8c. Cumulative alarm rate — multi-step bridge without a multi-step proof

The one-step theorem (§2) does not characterize trajectories. A
multi-step Lyapunov argument is a separate, harder theorem and is not in
scope for this document. **However**, the per-step alarm rate is itself a
multi-step signal:

> If `pointwise-unsafe-step` fires at rate `r > 0` over a long run, the
> cumulative absolute mass of any consistently-watched-and-grown token
> can inflate even when each individual step's growth is sub-ε.

Quantitatively: for any fixed `w ∈ W`, define `r_w` as the empirical
fraction of training steps on which `w` was in the grower set. The
expected log-odds drift `E[Δ log(p_w / (1 − p_w))]` per step is bounded
below by the conditional growth rate at triggered steps; a positive
`r_w` together with a positive expected step-conditional growth gives a
strictly positive drift, hence unbounded log-odds inflation in
expectation. This is the operational bridge: monitor `r_w`, not the
existence of any single unsafe step.

A formal multi-step bound — including the rate at which
small-initial-`p_w` tokens can be inflated under realistic SFT streams —
is a follow-on theorem for after A lands; flagged here so the
instrumentation is shipped with the framing already in place.

### 8d. Suggested implementation hook

Either:
- Inside `lile/objectives/sft.py` (and `weighted_sft.py`, `coh.py`, `ntp.py`),
  return `M_p(η)`, `growing` and the per-step `r_w` in `components: dict`
  so the metrics backends pick it up "for free."
- Or as a thin wrapper objective `pointwise_safe_sft` that delegates to the
  underlying loss and adds the alarm bookkeeping, opt-in per training run.

Owner call (PO / lile-backend) — not in scope for this proof. Flagging so
the design has somewhere to land when the watchlist primitive ships.

## 9. Open questions Mei should push back on before I open Lean

1. **Scope of `V`.** Lean statement above is general `V` with `Fintype`. Mei asked
   for V general, which I read as "not hardcoded to 3." Confirm this reading
   vs. "parametric in `V` but proven at each `V`" (the former is standard and what
   I'm defaulting to).
2. **Target condition.** I assume `p_t ∈ (0, 1)`. At `p_t = 1` the step is a
   no-op (no mass to redistribute). At `p_t = 0` the log-gradient blows up
   because `log p_t = −∞`. Standard. Flagging so we agree on boundary.
3. **Real vs. multi-step.** This is one step. Multi-step SFT composes non-linearly
   and a token can oscillate (grow, then shrink, then grow again as the
   distribution reshapes). I'm NOT claiming anything about the trajectory. If
   Mei wants a multi-step statement, that is a separate theorem (and harder —
   likely needs a Lyapunov argument or a direct recursion on `M_p`).
4. **Relation to Razin 2024.** The original Razin paper is about DPO/preference
   objectives and their non-monotonic mass displacement in the signed case.
   This sharpening says: even the *unsigned* (SFT) case has a mild
   displacement, but characterizable and usually benign. Good to be explicit
   that we're not contradicting Razin — we're extending the safety analysis
   into the "safe" side of their taxonomy.
