# Lean as rigor tool & non-linear frontiers for online sample efficiency

**Author:** Cleo (SVP Formal Verification & Mathematical Reasoning)
**Audience:** lile architects, Mei, future agents picking up this thread.
**Status:** living document; updates allowed without re-review provided the boundary policy in §0 is preserved.

---

## 0. Boundary policy (canonical, do not violate)

| Claim type | Where it lives | Why |
|---|---|---|
| Analytical implication (`A → B` for all `(p, η, w_+, …)` in stated domain) | Lean | Only place that *machine-checks* the universal quantifier. |
| Operational ceiling (compute-then-refuse at runtime) | Falsifier script + dispatch-site code | Computational, not propositional. Axiomatising would be a category error. |
| Calibration constant (95th percentile of N=k sweep) | JSON config + telemetry tag | Embedding it in Lean makes the "theorem" a function of which sweep we ran. |
| Empirical observation under regime label | `STATUS.md` / trajectory JSONL | Snapshot in time; not a universally quantified claim. |

This boundary is **load-bearing**. The §5 → §5.a/§5.b refactor in `unlike-kl-step-size-bound.md` exists because we tried to formalise an operational ceiling and Lean refused to close the proof — which is exactly Lean acting correctly.

---

## 1. What Lean has actually bought us so far

Two concrete wins, one near-miss, one anti-pattern caught.

### 1.1 Win — forced relabel of §5 from "bound" to "linearisation"

The original A draft labelled `η · w_+ · (||p||² - p_b² - p_g²)` as a **per-step ε bound**. The 200k-trial sweep caught the 26 % violation rate; Lean (had we attempted to close the theorem at that point) would have refused the sub-goal `η · ||δ||_∞ ≤ ε`. The refactor produced:

- §5.a `η_max^{lin}` — labelled **sanity metric, false-negative side**, never gates.
- §5.b `TV_sim^{emp}` — operational ceiling, runtime simulation.

This is Lean buying clarity, not a new theorem. Cost: zero (we hadn't started the proof yet). Value: the doc no longer claims something false.

### 1.2 Win — TV-conjunct out-of-scope discipline

When discharging `unlike_composite_step_window` I had three options for the broken TV side: (a) rescope, (b) calibrated multiplier, (c) axiom. Mei picked (a) on first principles ("axiom is for assertions, definitions are computational"). The rescope is documented in `unlike-kl-step-size-bound.md` §9 with the rationale spelled out, and the Lean file carries a top-of-file `/- TV out-of-scope -/` comment. Future readers see the boundary in the source, not in git archaeology.

### 1.3 Near-miss — Δ_j sign error in trajectory sweep

The `compositeDelta` sign error (initially `(w_+ − p_b/(1−p_b))`, correct is `(p_b/(1−p_b) − w_+)`) was caught by the falsifier giving an unrealistic "all-safe" result, **not** by Lean. But: had the canonical definition lived in Lean from day one and been the single source-of-truth that the Python sweep imported (e.g., via PyO3 or transcription from a single spec), the divergence would have been impossible. Future composite losses should canonicalise in Lean first, transcribe to Python second.

### 1.4 Anti-pattern caught — Φ_anc anchor-discount

The proposed `Φ_anc = Σ ε_i · exp(−α · E_i)` was algebraically clean but mathematically misaligned: anchor restoring force is on-S, off-S drift is what we wanted to bound. Lean would not have caught this — it's a domain-modelling error, not a derivation error. **Lean is not a substitute for asking "does my object actually measure the thing I care about."** The §3-orthogonal-side-theorem framing in the trajectory doc memorialises this.

---

## 2. Where Lean does NOT help (and why falsifiers / runtime are the right tool)

| Thing | Why Lean is wrong |
|---|---|
| Random-sweep sign errors | Falsifiers find these in seconds; Lean has no built-in counter-example generation for `Real`-valued universal claims. |
| Calibration constants (`K_session* = 0.27`) | These are 95-percentile statistics of an empirical distribution. Even if the sweep is reproducible, the *number* is not a universal claim. |
| Tokenizer geometry (`span_prefix` mask, T3.1) | The correctness criterion is "tokenizer-decoded bytes endswith X" — this depends on the tokenizer implementation. Formalising the tokenizer is overkill. |
| Float-vs-real divergence under bf16 | Mathlib has interval arithmetic, but the cost-value is poor for a daemon that already ships `torch.use_deterministic_algorithms(True)` as the operational guard. |
| "Did I deploy what I proved?" | Lean only checks the artifact you build. The wire-from-proof-to-Python step is itself out-of-Lean unless we adopt extraction (heavy machinery, not justified by a 35-line proof). |

---

## 3. Logic in vector spaces — what Lean targets are tractable and load-bearing

### 3.1 What we already have (in `RazinSafety`)

- `Distribution V`: open simplex on `Fin V` with sum-one + strict positivity invariants.
- `Distribution.softmax`, `Distribution.shift`: parameterisations used in both proofs.
- `sft_mass_flow_iff`: per-token mass-flow direction under SFT (B, no sorrys).
- `unlike_composite_step_window`: one-step `q_b ≤ p_b` from the linearised floor (A, one labelled `taylor-remainder` sorry).

### 3.2 What's available in Mathlib v4.30.0-rc1 that we haven't tapped

Useful items confirmed present in the pinned Mathlib:

- `Mathlib.InformationTheory.KullbackLeibler.Basic` — `klDiv`, `klDiv_self`, `klDiv_smul_*`, Gibbs' inequality.
- `Mathlib.InformationTheory.KullbackLeibler.ChainRule` — KL chain rule (relevant to trajectory composition).
- `Mathlib.MeasureTheory.Measure.Tilted` — exponential-tilting machinery; *the measure-theoretic name for the softmax-shift operation*. Reformulating our `Distribution.shift` as a tilt of a finite measure on `Fin V` would let us reuse Mathlib's tilted-measure lemmas instead of re-deriving each one.
- `Mathlib.Analysis.Convex.SpecificFunctions.Basic` — convexity of `Real.exp` and `Real.log`; load-bearing for any Jensen-style argument.
- `Mathlib.Analysis.MeanInequalities` — Hölder, Cauchy-Schwarz, generalised means.
- `Mathlib.Analysis.Calculus.MeanValue` — the only honest way to discharge the `taylor-remainder` sorry.

### 3.3 What's missing and worth building

In priority order — each is small, useful, and composes with the existing two theorems.

#### **Target T1 — Pinsker for finite distributions** (`tv_le_sqrt_klDiv_div_two`)

Statement: for `p, q : Distribution V`,
```
TV(p, q) ≤ √( KL(p || q) / 2 ).
```

Why it matters: this is the analytical TV ceiling we've been missing. After T1, we can replace §5.b's `TV_sim^{emp}` (operational simulation) with a *closed-form* upper bound:
```
TV(p, q_η) ≤ √( KL(p || q_η) / 2 ) ≤ √( (η²/2) · Var_p[δ] / 2 + O(η³) )
            = (η/2) · √( Var_p[δ] ) + O(η²)
```
This is linear in η (good), depends on the variance of the logit shift (computable in O(V)), and is an honest upper bound (not a 5×-under-bound linearisation). It would let us drop the calibration constant in §5.b.

Effort: Pinsker in Mathlib was open as of v4.30.0-rc1; the finite-`Fin V` version reduces to bounding `(p−q) · log(p/q) ≥ 2 · TV²` via the elementary `(a−b)² ≤ (a+b)·(a·log(a/b) + b·log(b/a))` lemma, which does have a Mathlib-shaped proof using `Real.log` convexity. ~60 lines.

#### **Target T2 — Softmax shift KL formula** (`klDiv_softmax_shift`)

Statement: if `p = softmax(z)` and `q = softmax(z + Δ)`, then
```
KL(p || q) = log Z − E_p[Δ]      where Z = ∑_k p_k · exp(Δ_k).
```

Why it matters: every KL anchor evaluation in `lile/objectives/unlike.py` is a special case. Once we have this, the `Δ_b ≤ log Z` step in `unlike_composite_step_window` becomes a corollary rather than a hand-derived intermediate.

Effort: pure algebra, ~30 lines, no analytical content beyond `Real.log` arithmetic. Could be discharged today.

#### **Target T3 — Variance-of-Δ lemma** (`var_compositeDelta_le`)

Statement: a closed-form upper bound on `Var_p[compositeDelta p b g w_+]` in terms of `(p_b, p_g, ||p||², w_+)`.

Why it matters: combined with T1 and T2 it gives the Pinsker route concrete: `TV(p, q_η) ≤ (η/2) · √V̄(p, b, g, w_+) + O(η²)` where `V̄` is the closed-form variance bound. This is the *honest* analytical replacement for the broken §5.a.

Effort: half a page of algebra, ~80 lines in Lean.

**T1 + T2 + T3 together would close the §5.a gap analytically and let us delete the `taylor-remainder` sorry in `UnlikeStepWindow.lean` because the final step becomes a Pinsker-bound argument rather than a Taylor-remainder argument.** This is the highest-leverage Lean follow-up.

#### **Target T4 — Trajectory composition as a supermartingale**

Statement: under per-step gating `η_k ≥ η_min^{emp}(p_k)`, the sequence `M_k := log p_k(b_k)` is a supermartingale w.r.t. the σ-algebra generated by `(b_k, g_k, w_{+,k}, η_k)`. Hence by Azuma-Hoeffding (bounded differences are guaranteed by the gating), the cumulative drift `∑ |M_{k+1} − M_k|` is sub-Gaussian with explicit constants.

Why it matters: the K_session* = 0.27 number is currently a 95-percentile of an empirical distribution under random `(b, g)` selection. A supermartingale bound would give a *worst-case* (not random-decorrelated) cumulative drift bound parameterised by N and the per-step gating, with concentration tail. This is what "K_session* with a guarantee, not just a prior" looks like.

Effort: Mathlib's `Probability.Martingale` is general but heavy; building this on `Fin V` discrete probability would need ~200 lines and is a real proof, not an algebraic exercise. Defer until T1-T3 land.

---

## 4. Numerical analysis — where rigour would change practice

### 4.1 Numerical instability: the three regimes in our code

1. **Softmax overflow on logits** — handled by `torch.softmax`'s built-in max-subtraction. No action needed.
2. **`log(1 − p_b)` underflow when `p_b → 1`** — *active risk* in `unlike.py`. The unlike loss is `−log(1 − p_b)`, and as the unlike target's probability approaches 1 the gradient blows up. The current code clips with `torch.clamp(p_b, max=1 − 1e-7)` (single-precision floor); in bf16 the meaningful clip floor is closer to `1 − 8e-3`. **Concrete action**: audit whether `unlike.py` uses fp32 or bf16 for the per-token probability extraction; if bf16, the clip threshold is silently in the wrong regime.
3. **KL divergence on the masked vocabulary complement** — `kl_anchor` in `unlike.py` computes `KL(p|_S || ref|_S)` after renormalisation. Under bf16 the renormalisation step amplifies relative error by `1/(p_b + p_g)`. When `p_b + p_g < 1e-3` the KL value is dominated by quantisation noise. **Concrete action**: cast the renormalisation to fp32, return KL as fp32 scalar, log both bf16-and-fp32 KLs to detect divergence.

What Lean buys here: nothing directly. What it would buy *if* we wrote interval-arithmetic versions of the per-step shifts: an analytical guarantee that the refuse-step direction is preserved under bf16 quantisation. Cost-value is poor for v1; revisit only if the cold-baseline A/B starts depending on bf16 ↔ fp32 differences.

### 4.2 Gradient behaviour worth pinning

#### **Gradient norm under composite loss**
For the composite step `L = −log(1 − p_b) + w_+ · (−log p_g)`, the per-logit gradient norm is bounded:
```
||∇_z L||₂² = (p_b · (1 + w_+))² + Σ_{k ≠ b} p_k² · (p_b/(1−p_b) − w_+)²·[k ∉ {b,g}] + …
```
Specifically `||∇||₂ ≥ p_b · (1 + w_+)` always — which means the unlike gradient is bounded *below* by `p_b`. **Implication**: when `p_b` is small (the unlike target is already low-probability), the unlike push is weak. This is the mathematical reason that lr=1e-5 can be unsafe in lile (per memory `lr_1e5_unsafe_for_unlike`): with small `p_b` the unlike contribution is dominated by the `w_+ · (−log p_g)` SFT push-up, which can elevate `p_b` indirectly through the softmax coupling. *Lean target*: a clean `composite_grad_lower_bound` lemma would make this gradient-direction failure mode explicit in the proof, so we can flag it at dispatch.

#### **Vanishing-gradient regime as a step-size lower bound**
A symmetric concern: if `p_b → 0` the unlike gradient vanishes, but so does the SFT-on-good gradient when `p_g → 1`. The composite step does *nothing* in that regime. Currently `lile/objectives/unlike.py` does not detect this; a per-token "gradient too small to refuse" predicate would let us short-circuit the refuse-step machinery (waste of a forward call) and emit a `gradient_dead_zone` event.

#### **Variance of stochastic gradient on token-position estimator**
The unlike loss is computed at `target_position` only. The gradient variance depends on which token is chosen as the bad target. For corpora with rare bad-tokens, the variance can swing 100× between samples. **Implication**: the empirical η_min from the bisection is itself a random variable; the operational floor should be `η ≥ η_min^{emp} · (1 + δ_var)` where `δ_var` accounts for cross-batch variance. *Lean target*: not directly relevant; this is a runtime-statistics question, falsifier territory.

---

## 5. Non-linear math directions for sample-efficient online learning

The lile mission is *bespoke household AI that live-learns* (project memory `lile_vision`). Sample efficiency = each correction generalises rather than memorises. Six concrete avenues, ordered by effort × payoff:

### 5.1 Mirror descent on the simplex with KL Bregman geometry

**The pitch.** SGD on logits is implicitly using Euclidean geometry, but the natural metric on the simplex is the KL Bregman divergence. Replacing the AdamW step with a *mirror descent* step under the negative-entropy mirror map gives an update of the form
```
p_{k+1, j} ∝ p_{k, j} · exp(−η · ∇L(p_k)_j),
```
which is *exactly the multiplicative-weights update* and has provably tighter regret bounds for online prediction over the simplex (`O(√(T · log V))` vs SGD's `O(√(T · V))`).

**Why it matters for lile.** Logit-space SGD penalises rare tokens disproportionately because the parameter-space distance of `(0.001 → 0.0001)` is tiny in logits but large in KL. Mirror descent's update is *scale-invariant in probability* — moving a 1% token to 0.1% takes the same effort as moving a 50% token to 5%. For a tutor stream where the bad targets are often rare tokens (misconceptions are tail outputs), this is a structural sample-efficiency improvement.

**Lean angle.** Mathlib's `Mathlib.Analysis.Convex.Mirror` (if present) gives the mirror-map machinery. Bregman divergences are first-class in `Mathlib.InformationTheory`. The regret bound (`O(√(T · log V))`) is a clean theorem to formalise once T1-T3 land.

**Falsifier angle.** Implement a `MirrorAdamW` optimiser as a 200-line drop-in for one objective; A/B against AdamW on the cold-baseline harness. If sample efficiency improves, ship.

**References to read first:**
- Bubeck, *Convex Optimization: Algorithms and Complexity*, ch. 4 (mirror descent on the simplex).
- Cesa-Bianchi & Lugosi, *Prediction, Learning, and Games* (multiplicative weights).
- arXiv:2007.05680 (Lei-Yi 2020, stochastic mirror descent for non-convex).

### 5.2 Online Newton / empirical Fisher natural gradient

**The pitch.** AdamW approximates the Fisher information matrix with diagonal second moments. For language model fine-tuning, the *true* Fisher has block structure over attention heads — a low-rank-plus-diagonal approximation (K-FAC, Shampoo) gives a step direction much closer to the natural gradient. Empirically, K-FAC-style optimisers reach the same loss with 2–3× fewer samples on small fine-tuning workloads.

**Why it matters for lile.** Each `/v1/train` call is one or a few samples. Sample efficiency = wall-clock-to-correct-behaviour. A 2× sample-efficiency gain halves the user-perceived "time to absorb a correction."

**Lean angle.** Out of scope — Fisher analysis is real-analytic, not algebraic; would need substantial new Mathlib infrastructure.

**Falsifier angle.** Shampoo and Adafactor are public. Drop one in for the unlike objective; A/B on cold-baseline.

**References:**
- arXiv:1503.05671 (K-FAC, Martens-Grosse).
- arXiv:1804.04235 (Shampoo, Anil et al.).
- arXiv:2010.06195 (Adafactor for LLMs).

### 5.3 Concentration inequalities for trajectory bounds (Empirical Bernstein)

**The pitch.** The current `Φ_obs ≤ K_session*` is a 95-percentile bound under random `(b, g)` selection — *Hoeffding-class* (worst-case bounded differences). But the per-step variance of `Δ` shrinks as the trajectory approaches a steady state (the gradient gets smaller as the model adapts). Empirical Bernstein inequalities (Maurer-Pontil 2009) replace `√(N · Δ_max²)` with `√(N · σ_emp² + N^(1/3) · Δ_max)`, which is much tighter when `σ_emp << Δ_max`.

**Why it matters for lile.** A tighter trajectory bound means we can afford *more* corrections per session before refuse, without compromising safety. Concretely: replace `K_session* = 0.27` with `K_session(N, σ_emp²) = 1.96 · √(N · σ_emp² / V) + Δ_max · log(2/δ) / N` for a 95% confidence interval. If `σ_emp²` is 10× smaller than the worst-case variance (typical for late-trajectory steps), this raises the effective session budget by ~3×.

**Lean angle.** Mathlib has Hoeffding (`Mathlib.Probability.IdentDistrib` and friends); empirical Bernstein is not present. Formalising it is real work but tractable (~400 lines) and would be a citable Mathlib contribution.

**Falsifier angle.** Easy: re-run the trajectory sweep, log per-step `Var_p[Δ]` alongside cumulative TV, fit the Bernstein bound, compare to the empirical 95-percentile.

**References:**
- Maurer & Pontil (2009), "Empirical Bernstein bounds and sample variance penalization." arXiv:0907.3740.
- Audibert, Munos, Szepesvári (2007), "Tuning bandit algorithms in stochastic environments." (Original empirical-Bernstein use case.)

### 5.4 Wasserstein / optimal-transport drift on token embeddings

**The pitch.** TV in token-index space treats `"yes" → "no"` and `"yes" → "yeah"` as equally bad. They are not. A 1-Wasserstein metric on token embeddings (`W_1(p, q) := inf_{coupling π} E_π[||emb(x) − emb(y)||]`) is a semantically-aware version of TV: drift between near-embedded tokens is cheap, drift between far-embedded tokens is expensive.

**Why it matters for lile.** The trajectory bound currently treats all token drift equally, which means a tutor that legitimately moves output style from "casual" → "formal" (low-W_1 drift, lots of cheap permutations) hits the same `K_session*` ceiling as a tutor that moves output from "correct" → "wrong" (high-W_1 drift). A W_1-based budget would let style adaptations proceed freely and refuse only on semantic regressions.

**Lean angle.** Mathlib has Wasserstein (`Mathlib.MeasureTheory.Measure.Wasserstein`) but the discrete-`Fin V` version on token embeddings would need a custom construction. Defer.

**Falsifier angle.** Heavy: requires precomputed token-embedding distances, a transport solver (`scipy.optimize.linear_sum_assignment` for V ≤ 1000, Sinkhorn for larger). Worth a prototype on a small vocab.

**References:**
- Peyré & Cuturi, *Computational Optimal Transport* (free book).
- arXiv:1907.05787 (Wasserstein in NLP).
- arXiv:2106.05217 (Sinkhorn divergences for language modelling).

### 5.5 SDE limit of small-step SGD (Li–Tai–E)

**The pitch.** Under η → 0, SGD converges weakly to a stochastic differential equation
```
dW_t = −∇L(W_t) dt + √η · Σ(W_t)^(1/2) dB_t,
```
where `Σ` is the empirical gradient covariance. The steady-state distribution is `π(W) ∝ exp(−2 L(W) / η)` (Gibbs), and the convergence rate to steady state is governed by the spectral gap of the Fokker-Planck operator.

**Why it matters for lile.** This gives an *analytical* answer to "what does the model converge to under continual online updates?" The Gibbs steady state means the model relaxes towards the loss-weighted stationary distribution of the training stream. Sample-efficient learning = high spectral gap = fast mixing. The spectral gap is computable (Hessian eigenvalues at the local minimum); there's a concrete `dispatch_warmup` schedule that exploits this.

**Lean angle.** Out of scope (heavy real analysis, requires functional-analytic infrastructure not in our Mathlib pin).

**Falsifier angle.** Simulate the SDE for a small Qwen3-0.6B fine-tune; compare predicted vs empirical loss trajectory. If the SDE prediction tracks empirical to within 10%, it's a usable model.

**References:**
- Li, Tai, E (2017), "Stochastic modified equations and adaptive stochastic gradient algorithms." arXiv:1511.06251.
- arXiv:2102.12470 (E-Ma-Wu, ML and the SDE perspective).
- arXiv:2106.06530 (Yang et al., implicit regularisation via SDE).

### 5.6 RKHS / kernel methods for active sample selection

**The pitch.** Not every correction is equally informative. A kernel-method view of the model (RKHS over token embeddings) lets us *predict* the marginal gain of a candidate correction before applying it. Standard active-learning bounds (Settles 2010, BAIT/BALD families) give `O(1/ε)` sample complexity to reach `ε`-loss, vs `O(1/ε²)` for random sampling.

**Why it matters for lile.** The user is the data source. If we can predict which corrections are "high-information" before asking the user to provide them, we cut the user's labelling burden by 10× or more. This is the *active-learning* angle that no continual-learning library currently exploits.

**Lean angle.** None; RKHS theory is real analysis.

**Falsifier angle.** Build a marginal-gain predictor head on the model embeddings; A/B "user-driven random" vs "agent-suggested high-information" correction streams.

**References:**
- Settles (2010), "Active learning literature survey." (Foundational.)
- arXiv:1906.08158 (BatchBALD, Kirsch et al.).
- arXiv:2106.09675 (BAIT, Ash et al.).

---

## 6. Top-5 prioritised next moves

Ranked by `(payoff × tractability) / cost`. Each is one workstream; multiple can run in parallel.

| # | Item | Payoff | Effort | Owner candidate |
|---|---|---|---|---|
| 1 | **T1 + T2 + T3** (Pinsker → softmax-shift KL → Δ-variance → analytical TV ceiling) | Replaces §5.a + deletes the `taylor-remainder` sorry → A theorem becomes sorry-free | ~3-5 days Lean | Cleo |
| 2 | **bf16 audit on `unlike.py`** (clip floor, KL renormalisation precision) | Catches a silent regime where refuse-step direction inverts | ~2 hours grep + numerical tests | lile-backend |
| 3 | **Mirror-descent prototype** on one objective (5.1) | Sample-efficiency upper-bound from `O(√(T · log V))` regret; if it ships, halves user-perceived correction latency | ~1 week Python + A/B | architect / lile-backend |
| 4 | **Empirical-Bernstein trajectory bound** (5.3) | 3× session budget at same safety; defers refuse-session events; user can keep correcting | ~1 week math + falsifier; Lean later | Cleo + Mei |
| 5 | **Active-correction predictor** (5.6) | 10× reduction in labelling burden; hardest payoff to predict but largest if it works | ~3 weeks; needs research-prototype mindset | Mei (research scope) |

---

## 7. Anti-targets (we should NOT do these)

- **Formalise the entire `lile/objectives/` directory in Lean.** The win is concentrated in the analytical claims (1-2 theorems per objective at most). Industrial-scale formalisation is a 10-100× cost multiplier with marginal added confidence beyond targeted theorems.
- **Replace AdamW system-wide before the mirror-descent prototype proves out.** Optimiser swaps cascade through the safety stack (η_min calibration, anchor scope, snapshot semantics). Prove in one objective first.
- **Pin K_session* to a Lean theorem.** It's a percentile of an empirical sweep. Wrong category.
- **Pursue Wasserstein TV before the V_1 Pinsker route closes.** Higher payoff long-term, but T1-T3 are 3 days of work that close an open sorry; W_1 is weeks of work that opens a new research question.

---

## 8. Open seams (things I'm uncertain about)

1. **Can Pinsker's bound be tighter than the empirical 5× for the §5 closed form?** Need to compute `(η/2) √(Var_p[δ])` numerically on the same 200k-trial sweep that falsified §5.a, see how often the Pinsker bound is loose vs tight. If Pinsker is loose by 10×+, T1-T3 is a less attractive path than I think.
2. **Mirror descent under quantisation.** Multiplicative-weights updates compound; bf16 quantisation noise compounds with them. May need fp32 EMA buffers for the mirror map.
3. **SDE analysis for non-convex losses.** The Li-Tai-E result needs `∇L` to be `L`-Lipschitz; for transformer losses this is empirically true but not proven. The convergence-rate constants are unusable without Lipschitz bounds. Mostly a *qualitative-insight* tool, not a quantitative one.
4. **Active-learning under continual setting.** Most active-learning literature assumes a *batch* setting where the agent picks K items and trains. Online active learning under continual updates is much less studied; the BatchBALD-style scoring may need redesign.

---

## 9. References (curated, alphabetical by first author)

- Anil et al., *Scalable Second-Order Optimization for Deep Learning*. arXiv:1804.04235 (Shampoo).
- Audibert, Munos, Szepesvári (2007). *Tuning bandit algorithms in stochastic environments.* (Empirical Bernstein.)
- Ash et al., *Gone Fishing: Neural Active Learning with Fisher Embeddings*. arXiv:2106.09675 (BAIT).
- Bubeck, S. *Convex Optimization: Algorithms and Complexity*. (Mirror descent.)
- Cesa-Bianchi & Lugosi, *Prediction, Learning, and Games*. CUP. (MWU.)
- E, W., Ma, C., Wu, L. *Machine learning from a continuous viewpoint*. arXiv:2102.12470.
- Kirsch, A. et al., *BatchBALD*. arXiv:1906.08158.
- Lei, Y. & Yi, Y. *Stochastic Mirror Descent for Non-Convex Optimization*. arXiv:2007.05680.
- Li, Q., Tai, C., E, W. *Stochastic modified equations*. arXiv:1511.06251.
- Martens, J. & Grosse, R. *Optimizing Neural Networks with Kronecker-Factored Approximate Curvature*. arXiv:1503.05671 (K-FAC).
- Maurer, A. & Pontil, M. *Empirical Bernstein bounds and sample variance penalization*. arXiv:0907.3740.
- Peyré, G. & Cuturi, M. *Computational Optimal Transport*. (Free; arxiv-style book.)
- Settles, B. (2010). *Active learning literature survey.* Univ. Wisconsin TR-1648.

---

## Document conventions

- **Targets** (`T1`, `T2`, …) refer to the table in §3.3; cite by number.
- **Section 5 avenues** (`5.1` …) are independent; they do not block each other.
- This document is read by future agents; if you change §0 (boundary policy), spawn a `code-reviewer` audit on the existing Lean proofs to confirm nothing was load-bearing on the old boundary.
- Anything in §6 that ships should be back-referenced here from its PR description so this doc stays the alignment surface.
