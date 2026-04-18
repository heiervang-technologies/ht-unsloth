# lile glossary

Terms that appear in lile code comments and docs but are not standard
across the LLM finetuning literature. Cited when they first appear so
contributors can cross-reference.

---

## Razin-safe

An objective is **Razin-safe** if it cannot exhibit the likelihood-
displacement failure mode identified by Razin et al. (2024),
*"Unintentional Unalignment: Likelihood Displacement in Direct
Preference Optimization"*.

### The failure mode

Paired preference objectives (DPO, IPO, and close relatives) define a
gradient that rewards *raising the chosen log-prob relative to the
rejected*. Critically, "relative to" allows both to fall — as long as
the rejected falls further. When the chosen and rejected responses are
lexically or semantically similar, DPO can push the chosen *down* in
absolute terms while pushing the rejected down harder. The probability
mass shed by the chosen–rejected pair lands on some **third, unintended
output** that no gradient targeted. In practice the model is technically
"preferring chosen over rejected" while emitting neither.

### Why SFT-family objectives are safe

Pure SFT gradients are **likelihood-up on a concrete target string**.
No "push rejected down" term, no relative margin, nothing that can
fall. Anything the model now emits with higher probability came
directly from an example we actually wrote down.

### Quick reference per objective

**Sharpened (2026-04-18):** Razin-safety decomposes into *aggregate-safe* (no mass
leaves the target as a group) and *pointwise-safe* (no individual non-target token
gains absolute mass). SFT-family objectives are aggregate-safe but NOT pointwise-safe
— Cleo's razin-safety-sharpened.md (in `docs/research/proofs/`) characterizes
exactly which tail tokens grow: `p_j < M_p(η)` where `M_p(η) := -(1/η) log Σ_k p_k
exp(η (𝟙[k=t] − p_k))`. **The unsafe regime is at small η**, not large — this
inverts the common "smaller LR is safer" intuition. See §Safety regime in DESIGN.md.
Machine-checked as `RazinSafety.SftMassFlow.sft_mass_flow_iff` (Lean 4 / Mathlib
v4.30.0-rc1, no axioms).

| objective      | aggregate-safe | pointwise-safe | reason                                               |
|----------------|:--------------:|:--------------:|------------------------------------------------------|
| `sft`          | ✓              | ✗              | tail non-targets with `p_j < M_p(η)` grow; adversarial if a dominant non-target absorbs shrinkage |
| `weighted_sft` | ✓              | ✗              | same mechanism, weight scales η                      |
| `ntp`          | ✓              | ✗              | same                                                 |
| `coh`          | ✓              | ✗              | same, applied per hindsight token                    |
| `kto`          | ✗ (mild)       | ✗              | signed unary term — undesirable samples *are* pushed down; likelihood displacement bounded by the KL anchor but not excluded |
| `hinge`        | ✗              | ✗              | pair margin                                          |
| `cppo`         | ✗              | ✗              | multi-candidate preference                           |
| `ccpd_v2`      | ✗              | ✗              | paired contrast with margin                          |
| `unlike`       | depends        | depends        | pure push-down (no `good_token_id`) is ✗ on both axes — a concrete "down" with no "up" target accumulates displacement fast. With a positive teacher (`good_token_id`) it becomes SFT-family dominant (aggregate-safe), but inherits SFT's pointwise unsafety **and** carries a counterintuitive η trap: **at small η, the positive teacher can push `p_bad` UP** (Cleo §2/§5 — the same B theorem mechanism applied at target=good makes bad a grower when `p_bad < M_p(η)`). Do NOT default to `lr=1e-5`; use the eta_min from Cleo's A closed-form bound (when it lands) or the empirical safe-η from the calibration sweep. **Pure-unlike requires a KL anchor with `scope="target_position"` and surgery tokens excluded; the primitive enforces tiered preconditions (error / warn / warn / warn) — pass `allow_unanchored=True` to override for research use. Tier 4 emits a known-unsafe-regime warning when `effective_lr < 5e-5` (heuristic floor, upgrades to Cleo A's per-sample `eta_min` when that lands).** |

Not-Razin-safe does not mean *broken* — these objectives work and are
useful — only that they carry the theoretical risk and need safeguards
(KL anchors, reasonable pair sampling, reference-model divergence
caps). Razin-safe objectives give you a free pass on that concern.

## composite-safe

A **composite loss** (e.g. `unlike` with a positive teacher: `-log(1-p_b) + w_+ · -log(p_g)`)
is **composite-safe** at `(p, w_+, ε_target, η)` iff both per-step empirical checks pass:
`η ≥ η_min^{emp}(p, w_+)` AND `TV_sim^{emp}(p, w_+, η) ≤ ε_target`. Cumulatively,
a session is composite-safe iff `Φ_obs < K_session`. Operational surface from
`docs/research/proofs/unlike-kl-step-size-bound.md` (A rev3) and
`unlike-trajectory-bound.md` rev1:

- `η_min^{emp}(p, w_+)` — **operational floor.** 1d bisection on the `q_b ≤ p_b`
  predicate at dispatch. Below this, the SFT-on-good side pushes `p_bad` UP
  against the unlike push-down.
- `η_min^{lin}(p, w_+)` — §4 linearization. **Compile-time sanity only.** Up to
  17× conservative vs empirical, **all on the false-positive side** (refuses
  more than necessary, never permits unsafe).
- `TV_sim^{emp}(p, w_+, η)` — **operational ceiling.** §5.b one-step simulation
  of the composite Δ; compute off-S TV on the resulting q. Refuse-to-step if
  > ε_target.
- `η_max^{lin}(p, w_+, ε)` — §5.a closed form. **NOT a bound — calibration-only.**
  Sweep showed 26% of steps exceed this formula, worst-case 5× (false-negative
  side — dangerous direction). Logged, never gates.
- `Φ_obs := Σ_i TV_sim^{emp}_i` — cumulative session drift, accumulated across
  every weight-updating step (feedback, replay, tutor) between snapshot
  boundaries. Reset-on-resume matches lile's default `disable_adapter()` anchor
  reference (trajectory §6.2).
- `K_session*` — refuse-session threshold. Default **0.27** (95th percentile of
  `Φ_obs` over the random-drift prior sweep, n=2000, N=100). WARN at
  `K_warn = K_session / 2 = 0.135`. Correlated-workload calibration is a
  telemetry follow-up; the per-step `TV_sim^{emp}` ceiling is what bounds
  individual steps regardless.

Under plain SGD / AdamW the KL anchor does not gradient-pull at step one (reference
distribution is `p|_S` itself); ε is a **post-step audit**, not an in-step constraint
(A §6.1 — natural-gradient reading rejected for AdamW). At step k > 1 the anchor
does develop a restoring force, but on the **on-S conditional**, not off-S drift
(trajectory §3 side theorem) — which is why the cumulative budget uses direct
off-S `TV_sim^{emp}` accumulation rather than an anchor-discounted functional.

Composes with **Razin-safe / B**: the per-token growth predicate `p_j < M_p(η)`
(B) and the per-target empirical-safety gate `(η_min^{emp}, TV_sim^{emp})` (A)
are two readings of the same small-η displacement mechanism; both bounds are
what `lile/objectives/unlike.py`'s tiered preconditions carry at dispatch time
(Tier 4 per-step, Tier 5 cumulative).

### Why it matters for lile

lile's live-learning loop ingests small, noisy feedback batches and
applies them continuously. Any per-step likelihood displacement would
accumulate quickly with no batch-level averaging to smooth it out.
Razin-safety is therefore preferred as the *default* route for ingesting
user feedback — `nl_critique_with_rewrite` → `coh`, `rewrite` →
`weighted_sft`. Preference objectives are available (`hinge`, `cppo`,
`ccpd_v2`) but reserved for cases where the lesson genuinely is
*"this is better than that"* rather than *"this is good"*.

---

## References

- Razin, N., Malladi, S., Bhaskar, A., Chen, D., Saparov, A., Arora, S.
  (2024). *Unintentional Unalignment: Likelihood Displacement in Direct
  Preference Optimization*. NeurIPS 2024.
- Liu, H., Sferrazza, C., Abbeel, P. (2023). *Chain of Hindsight Aligns
  Language Models with Feedback*. (The Razin-safety of CoH is an
  explicit design goal — feedback is converted to a trace that is then
  SFT-trained.)
- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., Kiela, D.
  (2024). *KTO: Model Alignment as Prospect Theoretic Optimization*.
  (Unary, desirable/undesirable objective used in lile for binary
  thumbs up/down feedback.)
