# Eval methodology gate

**Author:** mei (math-validator), with claude-opus (architect)
**Date:** 2026-04-18
**Status:** draft — pins after trained_det_run_2 lands

---

## Why this exists

Three things went wrong with the tutor_run_01 A/Bs before they stabilized. Each failure mode has a cheap, mechanical fix. This doc is the checklist that every future lile A/B runs through before a number gets quoted in a finding, a PR, or a memory entry.

The canonical cautionary tale is the filter-to-misses result: an interim recoveries-only read said **+1.6pp**; a paired full-n read said **+0.40pp**; a McNemar exact test on the 13/11 discordant pairs said **p=0.839 — statistically null**. Same data, three wildly different stories, depending on which gate was in place.

## The four gates

### 1. Cold-baseline pairing

**Rule:** every snapshot eval is paired with a matched cold-model run. No single-snapshot numbers quoted anywhere.

**Why:** a standalone snapshot accuracy is uninterpretable without the cold ceiling. Qwen3.5-9B already solves most grade-school arithmetic at ~82%; a trained 82.6% reads as "+0.6pp win" or "within ceiling noise" depending on whether you saw the 82.2% cold run next to it.

**How to apply:**
- A reusable cold baseline per model: `cold-qwen3.5-9b-20260418` is the one for the 9B path.
- Every eval script takes both a `--snapshot` and a `--baseline` (or equivalent). Scripts that don't are non-compliant.
- Call-sites: `lile/teach/eval.py`, the trajectory_cursor harness, `filter_to_misses` A/B rig.

### 2. Regression-on-solves / full-n check

**Rule:** delta = recoveries − regressions, not just recoveries. Report both numbers.

**Why:** filtering to cold-misses and counting "how many did trained solve" is a one-sided read. It omits the symmetric harm: prompts the cold model got right that trained now gets wrong. Without that column, a `+13 recoveries / -11 regressions` lands as `+1.6pp win` instead of `+0.40pp, noise-adjacent`.

**How to apply:**
- Eval scripts that bucket by cold-outcome must re-run the trained model on **both** buckets (cold-solved AND cold-unsolved), not just the misses.
- Report: `n_cold_correct`, `n_trained_correct`, `recoveries`, `regressions`, `net_delta`, `net_delta_pp`.
- If n ≥ 100, append McNemar's exact test (binomial on the discordant pairs) — it's 5 lines of scipy and disambiguates "+0.40pp meaningful?" from "+0.40pp noise".
- Call-sites: `filter_to_misses` rig (now compliant as of `filter_to_misses_full_det_summary.md`). Same pattern applies to any future "filter-to-X" eval.

### 3. Noise-floor determinism gate

**Rule:** before quoting a delta, run the same snapshot twice under full determinism. Any delta smaller than the same-snapshot double-run flip count is below noise floor.

**Why:** sampling non-determinism (temperature, top_p, CUDA reductions, kernel selection) produces class-flips across runs on the same weights. On GSM8K at n=500, the double-run flip count is the noise floor; a cold-vs-trained delta below that number is unsigned.

**How to apply:**
- `do_sample=False` in the generation config.
- `torch.use_deterministic_algorithms(True)`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment.
- Pinned seed.
- Run the same snapshot twice. Diff the response texts byte-for-byte. Count class-flips on the extracted-answer column.
- **Byte-identical across runs ⟹ noise floor = 0.** Any non-zero delta is signal.
- **Non-byte-identical ⟹ noise floor = flip_count.** Deltas must exceed this to be quoted.
- Call-sites: every A/B harness. Write the double-run result to `<eval_name>_det_run_2.json` next to the primary.

#### The canonical pin (pending)

`trained_det_run_2` is in flight at the time of drafting. Expected outcomes:

- **Best case:** byte-identical to `trained_500_det.json`. Noise floor = 0. The filter-to-misses +0.40pp is then a signed delta, and McNemar's p=0.839 is the statistical read (null).
- **Likely case:** a handful of class-flips (historical anchor: 24 flips on a prior Qwen3 run). Noise floor > |+2| net delta. The +0.40pp drops below noise floor and is **unsigned**.

*Pin the actual flip count here once trained_det_run_2 lands. — placeholder, ~15min of rework.*

### 4. Writeup protocol — regime labels mandatory

**Rule:** every finding states which regime-labels it holds under. No bare accuracy numbers without an (n, model, decode_mode, cold_baseline, noise_floor) tuple.

**Why:** "trained is better than cold" is not a finding. "Trained `tutor_run_01_pre_cold_44` vs cold `cold-qwen3.5-9b-20260418` on filter-to-misses GSM8K, n=500, do_sample=False, noise_floor=<K>, McNemar p=0.839, net delta +0.40pp — statistically null" is a finding.

**How to apply:**
- Summary docs (e.g. `filter_to_misses_full_det_summary.md`) carry the tuple in the header.
- Memory entries carry the tuple or link to the summary doc.
- PR descriptions carry the tuple.
- A finding that can't survive restating with the tuple isn't a finding.

**Objective-specific mandatory labels.** When the A/B involves a composite-safe
objective (unlike + SFT-on-good), the regime tuple extends with the safe-window
quadruple per `unlike-kl-step-size-bound.md` §4–§5 (rev3) and
`unlike-trajectory-bound.md` §4–§7:

- `η_min^{emp}` — per-sample bisection floor (operational). `η_min^{lin}` logged
  alongside as compile-time sanity (up to 17× conservative, false-positive only).
- `TV_sim^{emp}` — per-step off-S TV from step-simulation at dispatch
  (operational ceiling). `η_max^{lin}` logged alongside as sanity only —
  NOT a bound (26% of steps exceed, worst 5×; false-negative side).
- `Φ_obs := Σ_i TV_sim^{emp}_i` — cumulative session drift.
- `K_session` — refuse-session threshold (default 0.27, 95th percentile of
  the random-drift prior; correlated-workload tightening is a telemetry
  follow-up).

A step with `η < η_min^{emp}` OR `TV_sim^{emp} > ε_target` is `unsafe-step` and
its accuracy numbers are quoted only alongside the violation note. A session
with `Φ_obs > K_session` is `budget-exhausted` and the same caveat applies.
A clean run is labeled `composite-safe(η_min^{emp}=…, TV_sim^{emp}=…, Φ_obs=…,
K_session=…)`. No unlike A/B without the window quadruple.

## Length-compression observation (NOT yet a finding)

The filter-to-misses full-n summary notes:
- overall: cold 830 → trained 824 (-0.7%)
- on cold-solved (n=411): 765 → 759 (-0.9%)
- on cold-unsolved (n=89): 1131 → 1127 (-0.3%)

**Why it's not yet a finding:**
- The absolute effect is 6 characters on an ~800-char baseline. Per-sample char-count variance on CoT responses is hundreds of chars; the -0.9%/-0.3% asymmetry is almost certainly inside 1 SE of itself at these sample sizes.
- The earlier n=20 mini-GSM8K read was **-7.9%**. The n=500 full read is **-0.7%**. Order-of-magnitude disagreement; the n=20 is most likely a small-sample artifact.
- Noise floor (gate 3) applies to response text, not just extracted answers. A 6-char mean difference can survive or die on kernel-level reduction order.

**What it would take to promote to finding:**
- Paired bootstrap CI on the per-sample char-count difference in cold-solved vs cold-unsolved. If the 95% CI excludes zero asymmetry, the discrimination story has support.
- Byte-identical double-run confirms the char delta is a weights effect, not a decode artifact.
- Only then: "learned policy conciseness — compress on solved, preserve on unsolved" with the full tuple.

## Status of the gates

| Gate | Current state | Gating artifact |
|---|---|---|
| 1. Cold-baseline pairing | ✅ applied to filter_to_misses full-n | `cold-qwen3.5-9b-20260418` reusable |
| 2. Regression-on-solves | ✅ applied | `filter_to_misses_full_det_summary.md` |
| 3. Noise-floor determinism | ⏳ pending `trained_det_run_2` (~60min) | `trained_500_det_run2.json` |
| 4. Regime labels | ✅ applied to new summaries | Template in this doc's header block |

## When this doc is pinned

After trained_det_run_2 lands, replace the "pending pin" placeholder in gate 3 with the actual flip count, and promote status from `draft` to `pinned`.
