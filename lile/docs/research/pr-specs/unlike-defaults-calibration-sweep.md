# unlike defaults calibration sweep

**Author:** claude-opus (live-learn-architect), Mei (push motivation), Cleo (η dimension addition)
**Date:** 2026-04-18
**Status:** proposed
**Priority:** P2 — empirical validation track; does not block primitives
**Task:** #17
**Depends on:** task #14 complete (GPU frees), optionally #20 (safety_monitor as the instrument)
**Blocks:** nothing. Produces a tuning-guide doc.

---

## Problem

`unlike.py` defaults `rank_below=5` and `prob_above=0.1` are educated guesses. Good enough for a smoke-test trigger but unvalidated against the actual household-correction use case and — per Cleo's razin-safety-sharpened.md — now known to interact with η in a non-trivial way: at small η, the positive-teacher side of unlike can push the bad token *up* via the B theorem mechanism.

A sweep gives us:
1. Empirical defaults for `rank_below` and `prob_above` keyed on the target use case.
2. Empirical safe-η range, validated against Cleo's A closed-form bound (when A lands).
3. A tuning-guide table so callers with non-default trade-offs (e.g. aggressive surgery vs. conservative) can pick an informed point.

## Design

### Model

`unsloth/qwen3-0.6b-unsloth-bnb-4bit` — matches smoke scale, fast iteration, fits on a single 3090 with margin to spare.

### Corpus

`lile/teach/rlaif/calibration_corpus.jsonl`, ~20 hand-picked prefix entries covering the household-AI correction classes:

```json
{"prefix": "The antagonist in Harry Potter is", "bad_token": "Voldemort", "good_token": "he", "tag": "proper-name-avoidance"}
{"prefix": "The capital of Australia is", "bad_token": "Sydney", "good_token": "Canberra", "tag": "factual-correction"}
{"prefix": "I really", "bad_token": "hate", "good_token": "appreciate", "tag": "tonal"}
...
```

Per-entry fields: `prefix` (str), `bad_token` (str, tokenized once), `good_token` (str, tokenized once), `tag` (one of `proper-name-avoidance`, `factual-correction`, `tonal`, `safety-adjacent`, `terminology`).

Target: **20 entries**, 4-5 per tag, hand-curated so the bad token is plausibly a top-K candidate (else nothing triggers and the cell is noise).

### Held-out neutral set

`lile/teach/rlaif/neutral_corpus.jsonl`, ~20 prefixes where the bad tokens from the calibration corpus are NOT expected to be plausible candidates. Used to measure false-fire rate — trigger should almost never fire on these.

### Grid

| dim | values | rationale |
|---|---|---|
| `eta` | `{1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3}` | spans small-unsafe regime (Cleo B) to aggressive-but-safe regime. |
| `rank_below` | `{1, 3, 5, 10, 20, ∞}` | 1=argmax-only (original), ∞=prob-only. |
| `prob_above` | `{0.01, 0.05, 0.1, 0.3, ∞}` | ∞=rank-only. Covers near-miss and dominant cases. |

Total cells: 6 × 6 × 5 = **180**.
Per cell: all 20 calibration prefixes + all 20 neutral prefixes = 40 samples.
Per sample: ~1 second on Qwen3-0.6B (forward+backward with LoRA).
Total: **~2 hours of GPU time** (serial); parallelizable across cells if GPU memory allows.

### Metrics per cell

1. **Trigger rate on calibration corpus.** Fraction of 20 calibration prefixes where `_should_trigger` returns True on the pre-step forward. Target: ≥ 0.8 (catches the intended mistakes).
2. **False-fire rate on neutral corpus.** Fraction of 20 neutral prefixes where the trigger fires. Target: < 0.1.
3. **Single-shot correction success.** Run one `unlike` step, measure whether `bad_token` is no longer argmax at the prefix. Target: ≥ 0.6 (one-shot is a hard bar).
4. **p_bad Δ after step (Mei, added 2026-04-18).** The most direct operational check: did `p_bad` actually go down? This is the empirical validation of A's bound without waiting for the Lean proof. If `Δp_bad > 0` at a given (η, rank, prob) cell with a positive teacher, that cell is inside the B-characterized unsafe regime where SFT-on-good overpowers the push-down. Expected shape: `Δp_bad < 0` universally at large η; non-zero positive-fraction emerges below the A eta_min threshold.
5. **Post-correction drift on neutral set (dual scope).** Report both:
   - **Target-position, V\\{bad, good}:** KL( post || pre ) over the anchor-protected subspace. This is what the anchor promises; should be bounded tightly.
   - **Full vocab (target position + surrounding):** KL across the full distribution. This is what household-AI users actually experience as "semantic drift from one correction." Divergence from (a) indicates anchor leak beyond its stated scope.
6. **Grower-set cardinality (from safety_monitor, #20).** `|{j : p_j < M_p(η)}|` and the intersection with top-K non-target tokens. Validates against Cleo's B predictions; should be non-zero at small η and drop as η grows (monotonicity of M_p(η) says so).
7. **Watchlist-hit rate (from safety_monitor, #20).** Using `{bad_token, good_token}` as a 2-token watchlist, measure alarm rate. Calibrates against the theoretical threshold.

**Grid design:** paired — same prefix, same bad/good, same model state; only (η, rank_below, prob_above) vary across cells. Makes Δp_bad attributable to the cell settings rather than noise from rerolled state.

### Output

`lile/docs/research/unlike-defaults-calibration.md` with:
- Per-cell table of all 6 metrics.
- Pareto-front plot (trigger-rate × (1 - false-fire-rate)) with η encoded as color.
- Recommended defaults table: conservative / balanced / aggressive tuning points.
- Cleo-A validation plot: empirical safe-η lower bound vs. A's closed-form.
- Inversion-finding pin: "at eta=1e-6 with positive teacher, the bad token's probability INCREASES in 40% of calibration prefixes, confirming Cleo's B prediction. Do not default to tiny LR on reflex."

## Defaults-change gate

Current defaults (`rank_below=5, prob_above=0.1`) change **only** if the sweep surfaces a cell that strictly dominates them on BOTH trigger-rate AND false-fire-rate at the same η. Otherwise defaults hold and the doc publishes the table as reference.

The η default in `lile/config.py` is updated to match Cleo-A's lower bound (or the empirical safe-η threshold if A hasn't landed yet). This is a separate small PR, gated on the sweep completing.

## Tests

N/A — this is a research sweep, not a unit-testable component. The sweep script itself is a smoke-runnable module (`lile/teach/rlaif/run_calibration.py`) that any contributor can re-run.

## Rollout

1. Land `lile/teach/rlaif/calibration_corpus.jsonl` and `neutral_corpus.jsonl` (hand-curated).
2. Land `lile/teach/rlaif/run_calibration.py` (sweep driver).
3. Run on free GPU (~2h).
4. Commit `lile/docs/research/unlike-defaults-calibration.md` with the results.
5. Defaults-change PR (if gate met).

## Dependencies on other work

- `safety_monitor` (#20) is the instrument for metrics 5 and 6. If it isn't landed by sweep-time, the sweep still produces metrics 1-4 — metrics 5-6 become a follow-up run.
- Cleo's A bound (in flight) provides the A-validation line on the output plot. If A hasn't landed, the sweep publishes the empirical safe-η curve standalone; A's theoretical line gets layered in as a follow-up.
- GPU must be free: sweep blocks on trained_500_det (task #14) completing.

## Success definition

Sweep is a success if it produces:
- A published defaults-recommendation table.
- A pinned inversion-finding (small-eta unsafe) with empirical support.
- Either a defaults change (with gate met) or a clear "current defaults stand, here's why" statement with data backing.
