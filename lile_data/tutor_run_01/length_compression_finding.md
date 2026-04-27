# Length-compression finding: tutor_run_01 → ~8% shorter CoT, answers preserved

**Date:** 2026-04-18
**Status (2026-04-27):** **REFUTED at n=500.** Does not replicate. Retained as a methodology cautionary tale — see addendum at the bottom of this doc.
**Source:** mini-GSM8K A/B (n=20). See `mini_gsm8k_ab_summary.md`.

## Observation

Trained `tutor_run_01_40steps` produces shorter responses than cold `cold-qwen3.5-9b-20260418` on 11 of 20 mini-GSM8K prompts:

- Mean response length: cold 337.8 chars vs trained 311.1 chars (-7.9%)
- On the 11 prompts where they differ, the *answers* (extracted numerics) are byte-identical.
- On the 9 where response lengths are identical, answers are also identical.
- Extracted final answers: 20/20 identical across both runs.

## Interpretation

After 40 mixed-domain tutor SFT steps, the LoRA adapter is measurably compressing chain-of-thought without disturbing final-answer correctness on ceiling-bound grade-school arithmetic. This is *not* a capability signal — base Qwen3.5-9B already solves these prompts. It is a **style/conciseness signal**: the tutor's demonstrations (observed to be structured, mathematical, multi-step with LaTeX) are nudging generations toward a denser format.

This is a real forward-pass delta, just not on decision-critical tokens.

## Why log it

- The mini-GSM8K A/B read as "null" on accuracy but "non-null" on generation length. Preserving this before drift prevents a later "where did that 8% come from" archeology cycle.
- Length-compression without accuracy loss is itself a research artifact. If it persists at larger training scale on in-dist tasks, that is a finding.
- Gives the next training run a shape to diff against: after N more steps, does compression deepen, flatten, or reverse?

## What to measure next (cheap)

- Re-run the same 20 mini-GSM8K prompts on `tutor_run_01_pre_cold_44` (cursor=44) vs `tutor_run_01_40steps` (cursor≈40) to see if the additional 4 commits moved length further.
- Track mean `response_chars` per prompt across cold / 40 / 44 / future snapshots.
- At each A/B harness run (leg 1 filter-to-misses, leg 2 MATH-500) log mean response length alongside accuracy.

---

## 2026-04-27 addendum — refuted at n=500

The n=20 read does not survive scaling. Two independent gates kill it:

1. **n=500 paired re-run** (`filter_to_misses_full_det_summary.md`): mean response length cold 830 → trained 824 (-0.7%), with paired-bootstrap 95% CI on the cold-solved-vs-cold-unsolved asymmetry of [-3.32, +2.02] — straddles zero. The earlier "discrimination" reading (-0.9% on solved vs -0.3% on unsolved) is resample noise, not a learned policy.

2. **Same-snapshot deterministic noise floor** (`trained_500_det_run2.json`, pinned 2026-04-27): two byte-identical-weight runs of the same snapshot under full determinism produced **55 / 500 (11%) response-length flips**, with len-diff range −358 to +409 chars and mean −4.25. The trained-vs-cold mean delta of −6 chars is two orders of magnitude smaller than the per-sample run-to-run variation. Anything in this range is decode-time noise, not a weights effect.

**What the n=20 result actually was:** an A/B at the small-n end of the noise distribution where 11/20 sample-level len differences happened to lean negative. A single byte-identical re-run would have killed it; we ran it on n=20 only and on a single seed only. Promoting to "finding" before applying gates 1-3 is the failure mode this doc now anchors as a cautionary example.

**What it would have taken to be a real finding:** what the gate doc (`lile/docs/eval-methodology-gate.md`) now requires from the start — paired full-n with bootstrap CI excluding zero asymmetry, plus byte-identical double-run confirming the delta is a weights effect rather than a decode artifact. Both gates fail here.

**Operational consequence:** do not memorialize "trained policy compresses CoT" anywhere downstream. The base model + decoder produces ±10% len variation under fixed weights; any sub-10% mean delta from a training delta is unsigned without much larger n and per-sample paired CIs.
