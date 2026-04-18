# Filter-to-misses A/B — full-n (net delta) summary

- Cold snapshot: `cold-qwen3.5-9b-20260418`
- Trained snapshot: `tutor_run_01_pre_cold_44`
- n (cold): 500, n (trained): 500, shared ids: 500

## Pre-#38 (extractor at query time)
- Cold correct: 411/500 = 82.2%
- Trained correct: 413/500 = 82.6%
- Recoveries (cold miss → trained correct): 13
- Regressions (cold correct → trained miss): 11
- **Net delta: +2 = +0.40pp**

## Post-#38 rescore (answers_match rtol=1e-3)
- Cold correct: 411/500 = 82.2%
- Trained correct: 413/500 = 82.6%
- Recoveries: 13
- Regressions: 11
- **Net delta (post-#38): +2 = +0.40pp**

## Length-compression (observation, not finding — see asymmetry CI below)
- Overall: cold 830 → trained 824 (-0.7%)
- On cold-solved: cold 765 → trained 759 (-0.9%)
- On cold-unsolved: cold 1131 → trained 1127 (-0.3%)

### Paired-bootstrap asymmetry (solved% − unsolved%, B=2000, seed=42)
- Observed asymmetry: -0.58pp (solved compresses 0.58pp more/less than unsolved)
- 95% CI: [-3.32, +2.02]
- CI excludes 0: **no — asymmetry is within resample noise**

## Interpretation
- The pre-#38 net delta is the honest headline. Recoveries-only overstate training effect.
- A post-#38 net delta different from pre-#38 means the format class is biasing the result. Equal pre/post confirms capability-only signal.
- Length-compression split is within paired-resample noise — not yet a signal. Gate-pending determinism (trained_det_run_2 byte-identity) and larger-n replication.
