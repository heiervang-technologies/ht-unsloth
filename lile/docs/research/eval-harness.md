# lile — offline eval harness (research-only)

Status: draft. Owner: claude-opus (%4). Blocks: Mei's optimizer research doc (optimizer-sample-efficiency.md) for falsifiable verdicts.

## Why this exists

The autoresearch bar is: any proposed change to lile must not destroy verifiable skills in {common sense reasoning, mathematics, programming} while the model is being updated online. This is a **regression check, not a benchmark** — we're proving skills are intact after streamed feedback, not claiming state-of-the-art scores. Until a reproducible harness exists, no A/B result is defensible. This doc specifies the minimum harness the research phase needs. It is explicitly **not** the production `/v1/eval` route — that is a later studio-integration concern.

**Small-sample regime.** We cannot afford full-benchmark runs per PR. At n=100 per task, the 95% CI half-width on a binary accuracy is ~10pp — so this harness detects *catastrophic* skill loss (≥10pp drop) cleanly and cannot resolve fine-grained gains. That is fine: research bar for this phase is "does not break the model", not "moves SOTA".

## Design

lile already serves OpenAI-compatible `/v1/chat/completions`. The harness treats the running daemon as the model-under-test and never loads the LoRA adapter itself. Consequences:

- Zero duplication of model loading / tokenization / generation logic
- Evaluates the model *as users hit it* (same sampler, same template, same system prompt)
- Can freeze a snapshot via `/v1/state/snapshot/save`, run eval, then `/v1/state/snapshot/load` back — so A/Bs around a feedback stream are natural
- Works with whichever backend (lile daemon on :8768, studio backend on :8888, external vLLM/llama.cpp) happens to speak the same API

## Evals

Start with the smallest set that gives signal across the three domains.

| Domain | Benchmark | Runner | Metric | Wall-clock (Qwen3-8B) |
|---|---|---|---|---|
| Common-sense | HellaSwag (val) | lm-eval-harness `local-chat-completions` | acc_norm | ~5 min |
| Common-sense | ARC-Easy + ARC-Challenge | lm-eval-harness | acc_norm | ~3 min |
| Math | GSM8K (test) | lm-eval-harness `gsm8k_cot_zeroshot` | exact_match | ~15 min |
| Programming | HumanEval+ | evalplus | pass@1 | ~8 min |

GSM8K and HumanEval+ are strictly verifiable (exact-match answer, unit tests). HellaSwag/ARC are multiple-choice — verifiable via log-likelihood ranking over fixed choices.

Deferred to production harness: MATH, MBPP, BBH, MMLU, LiveCodeBench. These add wall-clock without adding signal the smaller set doesn't.

## Invocation shape (~150 LOC target)

```
uv run python -m lile.teach.eval \
  --endpoint http://127.0.0.1:8768/v1 \
  --model unsloth/Qwen3.5-9B \
  --tasks hellaswag,arc_easy,arc_challenge,gsm8k \
  --code-tasks humaneval_plus \
  --batch-size 16 \
  --limit 250 \
  --out lile_data/evals/<run_id>.json
```

- `--limit 250` for research iteration; full eval when promoting a PR.
- `--endpoint` points at whatever is serving the OpenAI API. No lile-specific deps.
- Output: a single JSON with `{run_id, timestamp, endpoint, model, commit_cursor_before, commit_cursor_after, tasks: {...}, raw: {...}}`. `commit_cursor_*` pulled from `/health` so the result is tied to a specific training state.

## Dependencies

New extras group `lile[eval]` in `lile/pyproject.toml`:

- `lm-eval` (lm-evaluation-harness) — HellaSwag, ARC, GSM8K
- `evalplus` — HumanEval+

Both are pulled only when someone runs `uv sync --extra eval`. Not loaded by the daemon.

## A/B protocol (how Mei's optimizer PRs use this)

For any optimizer change:

1. Baseline: fresh daemon on base model, run full eval → `baseline.json`
2. Apply a canned 500-event feedback stream (`lile/teach/replay_streams/mixed_500.jsonl` — TBD) to populate state
3. Run eval again → `streamed.json`
4. Swap optimizer (per-objective param groups / Lion8bit / ScheduleFree), repeat steps 1–3
5. Compare deltas

"Win" is reframed for the small-sample regime:

- **Primary (must hold):** no task drops by >10pp from baseline after the streamed phase. That is the skills-intact floor the harness can actually resolve at n=100.
- **Secondary (nice to have):** mean across the four tasks does not drop. Direction-only, not magnitude.
- **Catastrophic failure:** any single task drops by >20pp. That is a kill criterion — the optimizer change does not ship regardless of other signal.

Cold-state parity is not sufficient — the streaming delta is the point.

## What this unlocks (skill-targeted re-teach)

If the harness can detect regression, it can drive correction. Proposed follow-on loop (out of scope for the research doc, worth capturing):

1. After every N feedback events (or on `/v1/state/snapshot/save`), run the harness.
2. For any task that regresses by >T pp, pull a canonical training batch for that skill from `lile/teach/rehearsal/<task>.jsonl` (fixed, curated, small — ~50 samples per skill).
3. Feed those back into `/v1/train` as a rehearsal pass, tagged with the triggering eval delta.
4. Re-run the harness. Confirm the regression is corrected or downgrade the current LoRA to the last clean snapshot.

This turns the harness from a gate into a closed-loop rehearsal mechanism. It is a separate PR (and a separate research direction — skill-targeted experience replay for continual LoRA), but the harness as specified here is the prerequisite. Flagged for Mei so her optimizer plan can note which optimizers compose cleanly with a rehearsal loop (AdamW with stale `m`/`v` is suspect here; per-objective param groups help; ScheduleFree is a natural fit).

## Baseline choice for CI

- **Qwen3-0.6B** for per-PR CI smoke (runs ≤15 min across all four tasks at `--limit 100`)
- **Qwen3-8B** for maintainer-triggered pre-merge validation (~30 min)
- 70B+ deferred to a separate machine

## Open questions

1. lm-eval-harness's `local-chat-completions` needs a chat template. Does the lile daemon apply its own or does it expect raw strings? (Need to verify — server.py line 123 onwards.) If daemon applies template, harness must pass messages, not a flattened prompt. This is a one-line flag.
2. evalplus OpenAI mode — does it honor `temperature=0`? Deterministic pass@1 is what we want for research iteration.
3. Snapshot-save/load round-trip for A/B: does the merged LoRA state survive cleanly? This overlaps with Mei's optimizer concern #3 (snapshot-load reset). If it does not, A/B runs are noisy — flag for Mei.

## Envisioned product surface (Studio dashboard — separate PR)

Beyond the research doc: once the harness exists as a CLI, wrap it for Studio. This is the user-visible story and why the harness is worth building carefully now.

**Capability dimensions chart.** A Studio page (`/lile/capabilities` or a tab on the existing `/lile` page) showing each dimension over time. Dimensions and their ~100-sample validation sets:

| Dimension | Validation set | Metric |
|---|---|---|
| Common-sense | HellaSwag (100) | acc_norm |
| Reasoning | ARC-Challenge (100) | acc_norm |
| Mathematics | GSM8K (100) | exact_match |
| Programming | HumanEval+ (100) | pass@1 |
| Physics | TBD — curated subset of MMLU-physics / OpenBookQA-physics (100) | acc |
| Factuality | TBD — TriviaQA (100) | exact_match |

Chart type: one line per dimension, x-axis = commit_cursor (or wall time), y-axis = score. A horizontal dashed line per dimension at the cold-model baseline so drift is visible. Hover on a point → the snapshot id + feedback events between that point and the previous run.

**Scheduling.** Two modes:

1. **Nightly** — cron at 03:00 local, runs the harness against the current live state, writes one result JSON, updates the chart. Zero user friction.
2. **Idle-triggered** — if no chat request in the last K minutes AND no train step in the last K minutes, kick off an eval pass. Honors `mode_lock` so it cannot preempt a chat or train operation. This makes the dashboard feel live without adding request-path latency.

Both gated by the same `mode_lock` the train/chat paths already use — eval is just another consumer.

**Drift alerts.** When any dimension drops >10pp from its cold baseline (or the previous eval point, configurable), surface a badge on the Studio page with a "re-teach" button. Clicking triggers the skill-targeted rehearsal loop described above. This is the UX payoff of the research bar.

**Scope note.** Building this is a studio-integration PR chain (backend route, scheduler, frontend chart, rehearsal trigger), not part of the research doc. But the harness CLI must support `--json-only` + stable schema + low startup cost so the Studio wrapper is thin.

## Not in scope for this doc

- `/v1/eval` route implementation
- Studio frontend chart
- Scheduler (nightly / idle-trigger)
- Per-feedback-event eval (a production latency problem, not a research one)
- Comparative runs against closed models

## Next steps

- [x] Land `lile/teach/eval.py` (~150 LOC) with the four tasks wired
- [x] `lile[eval]` extra in `pyproject.toml` + CI smoke on the stub path
- [x] Baselines directory seeded with `stub.json` schema anchor
- [ ] Canned replay stream `mixed_500.jsonl` spanning all 4 feedback kinds
- [ ] Document in `lile/README.md` how to run locally
- [ ] Live Qwen3-0.6B + Qwen3-9B baselines committed to `lile/docs/research/baselines/` (needs a daemon run; CI scope is stub-only)

Target: harness + baselines landed before Mei's optimizer doc finalizes, so her predictions are backed by measurable deltas rather than handwaving.
