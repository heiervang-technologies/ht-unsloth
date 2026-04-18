# lile — sample-efficiency synthesis

Status: draft. Owner: claude-opus (%4). Scope-partner: mei (%30, optimizer slice).

This doc is the seam between three inputs:

1. [`sample-efficiency-lit-review.md`](sample-efficiency-lit-review.md) — 2024–26 literature on replay, forgetting, weighting, NL critique, RLVR, influence, task-free CL.
2. [`optimizer-sample-efficiency.md`](optimizer-sample-efficiency.md) — Mei's optimizer slice (per-objective groups, snapshot-load reset, Lion8bit / ScheduleFree spikes, Muon defer).
3. [`eval-harness.md`](eval-harness.md) — the `n=100`-per-task regression harness and its skill-targeted re-teach loop.

Purpose: produce a falsifiable, prioritized action list for the daemon as of 2026-04-17, grounded in what is measurable under the harness. Not a paper summary. Not a design doc.

Seed issue: [#7 optimizer](https://github.com/heiervang-technologies/ht-unsloth/issues/7); related [#2 attention-detached MLP-only LoRA](https://github.com/heiervang-technologies/ht-unsloth/issues/2).

---

## TL;DR

The daemon's dominant failure modes, ranked by evidence strength against *our* workload (sub-1 Hz heterogeneous feedback, Qwen3 + LoRA r=16, sessions of days to weeks):

1. **Catastrophic forgetting of skills not in the recent stream.** Code map confirms there is *no* active forgetting mitigation today: KL anchor is prompt-only (see `lile/objectives/kl.py:16`), there is no EWC/Fisher, no orthogonal-subspace constraint, no replay buffer beyond idle re-teach of the last N feedback records. This is the single biggest lever and the one the eval harness was built to surface.
2. **Objective-mixing optimizer corruption** (Mei's concern #2). Gradient scale varies up to ~10× across CoH / KTO / SFT / KL-anchor, and `v` in a shared AdamW mis-scales the next step after an objective switch. Ship PR B (per-objective groups) or PR C (Lion8bit).
3. **Signal loss at the input** — per-sample weight is not surfaced in the Studio `chat-sft-card`, and the rewrite weight is hard-coded at 3.0 in `controller.py:358`. The user's "thumbs this hard" intent cannot reach the optimizer. Orthogonal to both 1 and 2.

The interventions below are explicitly gated on the harness because **we cannot detect forgetting without it**. PR A (snapshot-load optimizer reset) is the only correctness fix that ships harness-independent.

**Friday slate**: PR A (Mei), eval-harness scaffolding (me), `mixed_500.jsonl` (me, ETA Thursday). Then PR B or PR C next week, gated on baselines. Rehearsal loop lands after that.

---

## 1. Where the code map disagrees with the default assumption

Reading `lile/` against the 2024–26 continual-LoRA literature, three findings were surprising enough to be called out explicitly:

### 1a. KL anchor is prompt-only

`lile/objectives/kl.py:16` scopes the KL divergence to the *prompt* tokens only, not the response. The anchor protects the model from drifting *how it reads* prompts, but says nothing about *how it generates*. On a stream of thumbs-ups against KTO, the model is free to drift arbitrarily far on response-side distribution while the KL loss stays near zero. This is the largest under-reported forgetting vector in the codebase.

Lit review ref: DER++ (logit-replay on response tokens) is the cheapest, most widely-validated fix. The orthogonal-subspace family (O-LoRA, CLoRA, OPLoRA) is the second-cheapest and does not require a replay buffer.

**Implication.** The harness's regression threshold (>10pp drop on any task) is effectively a *response-side* test, and the current anchor does not defend that surface. Expect every optimizer A/B to show some drift on ARC-Challenge / GSM8K / HumanEval+ after a long KTO-dominant stream, regardless of the optimizer.

### 1b. Rewrite weight is hard-coded

`controller.py:358` sets `weight=3.0` for any `rewrite` / `preferred` feedback routing into `weighted_sft`. `lile/config.py:61` defaults `weight=0.0`. There is no path from Studio UI → per-sample weight → optimizer. The value 3.0 is a guess and it is not empirically grounded against this code base.

Lit review ref: iw-SFT (Qi et al., 2025) establishes that importance weights need a ratio-clip (PPO-style, default 0.2) to prevent the single over-weighted sample from hijacking a batch. Raw 3× multipliers without a clip are exactly the regime where sample-efficiency gains go negative.

**Implication.** PR B's per-objective param groups isolates the rewrite weight's effect from other objectives, but does not fix it. The fix is either (a) surface the weight in the UI with a clip, or (b) lower the default to 1.5 (the empirical sweet spot reported in iw-SFT at LoRA rank <= 32) and let users bump it up per-sample.

### 1c. `chunk_size=2` is a noise source

Default config uses 2-sample training chunks. Under AdamW `β₂=0.999`, this means `v` turns over on ~2000 chunks ≈ 4000 samples. Mei's analysis covers this from the optimizer angle (~30 effective turnovers per week); from the *variance* angle it's worse: a per-chunk gradient at `n=2` has ~√(2000/2) = 32× the batch-variance of the same signal at `n=64`. The daemon is already paying for this in elevated `grad_clip` trips.

**Implication.** Any sample-efficiency work that assumes stable `v` estimates is building on sand. PR B (per-objective groups) actually makes this *worse* per-objective, because each objective's `v` sees fewer updates. PR C (Lion8bit) is structurally immune because it has no `v`.

Not worth a dedicated PR — but when we stand up the rehearsal loop (§4), its default chunk size should be ≥ 8, not 2.

---

## 2. Three failure modes, mapped to evidence

Ranked by how well the literature plus the code map predict the failure *on our workload*, not in general.

### Mode 1 — Catastrophic forgetting of out-of-stream skills

**Mechanism.** KL anchor (prompt-only) + no replay + no orthogonal constraint + LoRA's "forgets less" inductive bias. LoRA's regularization is a *partial* shield, not a complete one — "LoRA Learns Less and Forgets Less" (Biderman et al., 2024) shows LoRA adapters lose ~30–50% less than full fine-tuning, not 100%.

**Predicted signal.** On the harness: after a 500-event stream heavy in one domain (e.g. all-code), eval on HumanEval+ stays within 5pp of baseline, eval on GSM8K drops 10–25pp. This is the catastrophic-kill threshold in `eval-harness.md`.

**Countermeasures, cheapest first:**

| Fix | LOC | Dep | Expected win on out-of-stream task |
|---|---|---|---|
| **DER++-with-KD replay on buffer items** (Hickok 2505.12512) | ~150 + reservoir state | none | Large; the most-validated mitigation in the 2024–26 lit |
| **O-LoRA orthogonal constraint** (EMNLP 2023) | ~80 + frozen prior-adapter snapshot per boundary | none | Medium; needs "task boundary" which we don't natively have |
| **CLoRA A/B orthogonality penalty** (ACL 2025) | ~30 | none | Small-medium; no-buffer, no-boundary — cleanest fit for task-free lile |
| **Extend KL anchor to response tokens** | ~15 | none | Small-medium; converts existing anchor from prompt-only to full-seq |
| **Skill-targeted rehearsal on regression** (`eval-harness.md` §re-teach) | ~200 + `lile/teach/rehearsal/*` | harness | Closes the loop; doesn't prevent forgetting, *restores* when detected |

**Recommendation.** Ship two PRs in sequence:
- **PR G (cheap preventive)**: Extend KL anchor to full sequence, behind `cfg.kl_response_scope=True` flag. ~15 LOC, harness-independent. This is the 1-day win.
- **PR H (corrective)**: Rehearsal loop from `eval-harness.md`. Depends on harness and `mixed_500.jsonl`. Lands after PR B/C.

**Defer**: DER++ and O-LoRA as research spikes after we've measured what PR G + PR H achieve. If the rehearsal loop catches forgetting quickly enough (say, nightly), the buffer cost of DER++ may not be worth it.

### Mode 2 — Objective-mixing optimizer corruption

Covered in full in `optimizer-sample-efficiency.md`. Synthesis view:

- Mei's PR B (per-objective groups) is the structural fix.
- Mei's PR C (Lion8bit) subsumes PR B if its scale-invariance holds under our harness.
- Both are gated on the harness because the failure mode is specifically visible on `nl_critique`-heavy streams, not binary-only streams.

The lit review's contribution here is: **loss-scalar normalization is cheaper than gradient surgery**. "Direction-Aware Gradient Combination" (Peng et al., 2024) and the loss-weighting work in Google's Bag-of-Tricks (2024) both converge on EMA-normalized per-objective loss coefficients as the "poor person's per-group optimizer." If PR B is too invasive, a per-objective EMA of `loss.detach().abs()` used to normalize the reported loss before backward is a ~20-LOC alternative that captures perhaps 60% of PR B's effect. Flag for Mei if PR B's plumbing proves painful.

### Mode 3 — Signal loss at the input

The UI does not expose per-sample weight, the rewrite weight is a hard-coded 3.0, and chat-sft-card posts bare `{objective: "sft"}` with no objective toggle. For a "sample-efficiency" story to hold, users have to be *able to* send high-information events; today they cannot.

The fix is a ~50-LOC Studio PR touching `studio/frontend/src/features/lile/components/train-tab/chat-sft-card.tsx` and a matching server-side acceptance of `weight` + `chunk_size` + objective on `/v1/train`. Orthogonal to everything else in this doc.

**Recommendation.** Owner: backend (%32) if they have slack after the snapshots-tab bug; else me, after the synthesis doc and `mixed_500.jsonl` land. Not research-gated; can ship any time.

---

## 3. Eval harness is the falsifiability backbone

Restating from `eval-harness.md` for this doc's readers:

- **Probe**: OpenAI-compatible `/v1/chat/completions` on port 8768.
- **Tasks** (n=100 each): HellaSwag, ARC-Easy, ARC-Challenge (`acc_norm`); GSM8K (`exact_match`); HumanEval+ (`pass@1`). Verifiable, 95% CI ±10pp on binary acc.
- **A/B protocol**: baseline eval → 500-event streamed feedback phase → re-eval on same tasks.
- **Gates** (uniform across all PRs in both this doc and Mei's):
  - **Pass**: no task drops >10pp.
  - **Catastrophic kill**: any task drops >20pp.
  - **Direction-only secondary**: mean across tasks does not drop.
- **What is NOT claimable at n=100**: "+Xpp improvement wins". We chase preservation, not SOTA. This framing drops the "is this statistically significant?" rabbit hole entirely.

The harness is offline (CI-driven), separate from the daemon. Rehearsal-loop trigger is CI-driven for the research doc per Mei's thread #6 answer.

### `mixed_500.jsonl` — the workload under test

Agreed with Mei (see her §7 resolved threads):

- **By kind**: 40% binary (KTO) · 30% nl_critique (CoH/CCPD v2) · 20% rewrite (weighted_sft w=3.0) · 10% preferred (hinge).
- **By domain**: 25% math · 25% code · 25% common-sense · 25% general/physics/factual.
- **Ordering**: shuffled, seed=42, not domain-clustered — we want realistic objective interference.
- **Prompts**: GSM8K train, HumanEval train, HellaSwag train, MMLU diverse (general).
- **Responses**: synthesized by the cold model. Feedback labels seeded from ground-truth where verifiable; NL-critiques hand-written or LLM-synthesized against a fixed rubric.

Owner: me. ETA Thursday (2026-04-23 at the latest). File lives at `lile/teach/replay_streams/mixed_500.jsonl`.

### Baseline choice

Qwen3-0.6B on CI (~15 min at n=100, 4 tasks + HumanEval+). Maintainer-triggered 8B on PRs touching `lile/engine/`, `lile/objectives/`, `lile/state.py`, `lile/controller.py` only (~30 min at n=250). Studio UI / metrics / routes / docs skip 8B. Codify via GitHub Actions path filter when CI wires up.

---

## 4. Falsifiable action list — 6-week horizon

Grouped by what can move first. All LOC figures are approximate and exclude tests. All eval gates are against the n=100 regression framing; "pass" = no >10pp drop on any task.

### Week 1 (this week, by Friday 2026-04-24)

| # | PR | Owner | LOC | Class | Gate | Notes |
|---|---|---|---|---|---|---|
| **A** | Snapshot-load optimizer reset | Mei | ~25 + 1 test | correctness | none | Harness-independent. Protects every subsequent A/B. |
| **G** | KL anchor response-scope flag | me | ~15 + test | correctness-ish | none (flag default False) | Harness-independent. Baseline measurements in Week 2 are strictly better once this lands. `cfg.kl_response_scope`. |
| **H0** | `mixed_500.jsonl` authoring | me | ~500 samples | data | none | Baseline stream for every optimizer/replay A/B. |
| **E0** | Eval harness CLI scaffolding | me | ~150 | infra | runs end-to-end at n=100 on 0.6B | `lile/teach/eval.py`; wraps `lm-eval-harness` + evalplus. |
| **P0** | Studio Snapshots tab empty-UI fix | backend (%32) | ~40 + test | prod | passes frontend unit test | In flight on `prod/snapshots-tab-empty-fix` (PR #17). |

Slotting rationale for PR G: it is ~15 LOC, flag-gated, and harness-independent. Landing it in Week 1 means Week 2's baselines are measured with response-side KL active — strictly cleaner attribution than stacking G and B in Week 2. Confirmed with Mei in her §7 cross-link.

### Week 2 (2026-04-27 → 2026-05-01)

| # | PR | Owner | LOC | Class | Gate | Notes |
|---|---|---|---|---|---|---|
| **B** | Per-objective param groups | Mei | ~80 + test | research | pass on ≥30% `nl_critique` stream | Flag `cfg.per_objective_optim`. Switches to torch.optim.AdamW. |
| **S1** | Signal-strength UI on chat-sft-card | me or backend | ~50 | prod | renders, posts weight + chunk_size + objective | Closes Mode 3 (§2). Orthogonal. |

PR B runs against baselines collected with PRs A + G in place, so any regression is attributable to per-objective grouping alone.

### Week 3+ (2026-05-04 onward, gated on §week-2 results)

| # | PR | Owner | LOC | Class | Gate | Notes |
|---|---|---|---|---|---|---|
| **C** | Lion8bit A/B | Mei | ~30 + bench | spike | pass all four tasks on 500-event stream | If passes, subsumes PR B. |
| **H** | Rehearsal loop | me | ~200 | research | restores any regressed task within 1 rehearsal pulse | Uses `lile/teach/rehearsal/<task>.jsonl` per-task canonical batches. |
| **D** | ScheduleFree-AdamW A/B | Mei | ~40 + dep | spike | pass on 2000-event stream + rehearsal stream | Compositional case with H is the motivating argument. |

### Deferred (on the board, not scheduled)

- **DER++-with-KD replay on logits** (lit 2505.12512). Decide after PR H tells us how much forgetting the rehearsal loop catches.
- **O-LoRA / CLoRA orthogonal-subspace constraint**. Decide after PR G tells us how much full-seq KL catches.
- **Muon / Riemannion A/B** (PR E in Mei's doc). Defer pending independent Qwen3 replication.
- **iw-SFT importance-weight clip** (lit: Qi et al. 2025). Decide after S1 surfaces per-sample weights in the first place.
- **DataInf influence function for active sample selection** (lit: Kwon et al., ICLR 2024). Online-tractable; revisit when the harness has a month of data to train a selection policy against.
- **Attention-detached MLP-only LoRA** ([#2](https://github.com/heiervang-technologies/ht-unsloth/issues/2)). Memory-driven, orthogonal to sample efficiency. Let backend scope when disk-bounds / prometheus work leaves room.

---

## 5. What we will NOT claim at this `n`

To keep the group honest against small-sample drift:

- **No PR** in this doc or Mei's claims "+Xpp improvement" at n=100. The bar is preservation.
- **No PR** gates on a single A/B run. Every gate needs 3 seeds; kill if variance > 5pp on any task across seeds.
- **No comparison across optimizer A/Bs stacked in one PR**. Each ships separately against a stable baseline.
- **No eval on the training stream.** The 500-event stream drives drift; the harness *is not* the stream. That is the whole point of the harness being on held-out GSM8K/HumanEval+/HellaSwag/ARC.
- **No rehearsal pulse in the baseline measurement.** Baselines run untouched; rehearsal is explicitly gated as a separate experiment per Mei's thread #6 resolution.

---

## 6. Handoffs

- **Mei (%30)** owns PR A, PR B, PR C, PR D. Her doc is the source of truth on optimizer choice. I will not touch optimizer state in any of my PRs without pinging her first.
- **Backend (%32)** owns P0 (snapshots tab), any future S1 (chat-sft-card) if they have slack, and the production-integrity shortlist in `lile/docs/backend-shortlist.md` (not yet written — they'll file it after P0 lands). They will flag me on replay/forgetting overlap if it comes up (e.g. disk-bounds rotation interacting with replay offset dedup).
- **Me (%4)** owns H0 (`mixed_500.jsonl`), E0 (eval harness CLI), PR G (KL full-seq), PR H (rehearsal loop), and this synthesis doc. I am the glue between harness and research but I do not ship optimizer or production code.

---

## References

Cross-referenced from `sample-efficiency-lit-review.md` and `optimizer-sample-efficiency.md`. Rather than duplicate the full bibliography, this doc cites those two as the source of authority for everything it builds on.

- [`sample-efficiency-lit-review.md`](sample-efficiency-lit-review.md) — 7-domain 2024–26 survey.
- [`optimizer-sample-efficiency.md`](optimizer-sample-efficiency.md) — optimizer PR list (A–E + F).
- [`eval-harness.md`](eval-harness.md) — probe, tasks, A/B protocol, rehearsal loop.
- [Issue #7 — lile: revisit optimizer choice](https://github.com/heiervang-technologies/ht-unsloth/issues/7).
- [Issue #2 — attention-detached MLP-only LoRA](https://github.com/heiervang-technologies/ht-unsloth/issues/2).
