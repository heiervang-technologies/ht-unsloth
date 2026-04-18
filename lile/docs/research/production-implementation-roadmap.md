# `lile` — Production Implementation Roadmap

Status: draft. Owner: mei. Reviewers: claude-opus (PO), backend-track.
Date: 2026-04-17.

## Purpose

Translate the `[STRONG]` findings from `surveys/` into concrete, shippable PRs that hold under three constraints:

1. **Production-grade**: every candidate has a falsifiable eval gate, a memory/latency budget, and a rollback path.
2. **LLM-family-agnostic**: works across the families `lile` already supports and the ones most likely to show up next — Qwen3, Llama-3.x, DeepSeek-R1 / V3, gpt-oss, Magistral, Mistral-Small/Medium, Phi-3/4, Gemma-3.
3. **One invariant preserved**: **one model, one state.** No second model instance, no cross-process weight shipping. Everything below is either a LoRA update, a sampling/weighting change, or a single optional frozen-ref load (already in `state.py`).

The roadmap is ordered: **(A) already landed → (B) in-flight → (C) this-quarter → (D) design spikes → (E) explicit non-goals.**

---

## LLM-family compatibility matrix — what actually has to hold

Every candidate below must pass this checklist before entering B/C. None of the research findings are useful if they quietly hard-code Qwen3 assumptions.

| Axis | Qwen3 | Llama-3.x | DeepSeek-R1/V3 | gpt-oss | Magistral | Mistral | Phi-4 | Gemma-3 |
|---|---|---|---|---|---|---|---|---|
| Unsloth 4-bit LoRA | yes | yes | yes | yes (new) | yes | yes | yes | yes |
| Chat template via tokenizer | yes | yes | yes | yes | yes | yes | yes | yes |
| `<think>` reasoning block | yes | no | yes | yes | yes | no | no | no |
| `matmul_lora` monkey-patch fires | yes | **verify** | **verify** | **verify** | **verify** | **verify** | **verify** | **verify** |
| `for_training` / `for_inference` flip safe | yes | yes | yes | yes | yes | yes | yes | yes |
| LoRA target modules auto-discovery | yes | yes | yes | yes | yes | yes | yes | yes |

The three hard compatibility asks for every new PR:

- **Reasoning-family parser** (`lile/reasoning.py`): no PR may assume content-only streams. If `<think>` / chain-of-thought is present, splits must remain in `reasoning_content` not `content`. New objectives must train on whichever segment the user's feedback targets.
- **Residual-delta monkey-patch** (`lile/state.py`): confirm `_residual_delta` fires on the model's actual fast-path. For families that bypass `matmul_lora` (some Mistral/Phi forks do), add a family-specific bind site. One new test per new family: `test_residual_live_<family>.py`.
- **Chat-template safety**: no PR may hand-build prompts. Always go through `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.

A new test file `lile/tests/test_family_compat.py` (T3) is a prerequisite gate — exercises PR A's reset, PR B's param_groups, and residual attach for each family listed above, on CPU with a 1-layer mock model. No GPU required; families differ in config shape, not in CUDA path.

---

## Ordering and branch convention

Each PR below is one squash-merge into `ht` (per `feedback_squash_merge`). PR titles follow `feat(lile): <scope>` or `fix(lile): <scope>`. Every PR carries:

- **Eval gate**: reference to `eval-harness.md` framing (`n=100`, **Pass** = no task drops >10pp, **Kill** = any task drops >20pp).
- **Memory budget**: stated against measured 8.21 GB peak on Qwen3-8B LoRA-r16 bench.
- **Rollback**: flag name + default value; no PR lands enabled-by-default for more than one release cycle.

---

## (A) Already landed

### PR A — Snapshot-load resets optimizer `[SHIPPED]`
- `lile/engine/train.py::reset_optimizer`, `lile/controller.py::_handle_task` snapshot_load branch, `lile/tests/test_snapshot_optimizer_reset.py`.
- Fixes the correctness bug where `m`/`v` from pre-snapshot weight trajectory silently mis-scale post-snapshot updates.
- Family-agnostic (touches no model code).

---

## (B) In-flight — land this sprint

### PR B — Per-objective optimizer instances `[READY — gated on W1]`
- **Why**: `surveys/icml-2025.md` #2 (ConfPO) + `optimizer-sample-efficiency.md` §2 concern #2. Cross-objective `v`-corruption is the load-bearing failure mode for semantic feedback.
- **Correct implementation** (matches `optimizer-sample-efficiency.md` §3): **`self._opts: dict[str, torch.optim.AdamW]`**, one plain-32-bit instance per registered objective name. Each instance owns its own `m`/`v` cleanly. PyTorch `optimizer.state[param]` is keyed by tensor id, so multiple `param_groups` over the same LoRA params share state — only multi-instance isolates the second moment. `bnb.optim.AdamW8bit` is **not** multi-instantiable over the same tensors due to `GlobalOptimManager` singleton; plain `torch.optim.AdamW` is the shipping shape.
- **Scope**: ~80 LOC across `engine/train.py` (`_opts` dict + `_optimizer(name)` + `reset_optimizer` clears the dict), `config.py` (`ServeConfig.per_objective_optim: bool = False`, `per_objective_lr: dict[str, float] = {}`), and one regression test (`test_per_objective_optimizer.py`).
- **Eval gate**: on a mixed stream with ≥30% `nl_critique` events, PR B must shift at least two of the four harness tasks from "closer to 10pp gate" toward "cleanly passing." Kill if any task drops >20pp.
- **Rollback**: `cfg.per_objective_optim=False` default. Ship with flag `True` in CI only after one week of clean eval on 0.6B. VRAM cost is real: LoRA r=64 on a 7B+ base is ~50–200M trainable params → ~400MB–1.6GB fp32 Adam state *per objective instance*. N=5 objectives ≈ 2–8GB extra — fine on H100s, tight on consumer.
- **Regression test shape**: in addition to the `||Δθ||` ratio vs single-opt baseline from `optimizer-sample-efficiency.md` §3, pin `opt_sft.state is not opt_kto.state` as a static object-identity guard — cheapest way to catch any future "optimize back to shared state" regression.
- **Family gate**: `test_family_compat.py` must pass on all eight families listed above.

### PR G — Full-sequence KL-anchor scope `[OWNED BY PO]`
- `lile/objectives/kl.py`. Claude-opus's task — widen the KL anchor to cover the full response sequence rather than just the final token window.

### PR H — Rehearsal loop `[OWNED BY PO]`
- `lile/teach/rehearsal/` + controller hook. ~200 LOC. On harness regression detection, push a canonical small batch per skill.

---

## (C) This quarter — shippable implementations

These are ordered by expected eval impact × implementation cost.

### PR I — Intruder-dim LoRA diagnostic `[NEW]`
- **Source**: `surveys/neurips-2025-sample-efficient.md` #9 "Illusion of Equivalence" + `surveys/iclr-2025-2026.md` #12 "Spurious Forgetting".
- **Claim**: LoRA-induced forgetting is causally localized to "intruder dimensions" — novel high-rank singular vectors in adapter deltas. Scaling them down recovers pretraining behavior.
- **Implementation** (~80 LOC):
  - New module `lile/engine/intruder_diag.py`.
  - On each snapshot save + on demand via `/v1/state/intruder`, compute SVD of merged_delta per layer, flag singular vectors with singular value > `k * sigma_pretrain` (pretraining's typical singular-value tail).
  - Expose count + norm per layer as a new Studio-surfaced metric.
- **Mitigation variant** (optional, flagged): `cfg.intruder_damp_threshold=None` → `0.85`: rescale flagged vectors by α<1 during `merge_active_into_residual`.
- **Eval gate**: diagnostic ships gate-free. Mitigation lands iff it passes harness at α∈{0.7, 0.85, 0.95} without >10pp drop on any task.
- **Family gate**: pure linear algebra on merged deltas — family-agnostic.

### PR J — FLOW up-weighting for SFT/NTP `[NEW]`
- **Source**: `surveys/icml-2025.md` #1 FLOW.
- **Claim**: up-weighting samples where the **pre-trained base model** already scored well mitigates forgetting without extra compute.
- **Implementation** (~40 LOC):
  - Add `cfg.flow_weighting=False` flag. When set, `weighted_sft` and `sft` objectives compute an additional scalar weight `w_flow = softmax(-L_base / T)` per example, where `L_base` is the **base-model** loss on the example.
  - Requires the optional frozen-ref (`state.frozen_ref`, already in `state.py`). If frozen-ref is unloaded, the flag is a no-op with a one-time warning.
- **Eval gate**: on the `mixed_500.jsonl` stream, PR J + PR B baseline must show a stronger preservation of any task where the baseline AdamW8bit shows measurable drift. Kill if FLOW causes any task to drop >20pp.
- **Family gate**: works anywhere a frozen-ref is loadable, which is every family on the compat matrix.

### PR K — EOS / length regularization on preference objectives `[NEW]`
- **Source**: `surveys/colm-2025.md` #3 REFA.
- **Claim**: regularizing EOS token probability in DPO/KTO/hinge eliminates the length-hacking URSLA shortcut.
- **Implementation** (~25 LOC):
  - `lile/objectives/hinge.py` + `kto.py` + (optional) `coh.py`: add a term `λ_eos · (logp(<eos>|s) - logp(<eos>|s'))²` when both winner s and loser s' are available.
  - Default `λ_eos=0.0`. Ship with `λ_eos=0.1` in CI only after harness validation.
- **Eval gate**: measurable drop in average response length variance across feedback events without loss on any harness task. Reward hacking on length is already a known failure mode observable via manual inspection.
- **Family gate**: all families have `<eos>` (or equivalent end-of-turn token). The EOS token id is discovered via `tokenizer.eos_token_id` with a fallback to the family's assistant-end token (Qwen3: `<|im_end|>`, Llama-3: `<|eot_id|>`, etc.). One-liner per-family override in `lile/reasoning.py::FAMILY_CONFIG`.

### PR L — TTRL-style majority-vote pseudo-reward `[NEW]`
- **Source**: `surveys/neurips-2025-sample-efficient.md` #2 TTRL + `surveys/test-time-rl-2025.md`.
- **Claim**: majority-vote over N sampled rollouts serves as a pseudo-reward for RL on unlabeled prompts. +211% pass@1 AIME24 with zero labels on verifiable math.
- **Implementation** (~120 LOC, flagged):
  - New objective `lile/objectives/ttrl_mv.py`. For a prompt p with no user feedback, sample k=4 rollouts via the daemon's generate path; majority-vote over an answer-equivalence hash; treat majority answer as label; train SFT against that target.
  - Gated by `cfg.ttrl_pseudo_reward=False`. Only fires when the queue is idle AND the prompt passes a verifier filter (exact-match on math/code, otherwise skipped).
  - Verifier plugin lives at `lile/objectives/verifiers/` — default stub for math (gsm8k-style answer extraction) and code (executable-subset).
- **Eval gate**: on GSM8K held-out, TTRL pseudo-reward must not regress on any other harness task by >5pp. Kill if it does.
- **Family gate**: orthogonal to family — uses the daemon's generate path which is family-aware via `lile/reasoning.py`.

### PR M — Active preference-query selection (PILAF) `[NEW]`
- **Source**: `surveys/icml-2025.md` #6 PILAF + `surveys/colm-2025.md` #4 active dueling bandit.
- **Claim**: the 10–100 req/hr regime is the exact active-query regime where selecting *which* prompt to solicit feedback on beats uniform sampling.
- **Implementation** (~100 LOC):
  - New controller endpoint `POST /v1/feedback/solicit`. Given a recent prompt history, score prompts by `var(p, π_ref)` (disagreement between current policy and frozen-ref) and return top-k candidates.
  - Studio integration: in the Feedback tab, show a "suggested for feedback" section.
- **Eval gate**: after N=50 elicited feedback events, preserved-skill score ≥ baseline uniform-sampling's after N=75 events. Kill if no sample-efficiency gain at N=150.
- **Family gate**: requires frozen-ref; same compat surface as FLOW.

### PR N — 1%-pretraining-injection in replay streams `[NEW]`
- **Source**: `surveys/icml-2025.md` #11 Apple scaling law.
- **Claim**: injecting 1% pre-training-shaped data during fine-tuning prevents forgetting across scales.
- **Implementation** (~60 LOC):
  - `lile/teach/replay_streams/pretrain_mix.py`: a pluggable replay source that reads from a small local pretrain-shaped corpus (e.g., 100 MB of FineWeb-Edu sample) and yields one sample every ~100 live steps.
  - `cfg.pretrain_inject_ratio=0.01` default.
- **Eval gate**: measurable reduction in HellaSwag drift on 2000-event streams. Kill if any task drops >10pp.
- **Family gate**: completely data-side; family-agnostic.

### PR O — Low-confidence sample up-weighting `[NEW]`
- **Source**: `surveys/emnlp-2025.md` #7 "Low-Confidence Gold".
- **Claim**: centroid-clustering + tiny classifier that **keeps low-confidence samples** (hard, informative ones) beats typical "filter the noise" heuristics.
- **Implementation** (~80 LOC):
  - Per-event, cheap confidence proxy = policy's average logp on its own response. Low-confidence events get replay-priority 2×.
  - Not a new objective; a weight applied inside `ComputeQueue` prioritization.
- **Eval gate**: stream of 500 events with mixed feedback, PR O + baseline beats baseline-only on at least two harness tasks. Kill if queue latency regresses measurably.

---

## (D) Design spikes — not yet PRs

### Spike 1 — Hypernetwork LoRA generation (HyperLoRA / T2L / Meta-UCF)
- **Sources**: `surveys/meta-learning-llm-adaptation.md` (T2L, HyperLoRA), `surveys/iclr-2025-2026.md` #3 Meta-UCF, #4 Doc-to-LoRA.
- **Claim**: a hypernetwork conditioned on `(user_id, recent_feedback_embedding)` emits a per-session LoRA delta in one forward pass — no SGD.
- **Spike deliverable** (~200 LOC, behind `cfg.hypernet_lora=False`, offline-only):
  - `lile/teach/hypernet/`: take a small snapshot of feedback events per user, embed them, train a rank-r hypernet to reproduce the gradient-trained LoRA delta. Evaluate vs gradient path on one held-out user.
- **Gate**: if cosine similarity between hypernet-emitted and gradient-trained deltas < 0.6, kill the spike.
- **Family-impact**: hypernet architecture assumes a known LoRA target-module set; same compat matrix as PR B.
- **Why a spike, not a PR**: unproven at our exact regime (single-user daemon vs. the paper's multi-user batch). Don't bet a PR on it before the numbers land.

### Spike 2 — Streaming DataInf probe
- **Source**: `surveys/datainf-streaming-lora.md`.
- **Claim**: no published method does per-event influence estimation on streaming LoRA; `lile` could own this gap.
- **Spike deliverable** (~100 LOC): `lile/teach/datainf_probe.py` — rolling validation batch, first-order DataInf collapse, skip/priority decision per event.
- **Gate**: preserved-skill score on `mixed_500.jsonl` with probe-driven skip-decisions ≥ baseline. Kill if probe adds >30 ms per-event latency.

### Spike 3 — AdEMAMix8bit A/B
- **Source**: `surveys/optimizer-landscape-2025-2026.md`.
- **Claim**: AdEMAMix's **anti-forgetting** property (second, slower EMA) maps directly to `lile`'s catastrophic-forgetting concern. Could sit between ScheduleFree (PR D) and Muon (PR E, deferred) on the optimizer board.
- **Spike deliverable**: one-file wrapper around community `AdEMAMix8bit` + harness A/B on a 500-event stream.
- **Gate**: Pass on harness without >5% wall-clock regression vs AdamW8bit.

### Spike 4 — SEAL-style self-edit loop
- **Source**: `surveys/neurips-2025-sample-efficient.md` #1 SEAL.
- **Claim**: model emits its own training-data + hyperparams edits; outer RL rewards edits that improve evals. Knowledge-incorporation 32.7% → 47.0%.
- **Spike deliverable** (~300 LOC, opt-in): `lile/teach/self_edit.py` — after N feedback events, have the model propose a synthetic training example; if it passes a verifier, add to replay.
- **Gate**: forgetting rate across self-edits ≤ baseline rate across same number of real events.

---

## (E) Explicit non-goals

These came up in the surveys and are **not** shipping paths for `lile`:

| Candidate | Why not |
|---|---|
| **Muon / Riemannion as default** | Three 2025 papers converge on null effect for AdamW-pretrained bases. Qwen3 is AdamW-pretrained. Empirical ceiling = AdamW parity; not worth the spike budget against eval-harness gates. (Keep as deferred A/B, not roadmap.) |
| **Multi-instance AdamW8bit per objective** | `bnb.optim.AdamW8bit` is not multi-instantiable over the same tensors — `GlobalOptimManager` singleton collision. PR B uses plain `torch.optim.AdamW` per objective instead; the fp32 state cost is the documented trade-off (see PR B rollback). |
| **Single AdamW8bit with per-objective `param_groups`** | PyTorch keys `optimizer.state[param]` by tensor id; N groups over the same LoRA params share `m`/`v`. Isolates LR only, not the second moment — which is exactly the failure mode PR B is trying to fix. |
| **Sidecar inference copy (vLLM alongside `lile`)** | Breaks the "one model, one state" invariant. Phase 6 in `PLAN.md` may revisit for 2-GPU configs; not this quarter. |
| **Doc-to-LoRA / In-Place TTT as replacement** | Interesting as Spike 1, but gradient-based online LoRA is `lile`'s value proposition. Replacing it is a new product. |
| **PRM (process reward model) construction** | `surveys/test-time-rl-2025.md` Thread 4 — PRMs killed in 2025, "GRPO is secretly a PRM" proved implicit process credit assignment works. Do not build one. |
| **Full-parameter online finetuning** | LoRA is a hard requirement from the memory budget; full-FT breaks the mode-flip + residual-delta invariants. |
| **Profile-to-PEFT / personalized base merging** | Requires offline per-user merge steps that violate the single-process invariant. |

---

## Cross-cutting workstreams (not PRs — prerequisites)

These must land alongside the (C) batch:

### W1 — Family-compat test file (`test_family_compat.py`)
- CPU-only smoke: confirm PR A reset, PR B param_groups, and residual-delta attach point work for each family in the compat matrix above. Each family gets a one-layer mock config.
- Owner: mei. ~80 LOC. Blocks: any PR in (C) that touches model-path code.

### W2 — Verifier registry
- PR L (TTRL) + future RL work need verifiers. Build `lile/objectives/verifiers/__init__.py::VERIFIERS` with a small API: `verify(prompt, candidate) -> Optional[bool | float]`. Seed with math (GSM8K-style) and code (executable-subset via restricted `exec`).
- Owner: backend-track. ~150 LOC. Blocks: PR L.

### W3 — Studio intruder-dim + influence surfacing
- PR I's diagnostic and Spike 2's influence values need a Studio tab. Stable trajectory reference via `useLossSeries`-style hooks (per `lile`-skill invariant).
- Owner: frontend. ~200 LOC. Blocks: PR I full release.

### W4 — Chat-template audit
- Grep for any hand-built prompts in `lile/engine/*`, `lile/teach/*`. Replace with `tokenizer.apply_chat_template`. This is a one-time cleanup that unlocks gpt-oss + Magistral support.
- Owner: any. ~30 LOC + tests. Blocks: none — pure improvement.

---

## Quarter milestones

| Week | Target |
|---|---|
| W+0 | PR A shipped (done). W1 (test_family_compat.py) lands. |
| W+1 | PR B behind flag, CI green on 0.6B across all families. PR G + PR H land (PO track). W4 chat-template audit done. |
| W+2 | PR I (intruder diagnostic, no mitigation) ships. W2 verifier registry lands. |
| W+3 | PR J (FLOW) + PR K (EOS reg) behind flags; one of PR L (TTRL) / PR M (PILAF) lands. |
| W+4 | Spike 1 (hypernet) OR Spike 2 (DataInf) evaluated and either promoted to (C) or killed. PR N (pretrain-inject), PR O (low-conf) land. |
| W+5 | Full family compat matrix exercised on a live integration test with one real model per family (4-bit, Qwen3-0.6B / Llama-3-1B / Phi-3.5-mini / Gemma-3-1B). |
| W+6 | Spike 3 (AdEMAMix) + Spike 4 (SEAL) evaluated. Write quarter post-mortem. |

## Success criteria

- Every (C) PR ships behind a default-off flag with a harness gate and a documented rollback.
- `test_family_compat.py` green across all eight families on CPU.
- Zero regression on `test_concurrent_load.py`, `test_residual_live_path.py`, `test_merge_and_e2e.py` (the load-bearing invariant tests).
- One `[STRONG]` finding per survey section is either a shipped PR or an explicitly-killed spike by quarter end — no items allowed to rot.

## Open questions

1. **Hypernet-LoRA vs. gradient LoRA**: if Spike 1 shows cosine > 0.8, the product pivot is real. Who owns that call — PO, or product?
2. **TTRL without a verifier**: math and code verifiers are cheap; general domains are not. Is the daemon willing to ship a "no pseudo-reward when verifier unavailable" fallback, or do we build a cheap LLM-judge verifier?
3. **User-defined objectives**: nothing in this roadmap enables users to register their own objective. Should there be a plugin API?

---

## References

All surveys in `lile/docs/research/surveys/`. Plus:
- `lile/docs/research/sample-efficiency-lit-review.md`
- `lile/docs/research/sample-efficiency-synthesis.md`
- `lile/docs/research/optimizer-sample-efficiency.md`
- `lile/docs/research/eval-harness.md`
- `lile/DESIGN.md`, `lile/STATUS.md`, `lile/GLOSSARY.md`, `lile/PLAN.md`
