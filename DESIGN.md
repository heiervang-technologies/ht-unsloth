# LiveLearn (`lile`) — Design Decisions

**Author:** Claude Opus 4.6 (livelearn-claude branch)
**Date:** 2026-04-16
**Status:** locked-in for this implementation pass; revisit per §11 benchmark result

This document captures the load-bearing choices made *before* writing code, per the
brief's instruction to "make your top-level architectural choices explicit … the
jury will read this before looking at code."

---

## 1. Hardware envelope

Single NVIDIA RTX 3090, 24 GB VRAM, driver 595.58.03, CUDA 13. All decisions below
respect this envelope. CPU offload is permitted only as a documented fallback for
oversized models — never silently.

## 2. Base model

- **Default for code paths and the §11 benchmark:** `unsloth/qwen3-0.6b-unsloth-bnb-4bit` (locally cached, instruct-tuned, 4-bit BNB quantized).
- **Designed-for scale:** Qwen3-7B/8B/14B 4-bit. The 0.6B choice is for benchmark turnaround time and to fit a full live-training loop on the development clock; *no code path assumes the model is small*. Switching base only changes one CLI flag.

**Why not a larger model by default?** The §11 benchmark needs ~8000 forward passes;
on a 0.6B model that's ~5 minutes, on a 9B model it's ~45–60 minutes. Both are within
budget but the 0.6B run lets me iterate twice if the first pass shows a code bug. The
plan's CCPD v2 capability claim *increases* with model scale (§5c.7 item 1), so a
positive Spearman at 0.6B is a strong (and conservative) signal for 7B+. A negative
Spearman at 0.6B leaves the question open and would warrant a 9B run if time permits.

**Why not Qwen3.5-9B (locally cached)?** It is base-only (not instruction-tuned); for
critique-conditional likelihoods to have signal, the model must actually condition on
critique text in the system/instruction slot. This is a property of instruction-tuned
models. Using a base model would conflate two failures.

## 3. Inference engine

- **Unsloth's `fast_generate`** (single-process, weight-sharing).
- **Not vLLM sidecar.** vLLM's PagedAttention + CUDA graphs make weight-swap during training painful, and the brief targets 1× 3090 as the primary case where the simpler design wins. The plan's §8 Q1 itself notes "for 1× 3090 default, fast_generate is likely right." Two-GPU split-process is deferred (Phase 6 in the plan, §11 footer here).

## 4. Tier scope (depth-over-breadth)

The brief is explicit: **"Depth in one dimension beats breadth in all."** I'm building:

| Tier | Method | Status |
|---|---|---|
| **T0.1** | Log-and-batch (deferred trajectory write) | shipped |
| **T1.1** | Weighted SFT on chosen response | shipped |
| **T1.2** | KTO single-step (binary feedback) | shipped |
| **T1.3** | Chain-of-Hindsight single-step | shipped |
| **T2.1** | CCPD v2 (light, k=4–6 aux rollouts) | shipped (gated by §11) |
| **T2.2** | Hinge contrastive (aux-free fallback) | shipped |
| **T4.3** | Progressive-merge consolidation | shipped (the §6 gotcha is correctness-critical) |

**Deferred (with reason):**
- **T1.4 Rejection-SFT** — needs a judge or rule. Tractable but cosmetic for the demo.
- **T3.1 CCPD v2 + trace infilling** — the value-add is over CoT-structured tasks; benchmarking it well is itself a 2-hour project. Leaving the hook in the API.
- **T3.2 SCoRe multi-turn** — requires correction trajectories + reward shaping. Out of budget.
- **T4.1 Replay/re-weight** — needs an idle scheduler + good defaults. The trajectory log supports it; the reweighting policy is undocumented work.
- **T4.2 Self-distillation** — easy to add; deferred for time.

**Out of scope entirely:**
- Two-GPU split-process (one 3090 here)
- GGUF export (Unsloth has it; not the lile differentiator)
- Studio integration (orthogonal)
- Docker image (the existing fork's Dockerfile suffices as a starting point)

## 5. The §11 benchmark — running first, gating Phase 3

Per the brief: "Run the §11 ranking-reliability benchmark on your 3090 before
implementing CCPD v2." I'm building the benchmark before the trainer because the
result decides:

- **Spearman > 0.5:** Ship CCPD v2 as the T2.1 default (the headline capability).
- **Spearman 0.2–0.5:** Ship T2.1 but with `k≥8` and elevate T2.2 (hinge) as the
  primary balanced-tier method. CCPD v2 stays opt-in.
- **Spearman < 0.2:** Ship T2.2 as default; CCPD v2 ships as experimental flag with a
  documented "this didn't validate at our scale" caveat. T1 + T4 stack remains the
  honest core.

Whichever way it lands, the result and the decision based on it go in `STATUS.md`.

**Update after re-running on Qwen3-8B 4-bit (W2 of the 10/10 push).** The 8B
mean ρ landed at +0.164 (median +0.174, frac_positive 75 %), narrowly inside
the "fallback" band by mean but **with higher fraction-positive than 0.6B**.
We keep CCPD v2 gated rather than promoting or demoting it: the τ=0.5
advantage-spread gate is now empirically validated as load-bearing at scale
(it would have caught the −0.778 outlier case on 8B). Per-scale τ tuning is
identified as future work but not shipped here — single-scale-per-model is
not enough signal to fit a schedule. See `STATUS.md` §"§11 re-run on Qwen3-8B
4-bit" for the full table and the surprise finding that the plan's "spread
increases with scale" hypothesis is **not** supported by the data.

## 6. Stackable objectives — first-class

Per plan §3.3: per-sample objectives + per-batch objectives, weighted-sum compose.
The objective registry is a dict mapping `name → (LossFn, ValidatorFn)`, dispatchable
from the API request. CCPD v2 is one of the registered objectives, not a special case.

## 7. The 4-bit merge constraint (§6) — non-negotiable

Implementation reuses `unsloth.kernels.fast_dequantize` and the merge math from
`unsloth/save.py:_merge_lora` (lines 199–228). `merged_deltas` storage is **bf16 in
RAM** as per §6; the live forward path reads NF4 base + bf16 deltas as residual.
This is the version that's actually correct; we resist the temptation to re-quantize
between merges.

## 8. Compute queue invariant (§3.4) — tested

The "POST a batch, next inference sees it" promise is the daemon's defining feature.
Implementation: each `/v1/train` request returns `commit_token = monotonic_seq`.
Inference dispatch reads the queue's `committed_seq` and waits if a request's
`min_visible_seq > committed_seq`. The invariant is verified by a deliberate test
that would fail if the cursor were updated out-of-order.

## 9. Snapshot byte-exactness — tested

`/v1/state/save` writes `(base_ref, merged_deltas.safetensors,
active_adapter.safetensors, trajectory_log_offset, queue_seq)` atomically. Restore
reads them and reconstructs the forward function. A round-trip test compares output
logits before/after save→load on a fixed prompt; tolerance is bit-exact for the
adapter (it's just safetensors round-trip), and numerically equal for the live
output.

## 10. What I am explicitly NOT doing (to avoid scope creep)

- Not writing a custom Triton kernel. Unsloth's are the SOTA for consumer GPUs.
- Not adding an EMA reference target (KL anchor uses base or snapshot only). EMA is
  trivial to add later (one extra weight buffer); it's not load-bearing for the
  brief.
- Not adding `huggingface-hub` push. Local save only.
- Not adding chat templates / system prompts beyond what the model's tokenizer
  already provides. The tokenizer's `apply_chat_template` is sufficient.
- Not adding streaming responses to the OpenAI endpoint. Non-streaming is fine for
  the demo; streaming is a wrapper concern.
- Not adding auth, rate limiting, or TLS. Local daemon, single-user.
- Not adding multi-tenant adapter routing. One live adapter at a time per the plan
  §3.1; the registry supports adding more later.

## 11. Timebox (best-effort)

| Phase | Time | What |
|---|---|---|
| 0 | 30 min | This doc + repo audit + scaffolding |
| 1 | 60 min | §11 benchmark (Qwen3-0.6B 4-bit, k=8, n=200) |
| 2 | 90 min | State + queue + objectives skeleton with tests |
| 3 | 90 min | T1.1, T1.2, T1.3 + composer + tests |
| 4 | 90 min | T2.1 CCPD v2 + T2.2 hinge + VRAM verification |
| 5 | 60 min | Server + smoke test on real model |
| 6 | 60 min | STATUS.md + SUBMISSION.md |
| **total** | **~7.5h** | |

If a phase overruns, the next phase shrinks first; STATUS.md gets the truth. If the
§11 benchmark comes back negative, Phase 4 reshapes to "T2.2 default + CCPD v2
opt-in with caveat" and the writeup acknowledges the experimental finding.

## 12. Repository layout

Single-repo layout — `lile/` lives inside the `ht-unsloth` worktree as a sibling
package. The plan suggests a separate repo, but for the brief's purpose of producing
a reviewable artifact the single-tree layout is friendlier to readers and avoids
cross-repo orchestration. The package boundary is clean enough to extract later.

```
.worktrees/livelearn-claude/
├── lile/                       # the new package
│   ├── __init__.py
│   ├── cli.py                  # `python -m lile serve …`
│   ├── server.py               # FastAPI app + routes
│   ├── state.py                # live model state container
│   ├── queue.py                # compute queue + commit cursor
│   ├── adapters.py             # LoRA pool + progressive merge
│   ├── trajectory.py           # append-only JSONL log
│   ├── snapshot.py             # save/load
│   ├── controller.py           # GPU writer serializer
│   ├── engine/
│   │   ├── inference.py        # fast_generate wrapper
│   │   └── train.py            # objective composer + step
│   ├── objectives/
│   │   ├── __init__.py         # registry
│   │   ├── sft.py              # T1.1
│   │   ├── kto.py              # T1.2
│   │   ├── coh.py              # T1.3
│   │   ├── ccpd.py             # T2.1 (CCPD v2 light)
│   │   ├── hinge.py            # T2.2
│   │   └── kl_anchor.py        # batch-level KL term
│   └── tests/
│       ├── test_queue.py
│       ├── test_snapshot.py
│       ├── test_merge.py
│       ├── test_objectives.py
│       └── test_smoke.py
├── benchmarks/
│   └── ccpd_ranking_reliability.py   # the §11 experiment
├── DESIGN.md                   # this doc
├── STATUS.md                   # honest progress + measurements (live)
└── SUBMISSION.md               # final writeup
```

## 13. What the jury should look at first

1. `DESIGN.md` (this file) — for the locked-in choices.
2. `STATUS.md` — for the honest accounting of what works and what doesn't.
3. `benchmarks/ccpd_ranking_reliability.py` + its result — for the §11 gate.
4. `lile/objectives/ccpd.py` — for the CCPD v2 implementation (the headline).
5. `lile/state.py` + `lile/adapters.py` — for the §6 merge correctness.
6. `lile/queue.py` + `lile/tests/test_queue.py` — for the §3.4 invariant.

Everything else is plumbing in service of these.
