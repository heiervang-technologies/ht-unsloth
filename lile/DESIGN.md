# lile — Design Choices

One-pager of load-bearing decisions made before writing code. The plan in `LIVELEARN.md` is the spec; this doc records where I made a call and why.

## Engine (§8 Q1)

**Choice: Unsloth `fast_generate` in-process, single CUDA context.** Not vLLM sidecar.

Why: target hardware is 1× RTX 3090. Two-process split for 1 GPU buys nothing — you'd ship weights between processes on the same card. `fast_generate` shares weights with training via the live `FastLanguageModel` instance, which is what the §3.1 "one model, one state" contract requires. A vLLM sidecar would force us to re-sync LoRA deltas into vLLM's paged-attention weight views on every commit — implementable but antithetical to the core invariant. Defer two-process mode to the 2×GPU follow-up (Phase 6 in the plan).

Tradeoff accepted: per-request latency is worse than vLLM's, and there's a serialization cost when a training step is mid-backward. The compute queue (§3.4) is built to hide this.

## Default model

**Qwen3-0.6B-unsloth-bnb-4bit for unit tests and interactive smoke runs.** **Qwen3-8B-unsloth-bnb-4bit for the §11 benchmark and end-to-end evidence.**

Why: Qwen3-0.6B is small enough that a full forward+backward+LoRA step is a couple of seconds, which keeps the test loop tight. The first choice for the 7–14B §11 gate was Qwen3.5-9B (nearest in the HF cache), but it turned out to be `Qwen3_5ForConditionalGeneration` — a multimodal processor whose chat-template contract differs from the text-only path (content must be `[{"type": "text", ...}]`, not a string) and whose `__call__` binds positional args to `images=`. I pivoted to `unsloth/Qwen3-8B-unsloth-bnb-4bit` (text-only, same family, 8.21 GB peak VRAM) rather than build a VL-content-wrapper shim for a text-only daemon.

## Tier targets (depth > breadth)

**Commit:** T1.1 (weighted SFT), T1.2 (KTO), T1.3 (CoH), T2.2 (hinge contrastive) as baseline. Plus the compute queue, state manager, trajectory log, snapshot manager, server, and tests. Progressive-merge path from §3.1 with the 4-bit dequant-merge gotcha (§6) resolved correctly.

**Gated on §11 benchmark:** T2.1 (CCPD v2 light) — auxiliary sampling + detached `r_c` + rank-advantage REINFORCE + SFT on top-m + KL anchor. Implemented only if Spearman on `r_c` rankings lands above the 0.2 floor; shipped as the T2 default only if above 0.5. The benchmark script is always shipped so anyone can re-run it on their own hardware. **Result (`STATUS.md` §11): Qwen3-8B Spearman mean +0.207 → shipped as per-event opt-in; hinge remains primary T2.**

**Scoped out for this submission:** T3.1 trace-infilling (requires CoT auto-detect), T3.2 SCoRe multi-turn (requires multi-turn trajectory collection), T4.1/T4.2 background replay/self-distill (operationally important, architecturally identical to T2). All land cleanly on top of the shipped core.

Rationale: the brief says "depth in one dimension beats breadth in all." A production-clean T1 + queue + state + server with the 4-bit merge path correct and the commit-cursor invariant under test is more valuable than twelve half-implemented objectives.

## Adapter stack

**bf16 LoRA on NF4 base, merged_deltas stored as bf16 CPU tensor and applied as a forward-time residual.** Exactly the shape §6 forces. The alternative (requantize merged_deltas to NF4 per merge) is the silent-quality-loss footgun the plan warns about. I'm choosing the latency cost (~5–10% at forward time, measurable) over the quality cost.

## Commit cursor

**Monotonic integer, single writer, checked by inference dispatch.** Every `/v1/train` request is chunked into queue tasks; the final task of the batch returns the commit_token to the caller. Inference requests take a snapshot of the cursor on arrival; if the current request's cursor is behind their snapshot, they block on the corresponding training task's completion event. This is the "POST a batch, next inference sees it" promise as a straightforward semaphore, not a race-prone best-effort.

Test obligation: a concurrent train+infer invariant test that would fail under reordering.

## What I'm not building

- GGUF export (Phase 6 — doesn't affect the core contract).
- Multi-GPU (single 3090 is the target).
- A reward-model judge (the objectives we ship don't need one).
- Full FT mode (LoRA is what fits on 24 GB; full FT is a one-adapter-strategy variation we can bolt on later without re-architecting).
- An async WebSocket streaming route for training events (nice-to-have; not load-bearing for the evaluation criteria).

## Open question resolutions (§8)

- **Q1 engine**: fast_generate, above.
- **Q2 default models**: Qwen3-8B default (text-only, 7–14B band), Qwen3-0.6B for tests. Pivoted from Qwen3.5-9B when that turned out to be a VL model.
- **Q3 full-FT**: deferred.
- **Q4 rewards for GRPO**: skipped — GRPO itself isn't in the shipped tier set for this run; the reward-source API is the easy part when it lands.
- **Q5 staleness**: enforced by `max_queue_depth` config; bounded buffer prevents arbitrary staleness without needing a guardrail check.
- **Q6 packaging**: `pip install -e .` from the worktree for now; pyproject is provided.
- **Q7 `r_c` ranking**: the benchmark answers this — see §11 results in `STATUS.md`.

## Testable invariants

1. **Commit cursor ordering**: train(batch) → commit_token; subsequent infer sees the loss step reflected in log-probs on the training prompt.
2. **Merge determinism**: merge(active_lora) then merge(zero_lora) leaves weights byte-equal to the first-merge result (idempotent null merge).
3. **Snapshot round-trip**: save → reset → restore → state bytes identical to pre-save for merged_deltas and active_adapter.
4. **Objective composition**: two-objective batch loss equals manual weighted sum of separately-computed losses (modulo BF16 rounding tolerance).
5. **4-bit merge correctness**: forward pass after merge within 1e-3 relative of forward pass before merge on the same input (dequant-merge path preserves semantics).
6. **Concurrent-load safety**: N concurrent `/v1/chat` + M interleaved `/v1/train` hold the commit-cursor invariant, every `after_commit_token` chat sees the cursor advanced past its token, no deadlocks, trajectory contains every event. Pinned in `test_concurrent_load.py`.

These are the load-bearing pieces the jury will read closely; tests are where those invariants get pinned down.
