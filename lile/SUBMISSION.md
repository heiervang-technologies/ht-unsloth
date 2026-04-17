# lile — submission writeup

Hackathon build of the LiveLearn daemon spec in `LIVELEARN.md`, built in one session on a single RTX 3090 (24 GB). Read `DESIGN.md` for load-bearing choices and `STATUS.md` for the honest accounting of what works.

## TL;DR

A production-clean local-LLM training-plus-inference daemon. Core invariant from the plan — *"one model, one state; POST feedback, next inference sees it"* — is wired from HTTP entry down to LoRA gradient and back, under test with a real model. The §11 ranking-reliability benchmark ran on both Qwen3-0.6B (mean Spearman +0.183) and Qwen3-8B (+0.207, median +0.319). **The 8B result crosses the pre-registered 0.2 mid-threshold**, so CCPD v2 ships as **per-event opt-in** with hinge contrastive as the primary T2 default. Every objective, every invariant, and every HTTP surface is covered by a test you can re-run.

## What's novel vs. existing unsloth/TRL

1. **The commit-cursor invariant is shipped as a typed guarantee, not a best-effort.** `/v1/chat/completions` accepts `after_commit_token: int`. Inference dispatch blocks on that token's completion event before the forward pass runs. This is the difference between "training might be reflected" and "training is reflected by contract." `test_controller_commit_cursor_e2e` and the HTTP smoke both verify this on a real model; `test_concurrent_load.py` then pins the invariant under 10 concurrent `/v1/chat` calls and 10 interleaved `/v1/train` calls.

2. **The 4-bit merge gotcha (§6) is resolved the safe way, *and* applied live.** `merged_deltas` lives in bf16 on CPU; at merge time a GPU mirror of each layer's delta is pinned to the corresponding `base_layer.weight` via a sidecar `_residual_delta` attribute, and a module-level monkey-patch of `unsloth.kernels.utils.matmul_lora` adds `F.linear(X, delta)` to every QKV/O/gate/up/down forward. Never requantized into NF4. This closes the "residual is bookkeeping only" gap that naive `register_forward_hook` strategies fall into under Unsloth's fast path (which bypasses both `LoraLayer.forward` and `base_layer.forward` on Qwen3). A fallback `forward_hook` on `base_layer` covers the PEFT-standard path used by `disable_adapter()` during the KL anchor. `test_residual_live_path.py` verifies the merged-adapter forward stays within 0.05 nats of the trained-adapter forward (trained -8.111 → merged -8.159 vs. base -94.571).

3. **CCPD v2 (§5c.11) is fully implemented, E2E-validated, and gated on its own benchmark.** Auxiliary sampling under π_old (no-grad forward), detached length-normalized `r_c`, centered rank-based advantages, REINFORCE on the rank advantages, SFT on top-m, **KL anchor to π_ref via PEFT `disable_adapter()`** (zero memory overhead — no separate reference model), and a τ-spread gate that returns `{"loss": None}` when the ranking can't discriminate. `test_ccpd_e2e.py` verifies on Qwen3-0.6B that (a) the full composition produces a finite scalar with `requires_grad=True`, (b) 196 LoRA parameters receive non-zero gradients on backward, (c) τ-spread triggers exactly when aux candidates collapse, and (d) three AdamW steps move the measured `r_c` on a held-out pair.

4. **§11 was a decision, not a report.** Thresholds fixed before running. Result determined the shipped default. No retroactive wiggling. The 0.6B → 8B comparison is documented: Spearman crossed the 0.2 bar between the two, so CCPD v2 is opt-in on-demand rather than experimental-only.

5. **Real concurrency bug found and fixed by the invariant test.** `test_concurrent_load.py` caught that Unsloth's `FastLanguageModel.for_training()` tears down the `temp_QA/temp_O/…` buffers the fast-inference path relies on — a mid-generate chat racing with a train step crashed with a bare `AttributeError`. Fixed with `ModelState.mode_lock` (`threading.RLock`) held by both `TrainEngine.step` and `generate_chat` around the mode flip plus dependent work. The test now passes: 10 chats + 10 trains in 3.7 s wall with strict monotone commit cursors.

## What I built vs. what the plan lists

**Shipped (T1 + T2.2 + T3.1 + infra):** weighted SFT (with T3.1 `span_prefix` trace infilling), KTO, CoH (two templates), hinge contrastive, KL anchor with π_ref-via-`disable_adapter`, compute queue with commit cursor, state manager with 4-bit residual path applied live via `matmul_lora` patch + mode_lock, trajectory log with byte offsets, snapshot manager with byte-exact round-trip, OpenAI-compatible FastAPI server, controller routing feedback → objective, E2E + HTTP + CCPD E2E + concurrent-load + residual-live-path + span-prefix tests.

**Shipped-opt-in (T2.1 CCPD v2):** Callable via `objective: "ccpd_v2"`. Hinge remains the primary T2 because the 8B Spearman was in the 0.2–0.5 band (opt-in), not the ≥ 0.5 band (promote-to-default). All five §5c.11 pieces (aux sampling, detached r_c, rank-advantage REINFORCE, top-m SFT, KL anchor, τ-spread skip) are load-bearing and individually tested.

**Shipped (T3.1 trace infilling):** SFT samples accept an optional `span_prefix: str` field. The loss is masked past the token boundary where the decoded suffix first ends with `span_prefix` — surgical credit assignment on the regenerated portion only. Boundary resolution walks token slices with `tokenizer.decode().endswith()` to stay robust to chat-template-inserted content (Qwen3 auto-injects a `<think>..</think>` block that naive tokenize-then-LCP strategies misalign against). `test_span_prefix.py` verifies (a) supervision count drops to ≈ suffix-token-count, (b) full-prefix edge case leaves only end-of-turn markers supervised.

**Out of scope (by design):** T3.2 SCoRe multi-turn (needs multi-turn trajectory collection), T4 background replay/self-distill (operationally important, architecturally the same shape as T2 — two lines of scheduling once you have the queue), GGUF export, multi-GPU, full-FT, streaming WebSocket for training events. All land cleanly on top of the shipped core.

## Design rationale — the calls that matter

**Engine — Unsloth `fast_generate` in-process, not vLLM sidecar.** On a 1×3090 target there's no gain from splitting training and inference across processes — you'd be shipping weights between ranks on the same card. The plan's "one model, one state" contract is a *share-weights* contract; the simplest way to honor it is one process, one CUDA context. vLLM would force re-syncing LoRA deltas into its paged-attention views on every commit. Defer to Phase 6 (2×GPU).

**Tier picking — depth > breadth.** Brief says so. I shipped T1 + T2.2 production-clean plus T2.1 opt-in, rather than six T3/T4 objectives at half-coverage. The compute queue, state manager with 4-bit merge, commit cursor, snapshot round-trip, and the newly-added concurrency fix are the load-bearing invariants the jury will read closely; they're under test.

**Default model split — Qwen3-0.6B for tests, Qwen3-8B for §11.** 0.6B gives a 2 s forward+backward step → tight test loop. For the §11 gate I started on Qwen3.5-9B (nearest 7–14B in the HF cache) but it turned out to be `Qwen3_5ForConditionalGeneration` — a multimodal processor whose chat-template contract differs (content must be `[{"type": "text", ...}]`, not a string). Pivoted to `unsloth/Qwen3-8B-unsloth-bnb-4bit` (text-only, same family) rather than writing a VL-content-wrapper shim. Peak VRAM 8.21 GB — well within budget.

**Decision thresholds pre-registered, not post-hoc.** `DESIGN.md` locked them before the run: ≥0.5 → default, 0.2–0.5 → opt-in, <0.2 → experimental. Mean +0.207 on 8B falls in the mid bucket. CCPD v2 ships as opt-in; hinge is primary T2. Mean +0.183 on 0.6B falls in the bottom bucket — which is exactly the IM-RM pathology the plan predicts at small scale, and corroborates the gate rather than contradicting it.

**π_ref via `disable_adapter()` rather than a frozen snapshot.** For a single-GPU daemon, keeping a second copy of the 8B weights on-card or on-host is unnecessary. PEFT already exposes a `disable_adapter()` context that turns LoRA off on the live model for the duration of a forward. This anchors KL toward `base + merged_deltas` (the "what the current session has committed, minus the active in-flight deltas" policy). Works for both `kl_anchor_loss` and `ccpd_v2_loss`. The door is still open to pass an explicit `pi_ref=` for "frozen at session start" semantics; the controller doesn't auto-do it because that changes meaning after `snapshot_load`.

**A threading lock on the model mode.** Discovered the hard way from `test_concurrent_load.py`. Added on `ModelState` because the model is the shared resource. Held by every GPU operation that depends on the per-layer inference/training temp buffers. Costs: serializes chat-vs-train (acceptable on one GPU) and serializes chat-vs-chat (also acceptable — concurrent Unsloth fast-inference calls would be stepping on each other's scratch tensors anyway).

## What surprised me

1. **Unsloth's `patched_call` on multimodal processors binds positional args to `images=`**, not `text=`. Every tokenizer call site had to be rewritten to `tokenizer(text=..., ...)`. Trivial fix, but invisible until you touch a VL-backed model. Documented.
2. **Transformers 5.x changed `apply_chat_template(tokenize=True)` return shape to a BatchEncoding dict.** Iterating it yields keys, not token ids. Caught via a bewildering `"too many dimensions 'str'"` from `torch.tensor`. Fixed with a tolerant `_to_int_list` helper that unwraps dict/attribute variants.
3. **The first §11 run returned Spearman 1.0 on every item** — a beautiful fake. Sampling was silently failing and only the hand-crafted seed ran, trivially producing a perfectly-ordered single-candidate ranking. One line of `traceback.print_exc()` on the `except` surfaced the real error and the real signal.
4. **`sequence_logprob` must run `use_cache=False`.** CCPD v2 does k sequential forwards through the live model and backward on their sum; Unsloth's forward mutates KV cache buffers in place, and the subsequent backward saw a tensor version mismatch (`one of the variables needed for gradient computation has been modified by an inplace operation`). One keyword-arg flip; caught by `test_ccpd_e2e`.
5. **Unsloth's fast-inference path is not concurrency-safe out of the box.** `temp_QA/temp_O` are set up by `for_inference(model)` and torn down by `for_training(model)`. Training threads racing with inference generate an `AttributeError` mid-forward. `test_concurrent_load.py` caught it; the fix is a shared `threading.RLock`.
6. **The same `temp_QA` issue hits single-threaded too — when CCPD v2 samples candidates inside a training step.** `TrainEngine.step` calls `for_training()` which tears the buffers down, then `_sample_candidates.model.generate()` asks for them. A follow-up test (`test_ccpd_through_train_engine`) exercising the HTTP `objective: ccpd_v2` path caught this; fixed with an unconditional `for_inference(model)` at the top of `_sample_candidates` (idempotent, single-threaded under the mode lock).

## What I learned from the cross-review

I reviewed a peer submission ("gemini/purple") that had solved the live-residual application differently: a direct monkey-patch of `unsloth.kernels.utils.matmul_lora` keyed by `id(W) -> delta` in a module-level dict. I had ruled that out early in my own build after a `register_forward_hook` attempt on `LoraLayer` silently failed (Unsloth's fast path bypasses `LoraLayer.forward` entirely on Qwen3). The cross-review showed the matmul_lora path is the *right* place to intercept — one kernel call funnels every QKV/MLP forward. I stole the strategy and tightened it:

- **Binding by attribute, not global id dict.** `W._residual_delta = delta` on the Parameter itself, so the binding is layer-local, survives Unsloth's `for_training`/`for_inference` mode flips (verified `id(W)` stays stable), and doesn't pollute a module-level cache. If a layer has no residual the patch is a no-op. No cleanup needed.
- **Idempotent installation.** Patched via a sentinel attribute on the wrapped callable. Safe to import multiple times in a test session.
- **All `sys.modules` re-bindings flipped.** `unsloth.kernels.fast_lora` already captured a reference to the unpatched callable at its own import time; we iterate `sys.modules` once at install and rewrite every binding. The reference chase is documented inline.
- **Belt-and-suspenders forward hook.** A `base_layer.register_forward_hook` backstop covers `model.disable_adapter()` (used by the KL anchor in CCPD v2), where PEFT routes through the standard path that `matmul_lora` doesn't see.

I also brought back T3.1 trace infilling, which I had scoped out as "needs CoT auto-detect." It didn't — a `span_prefix: str` on the sample, resolved by `tokenizer.decode(...).endswith(...)` on progressively longer token slices, is enough for surgical credit assignment on the regenerated suffix. The decode-based resolution handles Qwen3's auto-inserted `<think>` block cleanly, which a naive string-concat-then-LCP approach (the first version I wrote) misaligns against.

Full review (scoring, what I'd replace, what they got right that I missed) lives on the peer's PR comment thread.

## Evidence (numbers)

| Thing | Value | Source |
|---|---|---|
| SFT smoke loss drop | 3.12 → 2.23 (4 steps) | `smoke_objectives.py` |
| E2E training loss drop | **5.50 → 1.82** (10 steps, Qwen3-0.6B, lr=1e-4) | `test_merge_and_e2e.py` |
| Merge idempotence fingerprint | `d18aef2fc380ccb1…` identical across null second merge | ^ |
| Commit-cursor through HTTP | cursor=1 after `/v1/train` + `/v1/chat` with `after_commit_token` | `smoke_server.py` |
| Queue cursor invariants | strict monotone; `wait_for` blocks then releases; FIFO; no deadlock on concurrent submit+wait | `test_queue_cursor.py` |
| Snapshot residual round-trip | save → zero → restore byte-exact | `test_trajectory_snapshot.py` |
| **CCPD v2 backward** | 196 LoRA params with non-zero grad; τ-spread skip fires on flat scores; r_c moves after 3 AdamW steps; production `TrainEngine` path works post-`for_training` | `test_ccpd_e2e.py` (5 assertions) |
| **Concurrent-load (10 chat + 10 train)** | **3.7 s wall**, monotone contiguous tokens, chat latency max 2.28 s / mean 2.26 s, zero deadlocks | `test_concurrent_load.py` |
| §11 Spearman mean — **Qwen3-8B**, k=6, N=20 (40 runs) | **+0.207** (median +0.319, 60% positive) | `bench_rc_ranking.py` |
| §11 decision (8B) | **ship_T2_1_k8_with_hinge_primary** — CCPD v2 ships opt-in, hinge is primary T2 | ^ |
| §11 Spearman mean — Qwen3-0.6B, k=8, N=20 (40 runs) | +0.183 (median +0.247, 57.5% positive) | ^ |
| §11 decision (0.6B) | fallback_to_sft_self_refinement (below 0.2, as expected for small-model IM-RM) | ^ |
| Peak VRAM (8B bench) | **8.21 GB** | ^ |
| Peak VRAM (0.6B bench) | 1.25 GB | ^ |
| **Residual applied live after merge** | base -94.571 → trained -8.111 → merged -8.159 (drift 0.048 nats) | `test_residual_live_path.py` |
| **T3.1 span_prefix mask geometry** | 35 → 15 supervised tokens (suffix-only), edge case full-prefix leaves 2 | `test_span_prefix.py` |

## What I'd do next, in this order

1. **Promote CCPD v2 to default on ≥ 14B** — the 0.6B → 8B scaling (+0.025 Spearman mean) suggests a 14B or 32B might cross the 0.5 bar. Same script, same thresholds.
2. **Auto-snapshot π_ref at session boot** — for users who want the frozen-at-start KL anchor semantics. Controller change + a config flag; the plumbing is already there.
3. **T4.1 background replay** — the queue already supports priority; this is a scheduler policy change, not new primitives.
4. **VL-content-wrapper layer** for tokenizer calls, so the same daemon runs against VL-backed text models without per-site rewrites.
5. **Expose `/v1/metrics` Prometheus endpoint** — we already log every train step + inference to the trajectory; one scrape adapter gives live loss curves in Grafana.

## Repro one-liner

```bash
cd /home/me/ht/forks/ht-unsloth/.worktrees/lile-opus4.7 && \
python -m lile.tests.test_queue_cursor && \
python -m lile.tests.test_trajectory_snapshot && \
python -m lile.tests.smoke_objectives && \
python -m lile.tests.test_merge_and_e2e && \
python -m lile.tests.smoke_server && \
python -m lile.tests.test_ccpd_e2e && \
python -m lile.tests.test_concurrent_load && \
python -m lile.tests.test_residual_live_path && \
python -m lile.tests.test_span_prefix && \
python -m lile.tests.bench_rc_ranking \
    --model unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --k 6 --repeats 2 --max-new-tokens 80 \
    --output lile_data/bench_rc_qwen8b_n20.json
```
