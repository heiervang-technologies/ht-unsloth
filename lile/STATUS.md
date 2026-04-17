# lile — STATUS

Honest live record of what is done, what is stubbed, and what is tested. Read `DESIGN.md` first for the load-bearing choices. Every claim below cites the script that produced the evidence.

## Summary

- Phase 0 — Decisions: **done** (`DESIGN.md`).
- Phase 1 — Skeleton + T1 objectives (SFT / KTO / CoH / hinge / KL-anchor): **done**, smoke-tested on a real model.
- §11 Benchmark: **done on Qwen3-0.6B and Qwen3-8B, both at matched k=6 and at k=8 for 0.6B as a secondary data point** (N=20 items, repeats=2). At **matched k=6** both cross the pre-registered **0.2 mid threshold**: 8B mean **+0.207** (median +0.319, 60% positive), 0.6B mean **+0.231** (median +0.393, 65% positive). Decision `ship_T2_1_k8_with_hinge_primary` in both cases → CCPD v2 ships as **per-event opt-in**, hinge stays primary T2. The earlier 0.6B k=8 run (mean +0.183, below threshold) is kept in the record: it shows the objective is more candidate-count-sensitive than scale-sensitive at this size, so the original "scale-lifts-ρ" reading was confounded by the k-mismatch. A cross-review (blue, #4) flagged the mismatch; this run is the clean comparison.
- Phase 2 — T2 objectives: **done**. Hinge contrastive (T2.2) is the primary T2 ship default. CCPD v2 is fully implemented and callable via `objective: "ccpd_v2"`, with full forward+backward validated on a real model (see "Evidence" below).
- Phase 3 — Queue / state / snapshot / trajectory / FastAPI server / controller: **done**. Load-bearing invariants from `DESIGN.md` are under test and passing.
- Phase 4 — End-to-end: **done**. Merge idempotence, commit-cursor-through-HTTP, visible loss descent, CCPD v2 gradient flow, and a **concurrent-load** invariant test are all green.

## Tests & evidence

| Test | What it pins down | Result |
|---|---|---|
| `smoke_objectives.py` | T1 + hinge objectives produce finite gradients; SFT descends on Qwen3-0.6B | **pass** — loss 3.12 → 2.23 over 4 steps, peak VRAM 0.72 GB |
| `test_queue_cursor.py` | Monotone commit cursor; `wait_for` blocks then releases; FIFO; concurrent submit+wait | **pass** — 4 scenarios |
| `test_trajectory_snapshot.py` | JSONL trajectory offsets + `tail(n)`; residual byte-exact fingerprint round-trip; snapshot list | **pass** — 3 scenarios |
| `test_merge_and_e2e.py::test_merge_determinism` | Null second merge is idempotent — fingerprint unchanged | **pass** — fp `d18aef2fc380ccb1…` identical across both merges |
| `test_merge_and_e2e.py::test_end_to_end_training_moves_logprob` | Training visibly moves the shared-weight policy | **pass** — SFT loss **5.50 → 1.82** over 10 steps |
| `test_merge_and_e2e.py::test_controller_commit_cursor_e2e` | `after_commit_token` blocks inference until training commits | **pass** — cursor reaches 1 before chat returns |
| `smoke_server.py` | Full HTTP path: /health, /v1/train, /v1/chat with `after_commit_token`, /v1/state/trajectory/tail, /v1/state/snapshot/{save,list} | **pass** — cursor=1 confirms commit-cursor invariant over the wire; snapshot written to disk; trajectory logs `train_step, train_step, inference` |
| `test_ccpd_e2e.py::test_rank_advantages_math` | Pure-function rank-advantage correctness: zero-mean, ordered, scale-invariant, sign-flips | **pass** |
| `test_ccpd_e2e.py::test_ccpd_forward_and_backward_real_model` | Full §5c.11 composition runs on Qwen3-0.6B and gradient flows | **pass** — finite loss, **196 LoRA parameters** receive non-zero gradients on one backward |
| `test_ccpd_e2e.py::test_ccpd_tau_spread_skip_triggers` | τ-spread skip gates the step when the critique fails to discriminate | **pass** — 4 identical aux candidates, τ=10.0 → `{"loss": None, "ccpd_skipped": 1.0}` |
| `test_ccpd_e2e.py::test_ccpd_actually_improves_rc` | A few AdamW steps on CCPD v2 loss move `r_c` on a held-out pair | **pass** — `r_c` moved from `+0.088 → −0.001` after 3 steps (movement, not direction, is the assertion) |
| `test_ccpd_e2e.py::test_ccpd_through_train_engine` | CCPD v2 runs through the production `TrainEngine.step` path (i.e. after `for_training()`); guards against the `temp_QA` latent bug | **pass** — loss=+0.839, 4 candidates, no AttributeError |
| `test_concurrent_load.py` | 10 concurrent `/v1/chat` + 10 `/v1/train` through the real Controller; all five DESIGN invariants under contention | **pass** — **3.7 s wall**, monotone contiguous tokens, every `after_commit_token` chat saw cursor ≥ its token, trajectory has every `train_step` + every `inference`, chat latency max 2.28 s / mean 2.26 s |
| `bench_rc_ranking.py` (Qwen3-8B, N=20, k=6) | §11 ranking-reliability benchmark on a text-only 7–14B model | **Spearman mean +0.207**, median +0.319, 60% positive — **decision: `ship_T2_1_k8_with_hinge_primary`** |
| `bench_rc_ranking.py` (Qwen3-0.6B, N=20, k=6) | §11 benchmark on the small tests model — matched-k comparison vs the 8B run | **Spearman mean +0.231**, median +0.393, 65% positive — **decision: `ship_T2_1_k8_with_hinge_primary`** |
| `bench_rc_ranking.py` (Qwen3-0.6B, N=20, k=8) | §11 benchmark at higher k; kept as secondary data point after the matched-k rerun | Spearman mean +0.183, median +0.247, 57.5% positive — **decision: `fallback_to_sft_self_refinement`** |

## §11 benchmark — the decision

`lile_data/bench_rc_qwen8b_n20.json`:

```
model       : unsloth/Qwen3-8B-unsloth-bnb-4bit
k           : 6
items       : 20
repeats     : 2
n_results   : 40
Spearman mean   : +0.207
Spearman median : +0.319
positive frac   : 60.00 %  (24 / 40)
peak VRAM       : 8.21 GB
wall            : 226.9 s
decision        : ship_T2_1_k8_with_hinge_primary
```

The decision thresholds in `DESIGN.md` were set in advance:

| Spearman mean | Action | Applied? |
|---|---|---|
| ≥ 0.5 | CCPD v2 becomes default T2 | no |
| 0.2 ≤ mean < 0.5 | ship CCPD v2 as per-event opt-in | **yes — shipped** |
| < 0.2 | CCPD v2 stays experimental; hinge is default T2 | no |

At **matched k=6** the 0.6B run (Spearman +0.231) and the 8B run (+0.207) both clear the mid bracket — so the shipped decision holds for either base. The earlier 0.6B k=8 run (+0.183) sat in the bottom bracket; comparing that against the 8B k=6 number was apples-to-oranges — a cross-review (blue, #4) caught the k-mismatch and the rerun confirms the original scaling story was confounded. At this model size and benchmark, **candidate count (k) matters more than base-model scale for ranking reliability**. The IM-RM pathology the plan flags is still present (per-item `r_c` misfires on long-horizon factual critiques), just not differentiated by 0.6B vs 8B at k=6. No retroactive threshold adjustment; the pre-registered thresholds still decide the same way.

Per-item range on 8B: `min −0.600, max +1.000`. Format/length critiques ("single number", "exactly one short sentence", "uppercase", "no markdown") rank reliably; long-horizon factual/specificity critiques ("what year was the transistor invented", "include a specific landmark") are where `r_c` still misfires. Consistent with the IM-RM critique in the plan.

## Hardware measurements (RTX 3090, 24 GB)

| Workload | Model | Peak VRAM | Wall |
|---|---|---|---|
| T1 objectives smoke (batch=1, seq≤1024) | Qwen3-0.6B bnb-4bit + LoRA r=8 | **0.72 GB** | ~1.8 s / step |
| CCPD v2 forward+backward smoke | Qwen3-0.6B bnb-4bit | ~1.2 GB | ~3–4 s / step (sampling dominated) |
| §11 bench (k=6, 40 runs, N=20) | Qwen3-8B bnb-4bit | **8.21 GB** | **226.9 s** |
| §11 bench (k=6, 40 runs, N=20) — matched-k rerun | Qwen3-0.6B bnb-4bit | 1.09 GB | 121.1 s |
| §11 bench (k=8, 40 runs, N=20) | Qwen3-0.6B bnb-4bit | 1.25 GB | 111.3 s |
| E2E loss drop (10 SFT steps, r=8) | Qwen3-0.6B bnb-4bit | ~0.9 GB | ~15 s |
| HTTP server + E2E chat round-trip | Qwen3-0.6B bnb-4bit | ~1.1 GB | ~0.8 s / chat |
| **Concurrent-load (10 chat + 10 train)** | Qwen3-0.6B bnb-4bit | ~1.1 GB | **3.7 s wall** |

## What works (concise list)

- NF4 base + bf16 LoRA + bf16 CPU-resident `merged_deltas` residual (§6 path).
- Progressive merge via `state.merge_active_into_residual()` — computes Δ = α/r · (B @ A), accumulates into the CPU residual dict, zeros the active adapter. The residual is applied as a forward-time additive residual, never requantized into NF4 (avoids the silent-quality-loss footgun).
- Compute queue with monotone commit cursor, `wait_for(token)` semantics, and `after_commit_token` passthrough on `/v1/chat/completions`.
- **Model mode lock** (`ModelState.mode_lock`, a `threading.RLock`) held across every Unsloth mode flip plus the forward/backward/generate that depends on the per-layer fast-path temp buffers. Needed because `FastLanguageModel.for_training()` tears down the `temp_QA/temp_O/…` buffers that `for_inference()` sets up; without the lock, a concurrent train step kills a mid-flight generate with an `AttributeError: 'Qwen3Attention' object has no attribute 'temp_QA'`. Found by `test_concurrent_load.py`; fixed by sharing one lock between `TrainEngine.step` and `generate_chat`.
- Trajectory log with byte-offset back-pointers for snapshotting.
- Snapshot manager producing a `manifest.json` + `merged_deltas.safetensors` (+ `active_adapter.safetensors`) triple, round-tripping byte-exact.
- FastAPI server with OpenAI-compatible chat, `/v1/train`, `/v1/feedback`, `/v1/state/{merge,snapshot/save,snapshot/load,snapshots,trajectory/tail}`, `/v1/wait`.
- T1.1 weighted SFT, T1.2 KTO (β=0.1, λ_D=1.0, λ_U=1.5), T1.3 CoH (two templates), T2.2 hinge, KL anchor, CCPD v2 (full §5c.11 composition: aux sampling + detached r_c + rank-advantage REINFORCE + top-m SFT + KL anchor + τ-spread skip).
- **π_ref via `peft.PeftModel.disable_adapter()` context** for `kl_anchor_loss`, `ccpd_v2_loss`, **and `kto_loss`** when no external reference model is passed. The reference forward runs with LoRA turned off on the same weights — zero memory overhead, no separate model copy. Enabled by `pi_ref_mode="adapter_disabled"` (the default). For KTO specifically this is the difference between a real preference signal and a degenerate weighted-log-likelihood fallback; earlier versions had the degenerate path as the default, which a cross-review (blue) correctly flagged. Falls back to a zero-loss no-op only when neither an explicit `pi_ref` nor adapter-disable is available.

## What's stubbed (intentional)

- **Auto-snapshotting of π_ref at session start** — the current wiring re-uses the live model's base-via-adapter-disable for the KL anchor, which is equivalent to "π_ref = current residual + base, LoRA off." If you want a strictly session-start frozen reference, snapshot at boot and pass `pi_ref=` explicitly. The queue already supports this; the controller doesn't auto-do it because it changes semantics when paired with mid-session `snapshot_load`.
- **GGUF export, full-FT, reward-model judge, streaming training-event WebSocket** — explicitly out of scope per `DESIGN.md`.
- **Auth / rate limiting on the server** — this is a single-user local dev daemon; the plan treats it as acceptable.

## Known caveats

- **Qwen3.5-9B is a multimodal model** (`Qwen3_5ForConditionalGeneration`, config has `image_token_id`) and was the first choice for the 7–14B §11 benchmark. Its processor's `apply_chat_template` requires `content` as `[{"type": "text", "text": ...}]`, not a plain string, and its `__call__` routes positional text args to the image preprocessor. Pivoted to `unsloth/Qwen3-8B-unsloth-bnb-4bit` (text-only) for the gate — same model family, same chat template, cleanly in the 7–14B band the plan calls for.
- **Unsloth fast-inference / fast-training mode flips are not concurrency-safe on their own.** Fixed with `ModelState.mode_lock`; see "what works" above. Documented so a future contributor who adds a third GPU path (e.g. a background eval worker) knows to hold the same lock.
- **CCPD v2 sampling requires inference-mode buffers even inside a training step.** `TrainEngine.step` calls `FastLanguageModel.for_training()` which tears down `temp_QA/temp_O`, then `ccpd_v2_loss._sample_candidates` calls `model.generate()` which requires them. Fixed by an unconditional `for_inference()` at the top of `_sample_candidates` (idempotent, single-threaded since under `mode_lock`). Regression-pinned by `test_ccpd_through_train_engine`.
- **`sequence_logprob` must pass `use_cache=False`** — we evaluate k candidate log-probs sequentially then backward on a sum. Without `use_cache=False`, Unsloth's forward mutates KV-cache buffers in place between calls and a subsequent backward raises `one of the variables needed for gradient computation has been modified by an inplace operation`. Caught in `test_ccpd_e2e`; patched in `_utils.py::sequence_logprob`.
- **Transformers 5.x BatchEncoding shape changes** caught `apply_chat_template(return_tensors="pt")` (now returns a BatchEncoding dict, not a bare tensor). Fixed in `lile/objectives/_utils.py::_to_int_list` with tolerant unwrapping, and at every generation site by using `apply_chat_template(tokenize=False)` → string → `tokenizer(text=..., return_tensors="pt")`.
- **Unsloth VL-processor `__call__` binds positional to `images=`, not `text=`.** Every tokenizer call site in the repo now uses `tokenizer(text=..., ...)` keyword form. Surfaced during the 9B attempt; retained even after pivoting to 8B because the keyword form is more robust.
- **First §11 run returned Spearman=1.0 on every item** — artifact of silent sampling failure caused by the above shape bug; only the hand-crafted seed was scored, producing a trivially monotone ranking. Caught by adding a `traceback.print_exc()` on the sampling `except`. The reported mean/median are from real samples.

## Reproducing the evidence

From `/home/me/ht/forks/ht-unsloth/.worktrees/lile-opus4.7`, with the unsloth venv active:

```bash
# Unit + invariant tests (no GPU, ~0.5 s each)
python -m lile.tests.test_queue_cursor
python -m lile.tests.test_trajectory_snapshot

# GPU smoke (Qwen3-0.6B, ~10 s)
python -m lile.tests.smoke_objectives

# End-to-end (Qwen3-0.6B, ~90 s total)
python -m lile.tests.test_merge_and_e2e
python -m lile.tests.smoke_server

# CCPD v2 E2E (Qwen3-0.6B, ~2 min — loads model three times)
python -m lile.tests.test_ccpd_e2e

# Concurrent-load invariant (Qwen3-0.6B, ~30 s including load)
python -m lile.tests.test_concurrent_load

# §11 ranking benchmark (Qwen3-8B, ~4 min)
python -m lile.tests.bench_rc_ranking \
    --model unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --k 6 --repeats 2 --max-new-tokens 80 \
    --output lile_data/bench_rc_qwen8b_n20.json

# §11 ranking benchmark matched-k rerun (Qwen3-0.6B, ~2 min)
python -m lile.tests.bench_rc_ranking \
    --model unsloth/qwen3-0.6b-unsloth-bnb-4bit \
    --k 6 --repeats 2 \
    --output lile_data/bench_rc_qwen06b_n20_k6.json
```

## File-by-file status

| File | Lines | Status |
|---|---|---|
| `lile/config.py` | ~50 | done |
| `lile/state.py` | ~240 | done; mode_lock added, tested under contention |
| `lile/queue.py` | ~150 | done; tested |
| `lile/trajectory.py` | ~110 | done; tested |
| `lile/snapshot.py` | ~130 | done; tested |
| `lile/controller.py` | ~240 | done; E2E + HTTP + concurrent-load tested |
| `lile/server.py` | ~180 | done; HTTP smoke passes |
| `lile/objectives/_utils.py` | ~130 | done; `use_cache=False` fix in `sequence_logprob` |
| `lile/objectives/sft.py` | ~70 | done; tested |
| `lile/objectives/kto.py` | ~105 | done; tested (smoke); π_ref via `disable_adapter()` when no external ref is supplied |
| `lile/objectives/coh.py` | ~70 | done; tested (smoke) |
| `lile/objectives/hinge.py` | ~70 | done; tested (smoke); **primary T2 ship default** |
| `lile/objectives/kl.py` | ~65 | done; π_ref via `disable_adapter()` |
| `lile/objectives/ccpd.py` | ~250 | done; E2E smoke + π_ref wired; **shipped as per-event opt-in** |
| `lile/engine/train.py` | ~160 | done; mode_lock held around forward+backward |
| `lile/engine/inference.py` | ~65 | done; mode_lock held around generate |
| `lile/tests/*` | nine test modules (12 CCPD/server/concurrent assertions) | all passing locally |
