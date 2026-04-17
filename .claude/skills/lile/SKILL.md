---
name: lile
description: Orient yourself before touching `lile/` — the LiveLearn live-training daemon. Use when editing any `lile/**` file, `studio/backend/routes/lile.py`, `studio/frontend/src/features/lile/**`, or when debugging commit-cursor / residual-delta / queue behavior. Covers the load-bearing invariants, entry points, how to add objectives / metrics backends, and the anti-patterns that have bitten past contributors.
---

# lile — LiveLearn FYI

`lile` is an online-learning daemon that wraps a single Unsloth `FastLanguageModel` with a FastAPI server. **Inference and training share the same model weights in one CUDA context.** Clients `POST /v1/train` with a batch, and the next `POST /v1/chat/completions` can block on a `commit_token` to observe the update.

This skill is the orientation pass. Deep design lives in `lile/DESIGN.md`; current test evidence lives in `lile/STATUS.md`; terminology lives in `lile/GLOSSARY.md`.

## The one invariant that must not break

> **One model, one state.** Training and inference run in the same Python process against the same `FastLanguageModel` instance. There is no weight-sync step. Never introduce a pattern that assumes a separate inference copy.

If you find yourself writing code that pickles weights, ships tensors cross-process, or maintains "the inference copy" and "the training copy", stop — you are about to break the load-bearing contract.

Corollaries:

- Training and inference serialize through `ModelState.mode_lock`. Unsloth flips per-layer temp buffers on `for_training()` / `for_inference()`; concurrent entry into either mode from different callers tears those buffers apart.
- There is no vLLM sidecar. A 2-GPU follow-up may add one (Phase 6 in `PLAN.md`); until then, do not plan around it.

## Core files — where to look first

| File | What it owns |
|---|---|
| `lile/server.py` | FastAPI routes (`/v1/chat/completions`, `/v1/train`, `/v1/feedback`, `/v1/state/*`, `/v1/wait`, `/health`). OpenAI-compatible SSE. Thin — delegates to `Controller`. |
| `lile/controller.py` | The orchestrator. `generate`, `stream_generate`, `submit_train`, `submit_feedback`, `request_merge`, snapshot save/load. Holds the queue, trajectory, and state. |
| `lile/queue.py` | `ComputeQueue` — single-worker FIFO with monotone `commit_cursor`. `wait_for(token)` is the "POST a batch, next inference sees it" guarantee. |
| `lile/state.py` | `ModelState` — the model, tokenizer, `mode_lock`, `merged_deltas` residual, and the **module-level monkey-patch on `unsloth.kernels.utils.matmul_lora`** that applies `_residual_delta` at forward time. |
| `lile/engine/train.py` | `TrainEngine.step` — composes registered objectives, runs forward+backward, updates LoRA, returns `{loss, components}`. |
| `lile/engine/inference.py` | `generate_chat` / `generate_chat_stream` — uses HF `TextIteratorStreamer`; holds `mode_lock` across the Unsloth mode flip and the entire generate. |
| `lile/engine/replay.py` | `IdleReplayScheduler` — optional background re-training from the trajectory when the queue is idle. |
| `lile/objectives/` | One file per objective. All register into `objectives/__init__.py::OBJECTIVES`. |
| `lile/reasoning.py` | Streaming `<think>…</think>` parser → splits `reasoning_content` vs `content` deltas. Family registry for Qwen3 / DeepSeek-R1 / Magistral / gpt-oss. |
| `lile/logging_backends.py` | Optional no-throw metrics sinks (W&B / TensorBoard / MLflow / trackio) — trajectory JSONL is still the canonical record. |
| `lile/trajectory.py` | Append-only JSONL at `lile_data/trajectory.jsonl`. Every `train_step`, `inference`, and `feedback` event lands here with a stable byte offset. |
| `lile/snapshot.py` | Save/load `merged_deltas` + active LoRA adapter to disk. `residual_fingerprint()` is what tests use to prove byte-equality. |

## The concepts you must hold in your head

**Residual delta.** Upstream Unsloth bypasses both `LoraLayer.forward` and `base_layer.forward` on Qwen3's fast path, so `register_forward_hook` silently never fires. We intercept the single funnel `unsloth.kernels.utils.matmul_lora` via a module-level monkey-patch installed at import of `lile/state.py`. The patch looks for `W._residual_delta` (a bf16 GPU tensor bound as an attribute of the Parameter) and adds `F.linear(X, delta)` to the output. Binding by attribute survives Unsloth's mode flips — `id(W)` stays stable across `for_training` / `for_inference`. Do not replace this with a `dict[id(W), delta]` lookup; that was tried and broke on mode flip.

**Commit cursor.** Monotone integer, single writer (the queue worker), read by inference dispatch. Every `/v1/train` request is chunked into queue tasks; the last task's token is returned as `commit_token`. Clients pass it back as `after_commit_token` on their next chat call; the controller awaits `queue.wait_for(token)` before generating. `test_concurrent_load.py` pins the invariant under contention — keep it green.

**Compute queue.** FIFO, single worker, bounded by `cfg.max_queue_depth`. Staleness is bounded by the depth cap, not by a guardrail check. Training tasks block inference only for the forward+backward window — the queue is there to hide that cost.

**Trajectory as source of truth.** `lile_data/trajectory.jsonl` is canonical. Metrics backends in `logging_backends.py` are optional *mirrors* that must never raise into the hot path — catch-and-log inside every adapter method.

**Razin-safety.** Read `GLOSSARY.md` before adding an objective. SFT-family objectives are "likelihood-up on a concrete target" and can't shift mass to unintended outputs. Preference-margin objectives (hinge, DPO-shaped, CCPD v2) are useful but need KL anchors or reference caps. Default user-feedback ingestion routes through Razin-safe objectives (`rewrite` → `weighted_sft`, `nl_critique_with_rewrite` → `coh`).

## How to contribute

### Adding a new objective

1. Create `lile/objectives/<name>.py`. Export `<name>_loss(model, batch, **kwargs) -> {"loss": Tensor, "components": dict[str, float]}`. The loss must be scalar and graph-attached; `components` is for logging only.
2. Register in `lile/objectives/__init__.py::OBJECTIVES`. If the import is optional (heavy extra deps), wrap in `try/except` like `ccpd_v2`.
3. Add to the Razin-safety table in `GLOSSARY.md`. If not safe, document the anchor / cap that makes it acceptable in-loop.
4. Write a smoke test in `lile/tests/smoke_objectives.py` (finite gradients on Qwen3-0.6B).
5. For anything beyond T1, add an end-to-end test analogous to `test_ccpd_e2e.py` — at minimum, `test_<name>_forward_and_backward_real_model` proving grads flow through the production `TrainEngine.step` path (after `for_training()`).

### Adding a metrics backend

1. Add a class to `lile/logging_backends.py` implementing the `MetricsLogger` protocol (`log_params`, `log_metrics`, `close`).
2. Catch every exception inside every method and `log.warning` — an adapter must never crash the train loop.
3. Add the backend name to `_BACKENDS` at the bottom and to the docstring list at the top.
4. Add a test in `tests/test_logging_backends.py` — import-guard so CI without the optional dep still passes.

### Adding a server route

1. Route code goes in `lile/server.py`. Keep it thin — parse pydantic, dispatch to `Controller`, format response.
2. If the route mutates training state, it must go through `Controller.submit_*` which enqueues on the compute queue. Never touch `state.model` or `state.tokenizer` from a route directly.
3. Mirror the route in `studio/backend/routes/lile.py` if Studio should expose it (transparent proxy for `/v1/*`, bespoke handler for `/capsule/*`).

### Adding a Studio UI slice

Lives in `studio/frontend/src/features/lile/`. Stores are zustand; polling hooks subscribe to a stable trajectory reference (see `chart-utils.ts` for the hook wrappers — raw selectors return a new array each render and trip `useSyncExternalStore`, causing "Maximum update depth exceeded"). New chart cards should use `useLossSeries` / `useGradNormSeries` / etc. rather than inlining `useLileCapsuleStore((s) => …)` with a derived array.

## Running the tests

```bash
# Unit-level smoke (fast, CPU-only where possible)
uv run python -m lile.tests.smoke_objectives

# Full HTTP path (requires a GPU with a 4-bit Qwen3-0.6B in HF cache)
uv run python -m lile.tests.smoke_server

# Concurrent-load invariant — the one that fails under reordering
uv run pytest lile/tests/test_concurrent_load.py -xvs

# CCPD v2 end-to-end
uv run pytest lile/tests/test_ccpd_e2e.py -xvs
```

Use `uv`, never `pip`. Every test under `lile/tests/` that has `test_` prefix is a pytest; files prefixed `smoke_` are executable scripts.

## Anti-patterns — do not

- **Do not introduce a second model instance** (a "reference model" or "inference copy"). KL-anchor uses `disable_adapter()` on the live model by default; the optional frozen-ref toggle (`ModelState.load_frozen_ref`) loads a second base-only NF4 model in eval/no-grad and pays the VRAM cost only if the user opts in.
- **Do not catch-and-swallow inside `Controller` or `TrainEngine`**. Those need to surface errors via the queue task's exception so `wait_for` rejects properly. Swallow only in metrics adapters (they're optional side-effects).
- **Do not do blocking I/O on the asyncio loop.** Inference runs in a thread; training runs in the queue worker; the event loop is for routing and `q.put`. If you need to call into PyTorch from a route, use `loop.run_in_executor` or `asyncio.run_coroutine_threadsafe` like `stream_generate` does.
- **Do not normalize or reformat the trajectory JSONL on write.** Consumers (the snapshot, the replay scheduler, trajectory-offset dedup) rely on stable byte offsets. Append-only, one JSON object per line, no re-writing.
- **Do not wire a raw zustand selector returning a new array into a chart component.** Use the `use*Series` hook wrappers in `chart-utils.ts` (see the React invariant above).
- **Do not merge feature branches with a standard merge commit.** Squash-merge into `ht` — history stays linear and the HT-CHANGELOG line matches the PR title.

## Gotchas the tests already guard against

- **4-bit merge path** (`test_merge_and_e2e.py`): merging requantized deltas back into NF4 silently degrades. The ship path keeps `merged_deltas` as bf16 and applies as a forward-time residual. Idempotent null merge is pinned.
- **Residual live at forward time** (`test_residual_live_path.py`): after `merge_active_into_residual()` zeroes the active adapter, the forward stays within 0.05 nats of the pre-merge trained-adapter forward — not an ~86-nat gap. If this test regresses, the monkey-patch bind site is wrong.
- **T3.1 span_prefix mask geometry** (`test_span_prefix.py`): token boundary between kept prefix and regenerated suffix is resolved by walking token slices and checking `tokenizer.decode(...).endswith(span_prefix)`. A naive tokenize-then-LCP approach misaligns on Qwen3's auto `<think>` block; don't simplify it.
- **Concurrent train+infer** (`test_concurrent_load.py`): 10+10 under contention, all DESIGN invariants hold. If a refactor trips this, the mode_lock, queue worker, or commit-cursor read path got shaken.

## Where to read deeper

- `lile/DESIGN.md` — every load-bearing decision with a rationale.
- `lile/STATUS.md` — test-by-test evidence; always check this after a substantive change.
- `lile/GLOSSARY.md` — `Razin-safe` and other terms that appear in code comments.
- `lile/PLAN.md` — the long-form spec and roadmap (phases 0–6).
- `lile/SUBMISSION.md` — the narrative that framed the initial build, if you're reconstructing intent.
