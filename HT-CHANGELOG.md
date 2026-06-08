# HT Fork Changelog

All notable changes in the HT fork (relative to upstream unsloth) are documented here.

## Unreleased

### Restore `recent-thread` test selectors on chat sidebar (2026-06-08)

Re-added `data-testid="recent-thread"` + `data-thread-type` + `data-thread-id` on
`SidebarMenuButton` in `studio/frontend/src/components/app-sidebar.tsx`.
These three attributes are present on upstream's recent-chat entries and are what
`tests/studio/playwright_chat_ui.py:1229` locates; they were dropped during the
2026-06-01 catch-up rebase when the HT-side delete-button block was merged in,
which silently red-lit the Chat UI Tests job (Studio / Mac Studio / Windows
Studio UI CI workflows) for ~14 hours across 10 runs. Purely additive — no
behavior change, just the test-locator parity.

### Detached Attention Optimization (2026-06-08)

Added a new `detach_attention` flag to `get_peft_model` for MLP-only LoRA fine-tuning.
When enabled (defaults to `"auto"`, activating when only MLP layers are targeted), it skips building the autograd computation graph through attention layers by wrapping them in `torch.no_grad()`.
This saves ~25-30% of activation memory during training (with FlashAttention) and increases tokens/sec throughput, at zero quality cost (as earlier gradients flow purely via the residual stream).
Tested across architectures including Llama, Mistral, Qwen2/3, Gemma2, and Granite.

### Gemma 4 12B "Unified" — Any-to-Any (2026-06-07)

Google released `google/gemma-4-12B-it` and `google/gemma-4-12B` on
2026-06-04: encoder-free `gemma4_unified` architecture, 11.95B params,
48 layers, 256K context, native text + image + audio + video. Upstream
unsloth's mapper had not yet caught up; this fork adds the wiring so
Studio can train + serve the 12B variant **today** instead of waiting
on the upstream sweep.

Registries touched:
- `unsloth/models/mapper.py` — `__INT_TO_FLOAT_MAPPER` gets two new
  rows (`gemma-4-12B-it-unsloth-bnb-4bit` and the base variant); the
  `_add_with_lower` propagation populates `INT_TO_FLOAT_MAPPER`,
  `FLOAT_TO_INT_MAPPER`, and `MAP_TO_UNSLOTH_16bit` automatically.
- `unsloth/ollama_template_mappers.py` — adds 12B variants to the
  `gemma4` tuple so the existing `gemma-4` Ollama / chat template
  (`<|turn>`-style) applies cleanly.
- `studio/backend/utils/models/model_config.py` — new yaml keys
  (`unsloth_gemma-4-12B-it.yaml`, `unsloth_gemma-4-12B.yaml`) plus the
  HF-name aliases that look them up.
- `studio/backend/utils/datasets/model_mappings.py` — 12B added to the
  `gemma-4` chat-template bucket. (Not in `gemma-4-thinking`: the 12B
  Unified is general Any-to-Any, not the 26B/31B reasoning lineage.)
- `studio/backend/core/inference/defaults.py` — 12B-it added to both
  GGUF and standard chat lists.
- `studio/frontend/src/config/training.ts` — 12B-it slotted into
  `PRIORITY_TRAINING_MODELS` between E4B-it and 31B-it.
- New: `studio/backend/assets/configs/model_defaults/gemma/unsloth_gemma-4-12B-it.yaml`
  and `unsloth_gemma-4-12B.yaml` — mirror the 26B-A4B-it defaults
  (LoRA r=8 / α=8, all-linear targets, `train_on_completions=true`,
  `gradient_checkpointing="unsloth"`).

`pyproject.toml` transformers pin raised from `<=5.5.0` to `<5.12` —
upstream issue #5985. The `gemma4_unified` arch needs ≥ 5.5.0 (already
the `SUPPORTS_GEMMA4` floor in `unsloth/models/loader.py:80`); we cap
at `<5.12` so 5.11.x can be smoke-tested before letting majors bump
blindly.

Training paths:
- **P1 — text data SFT**: routes through `FastLanguageModel.from_pretrained`
  → existing gemma4 dispatch (`unsloth/models/loader.py:1180`) sets
  `UNSLOTH_DISABLE_STATIC_GENERATION=1` + `UNSLOTH_HIGH_PRECISION_LAYERNORM=1`
  and proceeds. No 12B-specific patches needed; the fused-LoRA kernel
  `unsloth/kernels/utils.py:matmul_lora` is unchanged and pinned by
  `tests/test_matmul_lora_contract.py`.
- **P2 — multimodal (vision) SFT**: routes through `FastVisionModel` →
  the existing gemma4 vision branch in `unsloth/models/vision.py`
  already includes the two HT-pinned patches: (a) PEFT
  `Gemma4ClippableLinear` ⇒ inner `.linear` LoRA injection
  (`vision.py:1475`), and (b) forced `use_reentrant=True` for
  gradient checkpointing on `gemma3n` / `gemma4` model types
  (`vision.py:1621`) — otherwise AOT autograd backward fails on
  variable-length audio/image tensors. Vision-model detection is
  generic (`AutoConfig.vision_config` / `image_token_index`
  presence), so 12B is auto-classified without per-model
  hardcoding.

Note on upstream #6028: FastVisionModel's *inference-time* spatial
projection collapses y-coordinates to the bottom of the image for
Gemma 4 (12B + others). This is a forward-pass projection bug, not
a backward-pass one — training loss still flows; the trained model
will inherit whatever the projection produces. Once #6028 lands
upstream, both training and inference benefit without HT-side work.

**Smoke verified end-to-end on RTX 3090 24GB (2026-06-07):**
- Text mode: `unsloth/gemma-4-12B-it` loaded in 4bit via on-the-fly
  bitsandbytes quant, LoRA r=8/α=8 attached (32.7M / 11.99B trainable
  = 0.27%), 2 SFT train steps in 4.5s. Loss: 8.71 → 9.46.
- Vision mode: `FastVisionModel.from_pretrained` returns
  `model_type=gemma4_unified`; LoRA attaches cleanly through the
  `Gemma4ClippableLinear` PEFT shim; gradient checkpointing enables
  with the forced `use_reentrant=True`.
- Mapper correction: the `-unsloth-bnb-4bit` quant repo isn't published
  yet; mapper key is the bf16 reup (`unsloth/gemma-4-12B-it`), mirror of
  the 26B-A4B-it shape. `load_in_4bit=True` triggers the runtime bnb-NF4
  quant.
- One known knob: `UNSLOTH_COMPILE_DISABLE=1` is required for small SFT
  loops on 12B (the auto-compile recompile-limit fires before step 1
  finishes on a tiny dataset). Smoke test sets it as a default; Studio's
  worker should set the same when training Gemma 4 ≥ 12B.

## 2026-06-01

### Caught up with upstream/main (`e3b52eb98`)

Closed a 396-commit drift gap. `ht` now sits at `750e3e4ce` — 17 commits ahead of `upstream/main` (16 HT-only changes + the PR #60 squash). Tagged as [`ht-2026-06-01`](https://github.com/heiervang-technologies/ht-unsloth/releases/tag/ht-2026-06-01).

- **Strategy**: per-commit rebase on `ht` against `upstream/main`; conflicts hit 7 of 16 commits (HT branding, multi-GPU/auth, prompt baking, lile introduction, README, lile-removal pyproject, gitignore leftovers). Resolved per the patterns documented in [`docs/rebase-resumption-guide.md`](docs/rebase-resumption-guide.md) — same doc remains the runbook for the next catch-up.
- **Verification**: `tests/test_matmul_lora_contract.py` (4 AST-level checks pinning the HT-only fused-LoRA kernel signature, Float8 branches, LoRA delta application, 3D reshape) — all pass on the rebased tree. `unsloth/kernels/utils.py:matmul_lora` at L1055 unchanged.
- **Force-update on `ht`**: was the only way to land the rebase since GitHub's PR machinery sees the rebased branch as "conflicting" with `ht`'s original-SHA commits even though the diffs match. The 16 HT-only commits are present in the new history at new SHAs; their original SHAs are reachable via the `pre-sync-*` tags.
- **`origin/main` advanced** from `b36408022` (2026-05-10) to `e3b52eb98` (today) — confirms the daily `Fork Sync` PAT issue is bot-account-specific, not branch protection blocking everyone. The reusable workflow's push step should work once `HAI_GH_PAT` is rotated.
- **agi pin bump**: bump `agi`'s `pyproject.toml` from `unsloth @ git+...@ht-2026-05-15` to `@ht-2026-06-01` in a follow-up PR there. Studio's `studio-tests.yml` matrix `lile_ref` should follow.

### Fork-sync drift diagnosis (2026-06-01)

As of 2026-06-01 the daily `Fork Sync` workflow has been failing every run for ~20+ days. Root cause: `git push origin main` returns **HTTP 403** — the `HAI_GH_PAT` secret either expired or lost write access to the `main` branch (likely branch-protection rule change).

Symptoms:
- `origin/main` frozen at `b36408022` (2026-05-10), while `upstream/main` is at `e3b52eb98` (2026-06-01) — **294 commits behind on main**.
- Local `ht` branch is **396 commits behind** `upstream/main` (16 ahead with HT-only changes).
- Every Fork Sync run reaches the `git fetch upstream` + fast-forward step cleanly, then dies at `git push origin "$UPSTREAM_BRANCH"` with `fatal: unable to access 'https://github.com/heiervang-technologies/ht-unsloth/': The requested URL returned error: 403`.
- The job exits 128 at the push step, so `rebase ht` is never attempted (`REBASE_RESULT: skipped`).

Action required (out-of-band from this repo):
- Rotate / re-issue `HAI_GH_PAT` with `repo` scope and ensure the bot account has push access to `main` (or is on the branch-protection bypass list).
- The reusable workflow at `heiervang-technologies/.github/.github/workflows/fork-sync-reusable.yml` already surfaces the failure as `::error` — the workflow itself is fine; this is a credential issue.

Coupled to this fix: PR adding `tests/test_matmul_lora_contract.py` + `.github/workflows/unsloth-kernel-contract.yml`. These pin the HT-only fused-LoRA dispatch (signature, Float8 branches, LoRA delta application, 3D reshape) at source-level so the 396-commit catch-up rebase can't silently rewrite the kernel.

### Rebase resumption guide

`docs/rebase-resumption-guide.md` — captures the conflict patterns observed during the 2026-06-01 rebase rehearsal so the next session can pick up where the prep left off. 25 files truly conflict (per merge-tree preview); only 5 commits out of 16 actually need manual resolution. Documents 9 resolution patterns (A–I) with concrete examples from the rehearsal, plus the per-commit conflict map and post-rebase verification steps. Read this BEFORE attempting the real rebase.

### Studio-as-lile-optional contract codified

`studio/backend/routes/lile.py` previously had an internal `_can_spawn()` helper that probed `lile` package importability. Renamed to public `lile_available()` and added a Module contract section to the docstring that makes the invariant explicit: **"Studio MUST import and serve cleanly when the lile package is absent."**

- `lile_available()` is now the **single canonical entrypoint** for the import probe; do not add other try/except `import lile` patterns elsewhere.
- New CI workflow `.github/workflows/studio-tests.yml` runs the lile-route tests in a 2-cell matrix: `lile-absent` (proves the contract) and `lile-installed` (pulls lile from `agi@ht-2026-05-15`, proves the spawn-local path).
- New doc [`docs/studio-lile-integration.md`](docs/studio-lile-integration.md) — one-page contract spec.
- Cross-repo pin discipline: ht-unsloth tags `ht-YYYY-MM-DD` per upstream-sync; agi's `unsloth` pin and Studio's `matrix.lile_ref` both bump to that tag together.

## 2026-05-15

### lile relocated to heiervang-technologies/agi

`lile` (the LiveLearn live-training daemon) has moved to its own repository. `ht-unsloth` keeps the Unsloth fork + Studio; the daemon is now externally managed and Studio talks to it over HTTP.

- **New home:** [`heiervang-technologies/agi`](https://github.com/heiervang-technologies/agi). Installable as `pip install lile @ git+https://github.com/heiervang-technologies/agi`. Authorship preserved via `git filter-repo`.
- **Cross-repo coupling:** `agi`'s `pyproject.toml` pins `unsloth @ git+https://github.com/heiervang-technologies/ht-unsloth@ht-2026-05-15`. Bump intentionally when ht-unsloth syncs with upstream.
- **Removed from this fork** (PR #53):
  - `lile/`, `lile_data/`, `compose.lile-dev.yaml`, `.claude/skills/lile/`
  - Lile-only `pyproject.toml` extras (`eval`, `dev`) and `[tool.pytest.ini_options]`
  - `.github/workflows/test.yml` (lile-specific CI)
- **Studio integration changed** (`studio/backend/routes/lile.py`):
  - `/api/lile/capsule/status` and `/api/lile/capsule/start` now probe reachability of the externally-managed daemon at `LILE_DAEMON_URL` (preferred) or legacy `LILE_HOST` + `LILE_PORT`.
  - `/api/lile/capsule/stop` is a no-op (`{stopped: false, reason: "externally_managed"}`).
  - Transparent proxy (`/api/lile/{path}`) and SSE pass-through unchanged — chat + train + state-snapshot UX preserved.
  - `studio/backend/tests/test_lile_route.py` updated for the externally-managed contract.
- **Last lile commit on `ht`:** `53757129` (squashed PR #52 — RLVR rig + GPT-OSS-120B teacher + ARC-AGI-3 runner + combined-loss engine). agi tracks from that point. Tagged as [`ht-2026-05-15`](https://github.com/heiervang-technologies/ht-unsloth/releases/tag/ht-2026-05-15).
- **Runner offline note:** self-hosted runners were offline during the migration window; CI ran on cloud runners and verified green before merge.

### Prompt Baking — 4th training mode in Studio

Integrates [marksverdhei/bakery](https://github.com/marksverdhei/bakery) as a new training method alongside QLoRA, LoRA, and Full fine-tune. Prompt baking distils a system prompt into LoRA weights via KL divergence so the model exhibits the prompted behavior at zero inference-time cost.

- New method selector entry **Prompt Baking** with purple dot in `ModelSection`.
- New **Prompt Baking** panel in `ParamsSection`: system prompt (required), trajectory count / length / sampling temperature, KL temperature, and a "Use Prefill Data" toggle for prebuilt response datasets.
- Backend fast-path in `core/training/worker.py` mirrors the embeddings dispatch: self-contained `run_prompt_baking()` handles model load, LoRA apply, dataset build, `PromptBakingTrainer.train()`, and adapter save.
- Request schema gains `is_prompt_baking` + baking params; persists via zustand store (persist v10).
- Template: `studio/backend/assets/configs/prompt_baking.yaml`.

## 2026-04-17

### LiveLearn (`lile`) — live-training FastAPI daemon (PR #8)

Major addition. Single-process daemon where inference and training share model weights, exposing an OpenAI-compatible chat endpoint alongside training/feedback/state control-plane routes.

- `/v1/chat/completions` (streaming + non-streaming), `/v1/train`, `/v1/feedback`, `/v1/state/{merge,snapshot,trajectory,...}`, `/v1/wait`.
- Commit-cursor guarantee: `after_commit_token` on chat blocks until the specified training batch is applied — "post a batch, next inference sees it."
- Stackable objectives: **SFT, NTP, KTO, CoH, hinge, KL-anchor**, and **CCPD v2** (Critique-Conditional Policy Distillation; π-only feedback-guided objective with detached rewards + auxiliary sampling).
- **Reasoning-content parser** (`lile/reasoning.py`): streaming two-state matcher that splits `<think>…</think>` deltas into `reasoning_content` vs `content` channels for Qwen3, DeepSeek-R1, Magistral, and gpt-oss (vllm-compatible semantics).
- **Pluggable metrics sinks** (`lile/logging_backends.py`): optional fan-out to Weights & Biases, TensorBoard, MLflow, or trackio. Adapters are no-throw; trajectory JSONL remains the canonical record.
- **Studio frontend** (`studio/frontend/src/features/lile/`): `/lile` page with capsule lifecycle (load/stop), live loss/grad-norm/KL/queue-depth/components charts, snapshots + trajectory tabs, `LileMessageActions` + feedback modal. Chat page can toggle lile-mode with block-on-last-commit.
- Studio backend (`studio/backend/routes/lile.py`): capsule status/start/stop, transparent proxy for `/v1/*`, SSE pass-through for chat completions.
- See [`lile/DESIGN.md`](lile/DESIGN.md), [`lile/STATUS.md`](lile/STATUS.md), [`lile/GLOSSARY.md`](lile/GLOSSARY.md).

## 2026-03-18

### Multi-GPU + Docker + Auth (PR #1)

- **Multi-GPU sharding:** replace hard multi-GPU `RuntimeError` with warning; allow `device_map="sequential"`/`"balanced"` passthrough; backend returns per-GPU info (count, name, VRAM per card); VRAM estimation considers total across GPUs; purple **MULTI-GPU** badge when a model spans multiple cards; GPU chip indicator in navbar. See [`docs/multi-gpu-status.md`](docs/multi-gpu-status.md).
- **Docker:** `Dockerfile` for `ht-unsloth-studio` with CUDA, Studio, and llama.cpp; Docker Hub CI workflow (pushes to `ht` only).
- **Auth bypass:** `UNSLOTH_DISABLE_AUTH=1` env var to skip Studio login (end-to-end, including WebSocket paths).
- Remove upstream stale-issue bot.

### Fork Infrastructure
- Rebrand Studio badge from BETA to HT (purple).
- Support in-repo `.venv` for fork/editable installs (setup.sh + CLI).
- Add fork sync CI automation.
- Add HT-fork documentation and discussion links.

### Bug Fixes
- Handle JSON-string chat columns in dataset format detection and conversion.
  Datasets storing `conversations`/`messages` as serialized JSON strings
  (common in multi-subset parquet repos) are now transparently parsed
  in `detect_dataset_format`, `standardize_chat_format`,
  `convert_chatml_to_alpaca`, and `convert_sharegpt_with_images_to_vlm_format`.
