# HT Fork Changelog

All notable changes in the HT fork (relative to upstream unsloth) are documented here.

## Unreleased

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
