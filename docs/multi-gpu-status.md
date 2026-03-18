# Multi-GPU Training in Unsloth: Status Report

**Date:** 2026-03-18
**Scope:** Upstream unsloth, HT fork, and HT Unsloth Studio integration

---

## Executive Summary

Multi-GPU support in Unsloth has been "on the roadmap" since issue #127 (Oct 2024). After 18 months, DDP for text models is partially working via merged PRs, but the experience remains fragile, architecture-dependent, and completely absent from Studio. The user's goal --- QLoRA with model sharded sequentially across two cards when too large for one --- maps to `device_map="sequential"/"balanced"`, which is the *least* stable of the two multi-GPU approaches.

---

## Two Approaches, Two Different Problems

### 1. DDP (Data Parallel) --- model replicated per GPU

Each GPU holds a full copy of the model and processes different data batches. Gradients sync via all-reduce.

| Aspect | Status |
|--------|--------|
| Text SFT | Working (Llama, Gemma) |
| VLM SFT | Partially working (PR #4240 pending for CPU offload fix) |
| GRPO/RL | Broken (`DDP` wrapper hides `model.config`) |
| DeepSpeed ZeRO | Broken (smart gradient offloading conflict, #4195) |
| FSDP | Not functional (bitsandbytes 4-bit incompatible) |

**Constraint:** Model must fit on ONE GPU. No memory saving --- just throughput scaling.

### 2. Model Sharding (`device_map="balanced"/"sequential"`) --- model split across GPUs

Different layers live on different GPUs. Single process, sequential pipeline.

| Aspect | Status |
|--------|--------|
| Inference | Working for some architectures |
| Training | Fails consistently with attention/RoPE device mismatches |
| QLoRA | bitsandbytes quantized weights can't be relocated after creation |
| Throughput | No scaling (same speed as single GPU, just enables larger models) |

**This is the approach needed for "model too large for one card" but it is the most broken.**

---

## Why This Has Been On The Roadmap Forever

### 1. Unsloth's architecture is fundamentally single-GPU

The codebase has a **hard RuntimeError** that blocks multi-GPU (`tokenizer_utils.py:1050`):
```python
if ((a - PRE_CHECK) >= 1).sum() > 1:
    raise RuntimeError('Unsloth currently does not support multi GPU setups')
```

The `__init__.py` literally says: *"Unsloth currently does not work on multi GPU setups --- sadly we are a 2 brother team."*

### 2. Custom CUDA kernels assume single-device tensors

Unsloth replaces standard attention/MLP with fused kernels that:
- Assume all tensors (Q/K/V, attention bias, RoPE embeddings) are on the same device
- Use in-place operations that break autograd under DDP loss scaling
- Apply `UnslothFusedLossBackward` which crashes when `loss *= num_processes`

### 3. bitsandbytes 4-bit quantization is incompatible with model sharding

- 4-bit weights are stored as `uint8`, not float --- FSDP can't shard them
- Quantization metadata (`Params4Bit` scales/zero-points) is lost when tensors move between devices
- bitsandbytes auto-quantizes anything moved to GPU, causing double-quantization corruption
- After quantization, weights **cannot be relocated** to a different device

### 4. GRPO/RL is the stated blocker

Maintainer @shimmyshimmer (mid-2025): *"Actively working on it. Unfortunately it's taking longer because GRPO is an issue."* The DDP wrapper hides `model.config` which Unsloth's RL code accesses directly.

### 5. Two-person team, single-GPU optimization is the value prop

Unsloth's competitive advantage is single-GPU speed (2-5x faster via custom kernels). Multi-GPU is fundamentally at odds with those kernel assumptions.

---

## Upstream Activity (PRs)

### Merged (foundation exists)

| PR | Description | Impact |
|----|-------------|--------|
| #3049 | Fix per-GPU position embeddings, device sync after RoPE | Foundational multi-GPU inference fix |
| #3751 | Non-reentrant checkpointing for VLM DDP | Fixed "marked ready twice" errors |
| #3917 | Load quantized models on correct per-rank GPU | Enabled DDP with 4-bit/8-bit models |
| #4059 | Move labels to logits device in cross-entropy loss | Fixed device mismatch in loss computation |
| #4063 | Move loss/n_items tensors to logits device | Companion to #4059 |
| #4143 | Fix multi-node distributed detection (check WORLD_SIZE) | Fixed 2-node setups |

### Open (pending)

| PR | Description | Significance |
|----|-------------|--------------|
| #4240 | DDP + VLM + CPU offload + TiledMLP fix | Restores CPU offload for VLM DDP |
| #4257 | Context parallelism (sequence sharding across GPUs) | Most advanced --- linear context length scaling |
| #4218 | Fix CUDA_VISIBLE_DEVICES for single-GPU on multi-GPU machine | Quality-of-life fix |

### External forks/repos

| Repo | Approach |
|------|----------|
| [thad0ctor/unsloth-5090-multiple](https://github.com/thad0ctor/unsloth-5090-multiple) | Patches accelerate compatibility for multi-5090 |
| [anhvth/opensloth](https://github.com/anhvth/opensloth) | More extensive multi-GPU patches |
| [cwpeng-cn/unsloth-multi-gpu](https://github.com/cwpeng-cn/unsloth-multi-gpu) | Tutorial/example repo |
| [minhtcai/unsloth-multi-gpu-vision](https://github.com/minhtcai/unsloth-multi-gpu-vision) | Multi-GPU VLM training examples |

---

## Failure Mode Reference

| Approach | Error | Root Cause |
|----------|-------|------------|
| `device_map="balanced"` | `Attention bias and Q/K/V on different devices` | Unsloth kernels assume single device |
| DDP + 4-bit | `Can't train 4-bit model on different device` | Must use `device_map={'': current_device()}` |
| DDP + GRPO | `'DDP' has no attribute 'config'` | DDP wrapper hides model attributes |
| Accelerate + SFT | `Output of UnslothFusedLossBackward is a view, modified inplace` | Loss scaling incompatible with custom backward |
| DeepSpeed ZeRO | `grad_reduc is None` | Smart gradient offloading nullifies gradients |
| torchrun launch | `module 'UnslothGKDTrainer' has no attribute 'UnslothGKDTrainer'` | Cyclic import under multiprocess spawn |

---

## What "QLoRA Sequential on 2 Cards" Actually Requires

The user wants: load a quantized model too large for one GPU across two GPUs sequentially, then train with QLoRA adapters.

This maps to `device_map="sequential"` + `load_in_4bit=True` + LoRA, which requires:

1. **bitsandbytes quantization must happen per-device** --- each layer quantized on its target GPU (not quantize-then-move, which fails)
2. **All intermediate tensors (attention bias, RoPE, causal masks) must follow the layer device** --- currently broken in Unsloth's fused kernels
3. **LoRA adapters must be on the same device as their base layer** --- PEFT handles this if `device_map` is set correctly
4. **Gradient computation must handle cross-device boundaries** --- pipeline parallelism needs activation checkpointing at device boundaries
5. **The hard multi-GPU RuntimeError check must be removed**

### Known working alternative: Answer.AI's FSDP-QLoRA

Answer.AI solved the bitsandbytes sharding problem by:
- Adding `bnb_4bit_quant_storage` to store 4-bit weights as float type (reinterpreted bytes)
- Copying quantization metadata onto `Linear4bit` layer
- Adding guards against double quantization

This is integrated into HuggingFace but **not into Unsloth's custom kernels**.

---

## Complexity of Integration into HT Fork + Studio

### Tier 1: DDP in Studio (medium complexity, high value)

Add a "Multi-GPU (DDP)" toggle to Studio that:
- Detects available GPUs via `torch.cuda.device_count()`
- Launches training with `accelerate launch` or `torchrun` instead of `mp.spawn()`
- Sets `device_map={'': torch.cuda.current_device()}` per rank
- Sets `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars
- Removes/softens the hard RuntimeError multi-GPU check

**Changes needed:**
- `studio/backend/core/training/training.py` --- launch subprocess via torchrun
- `studio/backend/utils/hardware/hardware.py` --- expose GPU count and selection
- `unsloth/tokenizer_utils.py` --- remove hard multi-GPU check
- `unsloth/models/_utils.py` --- don't force `DistributedType.NO` when multi-GPU intended
- Frontend: GPU count display, DDP toggle, per-device batch size config

**Limitation:** Model must fit on one GPU. Throughput scales, memory doesn't.

### Tier 2: Model sharding in Studio (high complexity, your actual need)

Add `device_map="sequential"` support for models too large for one GPU:
- Requires fixing all attention bias / RoPE device placement in Unsloth kernels
- Requires bitsandbytes per-device quantization
- Requires activation checkpointing at device boundaries
- May need to disable some Unsloth kernel optimizations for cross-device layers

**Changes needed (in addition to Tier 1):**
- `unsloth/models/llama.py`, `mistral.py`, `qwen2.py`, etc. --- fix RoPE/attention device placement
- `unsloth/models/loader.py` --- allow `device_map="sequential"/"balanced"`
- Possibly disable fused attention for cross-device layer pairs
- Frontend: model size estimation, auto-detect if sharding is needed, VRAM-per-GPU display

**Risk:** This is where upstream has been stuck for 18 months. The fused kernel assumptions are deeply embedded.

### Tier 3: Hybrid DDP + sharding (very high complexity)

Combine model sharding (to fit the model) with DDP (for throughput). This is essentially pipeline parallelism + data parallelism. Nobody in the Unsloth ecosystem has attempted this.

---

## Recommendation

1. **Start with Tier 1 (DDP in Studio)** --- the merged PRs (#3049, #3917, #4059, #4063, #4143) provide a foundation. This gives throughput scaling for models that fit on one GPU.

2. **For the "model too large for one card" case**, evaluate whether bypassing Unsloth's fused kernels for the cross-device layers is acceptable. If so, use HuggingFace's native `device_map="sequential"` with Unsloth's LoRA/optimizer patches only. This is a hybrid approach: Unsloth's LoRA setup + standard HF training loop.

3. **Monitor upstream PR #4257 (context parallelism)** --- this is the most sophisticated multi-GPU feature being developed and could provide an alternative path for long-context training.

---

## References

- [Unsloth Multi-GPU Docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth)
- [Unsloth DDP Docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp)
- [Answer.AI FSDP-QLoRA](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html)
- [Trelis Research: Multi GPU Training with Unsloth](https://trelis.substack.com/p/multi-gpu-training-with-unsloth)
- [GitHub Issue #2435](https://github.com/unslothai/unsloth/issues/2435) --- main multi-GPU tracking issue
- [GitHub Issue #127](https://github.com/unslothai/unsloth/issues/127) --- original DDP feature request
- [GitHub Issue #4195](https://github.com/unslothai/unsloth/issues/4195) --- DeepSpeed ZeRO crash
- [GitHub Issue #3915](https://github.com/unslothai/unsloth/issues/3915) --- DDP not working
