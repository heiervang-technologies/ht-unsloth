# `bitsandbytes` 8-bit optimizer stability — version guidance

- **Scope**: known-bad versions, block size tuning, stability regressions for `AdamW8bit` on LoRA.
- **Compiled**: 2026-04-17 (background agent).

Tags: **[STRONG]** · **[RELEVANT]** · **[BACKGROUND]** · **[SKIP]**.

---

## Version pin recommendation `[STRONG]`

Current `pyproject.toml` pin:
```
bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0
```

- `>=0.45.5`: block size was tightened in 0.45.x; older versions can produce noticeably higher optimizer-state quantization error on LoRA-sized param groups.
- `!=0.46.0`: known regression on Ampere CUDA forward path.
- `!=0.48.0`: intermittent NaN on `AdamW8bit.step()` under specific param-group counts.

**Recommendation**: keep the current pin. Agent found nothing below `0.44.0` lower bound to justify relaxing.

## Block size caveat `[RELEVANT]`
Older bnb (<0.45) defaults to block=2048 for AdamW8bit state quantization. Small LoRA tensors get fewer blocks and higher relative quant error. 0.45+ defaults to block=256 for optimizer state (vs block=64/128 for weight quant), giving better quantization fidelity at LoRA scale at negligible memory cost.

## Stability guardrails `[STRONG]`
- **Always use `grad_clip≥1.0`** with `AdamW8bit`. Without it, the dequant/requant path can amplify a rare outlier gradient into NaN `v`.
- **Rebuild the optimizer on snapshot load** (PR A, shipped). Quantized `m`/`v` from pre-snapshot weights are actively harmful on post-snapshot weights.
- **Avoid multiple `AdamW8bit` instances over the same tensors.** `GlobalOptimManager` is a process-level singleton; two instances share registration but split state. Use PyTorch-style param_groups on a single optimizer instead.

## Fallback path `[RELEVANT]`
When 8-bit is unavailable (no CUDA / no bnb install), `lile/engine/train.py::_optimizer` falls back to `torch.optim.AdamW`. This is already implemented and tested. Memory cost at LoRA r=16 on Qwen3-8B: ~102 MB per 32-bit state vs ~25 MB for 8-bit — negligible at our peak (8.21 GB measured).

---

## Sources
- [bitsandbytes release notes](https://github.com/bitsandbytes-foundation/bitsandbytes/releases)
- [HF bitsandbytes optimizers docs](https://huggingface.co/docs/bitsandbytes/main/en/optimizers)
- [arXiv:2510.02334](https://arxiv.org/abs/2510.02334) — Improving bnb quantization for AdamW
