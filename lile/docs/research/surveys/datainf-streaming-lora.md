# Influence functions for streaming LoRA adaptation

- **Scope**: DataInf, LESS, GREATS, and the open gap for streaming LoRA influence estimation.
- **Compiled**: 2026-04-17 (background agent).

Tags: **[STRONG]** · **[RELEVANT]** · **[BACKGROUND]** · **[SKIP]**.

---

## Current methods (all batch, all offline)

### DataInf `[RELEVANT]`
- Kwon et al., ICLR 2024. Closed-form influence-function approximation for LoRA-tuned models using a Hessian-free swap; scales to adapter-sized parameter counts.
- Typical use: rank all training samples by influence on validation loss; drop or up-weight top/bottom-k.
- **Limitation for `lile`**: designed for a batch of training examples against a fixed validation set — no streaming API.

### LESS `[RELEVANT]`
- Xia et al., ICML 2024. Gradient-feature ranking for instruction-tuning data selection; uses Adam-style second-moment normalization.
- **Limitation for `lile`**: pre-computed features on a static dataset.

### GREATS `[BACKGROUND]`
- NeurIPS 2024. Online variant that selects the next batch to train on from a candidate pool via influence heuristics.
- Closer to a streaming shape but still assumes a candidate pool, not a single arriving-one-by-one feedback event.

### Metagradient Descent / REPLAY (2025) `[RELEVANT]`
- Engstrom et al., arXiv:2503.13751. Exact gradients **through** model training at billion-parameter scale via reverse-mode AD + smooth-model-training.
- **If REPLAY scales to LoRA deltas**, it gives principled per-feedback-item importance weights — directly competitive with DataInf in the streaming setting. Unverified at daemon scale.

---

## The open gap `[STRONG]`

No published method combines **DataInf-style influence estimation with a streaming LoRA daemon**.

Specifically:

1. Existing influence methods assume a fixed training set. `lile` has an append-only trajectory.
2. Existing methods rank full gradients against a fixed validation set. `lile` needs a *per-feedback-event* estimate of "should this train step happen at all, and with what weight?"
3. Existing methods pay a Hessian-inverse cost. Online settings can't afford that per step.

## Sketch: a streaming DataInf probe for `lile`

A ~100 LOC experimental probe at `lile/teach/datainf_probe.py`:

1. Maintain a rolling fixed-size validation set (e.g., 50 examples from the regression harness).
2. On each incoming feedback event, compute the LoRA gradient `g` against the event + the LoRA gradient `g_val` against the validation batch.
3. Estimate influence ≈ `-g · g_val / (||g||² + ε)` (first-order DataInf collapse).
4. If influence < 0 (sample would hurt validation): skip the step OR down-weight it.
5. If influence > threshold: mark for replay-buffer priority.

**Gate**: this is speculative. Needs eval-harness validation before shipping. `[STRONG]` only insofar as the gap is real — whether our probe is the right closure is open.

---

## Sources
- [DataInf (ICLR 2024)](https://openreview.net/forum?id=9m02ib92Wz)
- [LESS (ICML 2024)](https://arxiv.org/abs/2402.04333)
- [GREATS (NeurIPS 2024)](https://arxiv.org/abs/2410.22108)
- [Metagradient Descent / REPLAY](https://arxiv.org/abs/2503.13751)
