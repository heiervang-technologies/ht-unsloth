# LiveLearn (`lile`) - Design Note

## Top-Level Architectural Choices

1. **Engine Choice (§8 Q1):** We are defaulting to **Unsloth's `fast_generate`** for the inference engine. The target hardware is 1x RTX 3090 (24GB VRAM). Maintaining a single process that swaps between generation and training via weight sharing is the only way to comfortably fit a 7B-14B model + context + LoRA + rollouts in 24GB.

2. **Default Model (§8 Q2):** We will target `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (or equivalent Qwen3 8B if available) as the primary validation model. It fits well in 4-bit with a 16-bit LoRA and leaves ~8GB VRAM headroom for the auxiliary serving KV pool.

3. **Depth vs Breadth:** The primary focus is deep implementation of **Tier 1 (KTO, CoH) + Tier 2.1 (CCPD v2 light)**. Tier 3 (trace infilling) and Tier 4 (background replay) are deferred until the core loop is robust.

4. **CCPD v2 Prerequisite:** We are running the ranking reliability benchmark (§11) *before* implementing the CCPD v2 gradient paths. We will test whether `r_c` produces reliable rankings of auxiliary rollouts using the detached scoring method.

5. **Merge Path:** The 4-bit progressive merge will explicitly maintain `merged_deltas` in bf16 or fp32. We will apply this as a residual during the forward pass, avoiding degradation from repeated 4-bit quantization cycles.

## Phased Approach for this Session
1. **Benchmark (§11):** Run ranking-reliability test.
2. **Core Loop:** Build `lile.server` and `lile.engine` with standard SFT.
3. **Advanced Tier:** Implement CCPD v2 (if benchmark passes) or hinge-contrastive (if benchmark fails).
4. **Queue & Merge:** Implement the async commit cursor and progressive merge path.
