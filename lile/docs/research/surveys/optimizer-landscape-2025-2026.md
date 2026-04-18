# Optimizer landscape, 2025–2026 — LoRA + online finetuning

- **Scope**: optimizer choice for LoRA-based online finetuning; AdamW/Lion/SF-AdamW/Muon/AdEMAMix landscape.
- **Compiled**: 2026-04-17 (background agent, findings already folded into `optimizer-sample-efficiency.md`).

Tags: **[STRONG]** · **[RELEVANT]** · **[BACKGROUND]** · **[SKIP]**.

---

## Key 2025 results

### AdEMAMix — Better, Faster, Older `[STRONG]`
- Pagliardini et al., arXiv:2409.03137. **Two** EMAs of the gradient (one fast, one slow); empirically tracks older gradients without destabilizing current training. Explicitly shown to **"significantly slow model forgetting"** on continual pretraining.
- **Why it matters**: anti-forgetting property maps directly to `lile`'s catastrophic-forgetting concern. The **most important omission** from the original optimizer doc. Candidate for a new PR between ScheduleFree (D) and Muon (E).
- Simplified-AdEMAMix (arXiv:2502.02431) removes one of the two EMAs for faster convergence; same anti-forgetting claim.

### Benchmarking Optimizers for LLM Pretraining — small-batch caveats `[STRONG]`
- arXiv:2509.01440. Broad sweep of AdamW / Lion / SF-AdamW / Muon / AdEMAMix / SOAP across pretraining budgets.
- **Key caveat for `lile`**: **SF-AdamW underperforms AdamW in small-batch regimes** — which IS `lile`'s regime at `chunk_size=2`. This is a risk note on PR D (ScheduleFree) in the optimizer doc.
- Muon wins mid-batch pretraining but the advantage collapses at small-batch or LoRA finetuning.

### AdamW8bit + param_groups — the correct idiom `[STRONG]`
- `bitsandbytes` `AdamW8bit` accepts PyTorch-style param_groups (`[{"params": ..., "lr": ...}, ...]`) via `register_parameters`. **A single AdamW8bit with multiple groups is idiomatic and avoids GlobalOptimManager singleton collisions.**
- **Why it matters**: the original §3 diff sketch proposed `self._opts: dict[str, torch.optim.Optimizer]` (multiple instances) — that was wrong. Correct implementation: one `AdamW8bit`, multiple groups keyed by objective name.
- No published guidance recommends multiple 8-bit-optimizer *instances* over the same tensors.

### Lion8bit at LoRA scale `[RELEVANT]`
- Chen et al., arXiv:2302.06675 (original Lion). Cautious Lion (Liang et al., arXiv:2411.16085) adds one-line stability improvement.
- Sign-based update makes step magnitude exactly `lr` regardless of gradient scale — removes the objective-mixing `v`-corruption concern by construction. Half the optimizer-state memory of AdamW.
- 2025 benchmarks: matches or beats AdamW on LM pretraining. Known hazard: non-convergence from sign discretization in certain settings (RLion paper, 2025).

### Muon / Riemannion call for Qwen3 `[RELEVANT]`
- Moonlight ([Liu et al., 2025](https://arxiv.org/html/2502.16982v1)): SFT with Muon does not beat AdamW when pretrain optimizer differs. Confirmed independently on Llama-3.2-3B ([Shi et al., 2025](https://arxiv.org/html/2509.23106v1)) and on Qwen2-0.5B / SmolLM2-360M / GPT2-medium ([MuonAll, Nov 2025](https://arxiv.org/html/2511.06086v1)).
- Qwen3 is AdamW-pretrained. Empirical ceiling on a Muon A/B for us is "matches AdamW" — not a win under the eval-harness bar.
- Riemannion-LoRA ([Bogachev et al., 2025](https://arxiv.org/abs/2507.12142)): 32.5% wall-clock overhead at r=16; does not evaluate on GSM8K or HumanEval. Defer.

### ScheduleFree-AdamW `[RELEVANT]`
- Defazio et al., [arXiv:2405.15682](https://arxiv.org/abs/2405.15682). Streaming validation in [Baek et al., "Through the River", 2025](https://arxiv.org/abs/2507.09846).
- Only optimizer in this set that explicitly targets no-known-horizon streaming. Composes cleanly with `lile`'s rehearsal loop.
- **Caveat**: small-batch regime may underperform AdamW (see 2509.01440 above).

---

## Open citations (added to `optimizer-sample-efficiency.md` References)

- [arXiv:2509.01440](https://arxiv.org/abs/2509.01440) — Benchmarking Optimizers for LLM Pretraining
- [arXiv:2409.03137](https://arxiv.org/abs/2409.03137) — AdEMAMix
- [arXiv:2502.02431](https://arxiv.org/abs/2502.02431) — Simplified AdEMAMix
- [arXiv:2505.02222](https://arxiv.org/abs/2505.02222) — AdEMAMix8bit discussion
- [arXiv:2509.03378](https://arxiv.org/abs/2509.03378) — bnb large-model quantized Muon
- [arXiv:2510.02334](https://arxiv.org/abs/2510.02334) — Improving bnb quantization for AdamW (block size)
