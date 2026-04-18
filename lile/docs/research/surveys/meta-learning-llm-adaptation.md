# Meta-learning for LLM adaptation, 2024–2026

- **Scope**: meta-learning approaches to sample-efficient LLM adaptation — MAML-family, ICL-as-GD, learned optimizers, hypernetwork LoRA, task clustering, test-time training, few-shot personalization.
- **Compiled**: 2026-04-17 (background agent `ab08374e2c5c7e273`).
- **Filter**: targeted gap in `sample-efficiency-lit-review.md` (no meta-learning section).

Tags: **[STRONG]** · **[RELEVANT]** · **[BACKGROUND]** · **[SKIP]**.

---

## 1. MAML / Reptile / first-order meta-learning at LLM scale

### MAML-en-LLM `[BACKGROUND]`
- Sinha et al., SIGKDD 2024. arXiv:2405.11446. Explicit MAML on LLMs; +2% unseen-domain, +4% adaptation. Second-order gradients + one model-copy per task; doesn't scale past ~7B.

### ABMLL — Low-Rank Amortized Bayesian Meta-Learning `[BACKGROUND]`
- arXiv:2508.14285 (Aug 2025). Constant scaling in task count; benchmarks against Reptile; larger scale than MAML-en-LLM or Kim & Hospedales 2025.

### ReptiLoRA `[STRONG]`
- 2025. Reptile + LoRA on Llama-2-7B, 0/1/5-shot summarization; first-order avoids MAML's second-order cost.
- **Why it matters**: closest first-order meta-learning result to `lile`'s regime. Suggests periodically treating the replay buffer as a task distribution and running a Reptile outer loop over per-user-cluster inner loops.

## 2. ICL as implicit meta-gradient descent

### Metagradient Descent / REPLAY `[STRONG]`
- Engstrom et al., arXiv:2503.13751 (Mar 2025). Exact gradients through model training at billion-parameter scale via reverse-mode AD + smooth-model-training. Used for dataset selection, LR schedules, poisoning defense.
- **Why it matters**: if REPLAY scales to LoRA deltas, it gives principled per-feedback-item importance weights — directly competitive with DataInf.

### COLD-Steer `[SKIP]`
- Training-free activation steering that approximates GD on in-context examples (2025). Inference-only, so nothing actionable for `lile`.

## 3–4. Learning-to-update / learned optimizers

### μLO — Compute-Efficient Meta-Generalization of Learned Optimizers `[BACKGROUND]`
- arXiv:2406.00153 (updated Nov 2025). Applies Maximal Update Parametrization to learned optimizers; meta-generalizes to 5× deeper, 25× longer training horizons. Still not a drop-in replacement for AdamW/Muon at Qwen3 scale.

### Negative result: no 2025–26 "Meta-SGD for LLMs" paper. Research gap.

## 5. Fast-adaptation LoRA (hypernetwork-generated)

### HyperLoRA `[STRONG]`
- Charakorn et al., ICLR 2025. <https://openreview.net/forum?id=u6vC7KaFel>
- Hypernetwork generates LoRA weights from a natural-language task description in one forward pass; compresses hundreds of LoRAs; zero-shots to unseen tasks.
- **Why it matters**: if `lile` can condition on `(user_id, recent_feedback_embedding)`, a hypernetwork could emit the per-session LoRA delta without SGD.

### Text-to-LoRA (T2L) `[STRONG]`
- Jun 2025. 73.4 avg / 67.7 zero-shot on held-out tasks; batched hypernet generation for all modules.
- **Why it matters**: cleaner batched inference story than HyperLoRA. Candidate for a design spike.

### Meta-LoRA `[STRONG]`
- Topal et al., Mar 2025. Shared domain-aware LoRA priors + per-identity specialization in ~50 steps.
- **Why it matters**: the "meta-initialization + fast per-user adapt" pattern `lile` was built for.

### HyperAdaLoRA `[BACKGROUND]`
- arXiv:2510.02630. Hypernetwork-driven dynamic rank allocation during training.

## 6. Task clustering + per-cluster LoRA heads

### TC-LoRA `[STRONG]`
- arXiv:2508.03999. Embedding-space clustering + per-cluster LoRA + CP decomposition. +1.4% Phi-3, +2.3% Mistral-7B.
- **Why it matters**: maps cleanly onto `lile`'s multi-user stream if users cluster by style/domain.

### K-Merge / K-Merge++ `[STRONG]`
- arXiv:2510.13537 (Oct 2025). On-device continual LoRA merging — merge incoming adapter with closest stored one when storage is full. Evaluated across 40 tasks / 5 domains / 8 languages.
- **Why it matters**: directly answers "what do I do when the adapter library grows without bound?"

### MoLE-CIE, D-MoLE, HMoRA, LoRA-Mixer `[BACKGROUND]`
- EMNLP Findings / ICML / ICLR 2025. MoE-style routing variants.

## 7. Test-time training for LLMs

### TTRL — Test-Time RL `[STRONG]`
- arXiv:2504.16084. +211% pass@1 on AIME24 from unlabeled test data via maj@n self-reward on Qwen-2.5-Math-7B.
- **Why it matters**: validates the "RL from implicit signals at inference" pattern `lile` targets.

### TTC-RL — Learning on the Job `[STRONG]`
- arXiv:2510.04786. Test-time curricula for targeted RL; notes **SFT on expert traces hurts TTT**.
- **Why it matters**: concrete warning for `lile`'s SFT-heavy default. Testable claim.

### TLM — Test-Time Learning for LLMs `[BACKGROUND]`
- arXiv:2505.20633. LoRA-based TTT, +20% on domain adaptation.

### qTTT, TTT-E2E `[SKIP]`
- arXiv:2512.13898 / 2512.23675. Long-context TTT; not directly applicable.

## Few-shot personalization (meta-learning flavor)

### FSPO — Few-Shot Preference Optimization `[STRONG]`
- arXiv:2502.19312 (Feb 2025). Reframes reward modeling as meta-learning; synthesizes 1M personalized preferences; real-user transfer works.
- **Why it matters**: directly applicable as a per-user reward-modeling head on top of `lile`'s KTO/hinge objectives.

### Meta Reward Modeling (MRM) `[STRONG]`
- arXiv:2601.18731. MAML-style inner/outer loop over users + robust-personalization objective for hard users.

### Fermi `[BACKGROUND]`
- arXiv:2406.18678 (v2 Mar 2025). Iterative prompt personalization from mis-aligned responses.

---

## Synthesis — four design-level implications beyond the existing lit review

1. **Meta-initialization is a real lever.** Meta-LoRA, ReptiLoRA, MRM, FSPO all show a pre-training meta-learning phase over users/tasks produces a LoRA init that adapts in tens of steps — plausibly `lile`'s cold-start-per-user story.
2. **Hypernetwork-LoRA (T2L, HyperLoRA-ICLR) is the most radical alternative to gradient-based online updates** — worth a design spike, not a full migration.
3. **K-Merge answers the unbounded-adapter-library problem** the current review doesn't address.
4. **TTRL + TTC-RL's warning** (SFT on traces hurts TTT) is a concrete, testable claim against `lile`'s SFT-heavy default and belongs on the eval harness.

**Negative results worth recording**: no Meta-SGD-for-LLMs paper in 2025–26; no paper combines DataInf-style influence with a streaming LoRA daemon (still the research gap — see `datainf-streaming-lora.md`).
