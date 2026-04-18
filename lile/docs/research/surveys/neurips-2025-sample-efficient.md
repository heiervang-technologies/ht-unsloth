# NeurIPS 2025 — sample-efficient LLM learning

- **Scope**: accepted NeurIPS 2025 papers covering test-time training, self-improvement, continual learning, LoRA forgetting, off-policy RL.
- **Compiled**: 2026-04-17 (background agent `a290d4e5a7d7cabd5`).
- **Filter**: excluded anything already cited in `sample-efficiency-lit-review.md` or `optimizer-sample-efficiency.md`.

Twelve papers, grouped by theme. Tags: **[STRONG]** = changes design / opens PR; **[RELEVANT]** = informs existing decision; **[BACKGROUND]** = context only.

---

## Test-time training / test-time RL

### 1. SEAL — Self-Adapting Language Models `[STRONG]`
- Zweiger, Pari et al. (MIT). arXiv:2506.10943. [NeurIPS poster 118690](https://neurips.cc/virtual/2025/poster/118690). [Code](https://github.com/Continual-Intelligence/SEAL).
- The model emits "self-edits" — synthetic training data + hyperparameters + LoRA configs — and an outer RL loop (ReST-EM) rewards edits that improve downstream accuracy. Knowledge-incorporation jumps 32.7%→47.0%. Reports **catastrophic forgetting across repeated self-edits**.
- **Why it matters for `lile`**: this is almost literally the daemon's loop — a model deciding how to turn a feedback event into a LoRA update. Mine the RL-on-edits formulation and the forgetting mitigation notes.

### 2. TTRL — Test-Time Reinforcement Learning `[STRONG]`
- PRIME-RL. [NeurIPS poster 117645](https://neurips.cc/virtual/2025/loc/san-diego/poster/117645). [Code](https://github.com/prime-rl/ttrl).
- Majority-vote over sampled rollouts as a pseudo-reward for RL on unlabeled inference-time data. +211% pass@1 on AIME24 for Qwen-2.5-Math-7B.
- **Why it matters for `lile`**: direct blueprint for reward-model-free online RL when explicit user feedback is sparse. Majority-vote surrogate could fill the 10–100 req/hr gaps.

## Self-improvement / self-play

### 3. Absolute Zero — Reinforced Self-play Reasoning with Zero Data `[RELEVANT]`
- Zhao et al. [NeurIPS poster 116121](https://neurips.cc/virtual/2025/loc/san-diego/poster/116121). arXiv:2505.03335.
- Single model proposes and solves its own tasks under RLVR; no curated question set.
- **Why it matters**: covers dead periods when no user feedback arrives — the daemon can self-propose practice tasks to stay warm without drift.

### 4. ExIt — Bootstrapping Task Spaces for Self-Improvement `[RELEVANT]`
- Minqi Jiang (Meta). arXiv:2509.04575.
- RL fine-tunes for multi-step inference self-improvement via only single-step tasks; Prioritized Level Replay on the task buffer with evolutionary mutation.
- **Why it matters**: curriculum mechanism for picking which stored feedback events to replay next — maps onto `lile`'s replay stream design.

### 5. CoVo — Consistent Paths Lead to Truth `[STRONG]`
- [NeurIPS poster 117063](https://neurips.cc/virtual/2025/poster/117063).
- Intrinsic reward from consistency + volatility of intermediate reasoning states across rollouts. Matches supervised RL without labels.
- **Why it matters**: drop-in intrinsic reward for KL/GRPO-style objectives when the user hasn't rated a response. Fills the reward-model-free slot.

## Continual learning / forgetting

### 6. Nested Learning — Hope architecture `[BACKGROUND]`
- Behrouz, Mirrokni et al. (Google). [NeurIPS poster 116123](https://neurips.cc/virtual/2025/poster/116123). [Google blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/).
- Reframes optimizers as associative-memory modules at distinct update frequencies. 98% retention on multi-task CL vs 70% baseline.
- Architecture-level insight — not a drop-in for a LoRA daemon, but the multi-timescale optimizer perspective could inform a slow-fast LoRA split.

### 7. SuRe — Surprise-Driven Prioritised Replay for Continual LLM Learning `[STRONG]`
- arXiv:2511.22367. Prioritize replay on high-loss / task-boundary sequences; Dual Learner + Surprise Replay hits 11.63 avg perplexity on M2D2, beating prior SOTA by >5pp.
- **Why it matters**: direct replacement/upgrade for `lile`'s current `replay_streams`. DER++ is already cited in the lit review; SuRe is the 2025 evolution.

### 8. GainLoRA — Gated Integration of LoRA for Continual Learning `[RELEVANT]`
- Nanjing U., NeurIPS 2025.
- Gating module over per-task LoRA branches with init/update constraints, preventing new branches from interfering on old tasks.
- **Why it matters**: architectural pattern for per-objective LoRA stacking — gating beats vanilla additive merging.

### 9. LoRA vs Full Fine-tuning: An Illusion of Equivalence `[STRONG]`
- [NeurIPS poster 115207](https://neurips.cc/virtual/2025/poster/115207).
- Identifies "intruder dimensions" — novel high-rank singular vectors unique to LoRA training — and shows forgetting is **causally localized to them**; scaling them down recovers pretraining behavior.
- **Why it matters**: concrete diagnostic + mitigation for LoRA forgetting in long-running daemons. Could become a metric in the Studio snapshots tab.

## Preference learning

### 10. RePO — Preference Learning through ReLU Optimization `[RELEVANT]`
- arXiv:2503.07426 / NeurIPS 2025.
- Replaces sigmoid/log-sigmoid in DPO-family with a binary ReLU threshold; convex envelope of the 0-1 loss. Competitive with SimPO at lower complexity.
- **Why it matters**: current `lile` objective list stops at KTO/IPO; RePO is the 2025 addition worth evaluating for pairwise user-feedback events.

## Off-policy / data-efficient RL

### 11. Difficulty-Targeted Online Data Selection + Rollout Replay `[STRONG]`
- Sun, Shen et al. arXiv:2506.05316.
- Attention-based adaptive difficulty estimator + experience replay of recent rollouts. 23–62% wall-clock reduction vs vanilla GRPO at equal quality.
- **Why it matters**: directly lowers the compute floor for `lile`'s GRPO path — our exact regime (sparse prompts, expensive rollouts).

### 12. Offline RL by Reward-Weighted Fine-Tuning for Conversation Optimization `[RELEVANT]`
- [NeurIPS poster 117620](https://neurips.cc/virtual/2025/loc/san-diego/poster/117620).
- Offline RL as reward-weighted SFT for short-horizon dialog policies.
- **Why it matters**: lightweight alternative to full PPO/GRPO for the conversational-feedback shape — candidate objective for `lile/objectives/`.

---

## Synthesis

Must-reads: **SEAL** (architectural twin of `lile`) and **SuRe** (immediate upgrade path for replay). **TTRL + CoVo** together cover the reward-model-free regime when feedback is silent. **RePO** belongs on the preference-objective short list alongside existing KTO/IPO. **Intruder-dim mitigation** from "Illusion of Equivalence" is a mechanical fix that can ship as a diagnostic first, mitigation second. **Nested Learning / Hope** is a longer-horizon bet — read it for the frame, not a direct port.

## Sources

- [NeurIPS 2025 Recap — Amplify Partners](https://www.amplifypartners.com/blog-posts/neurips-2025-recap)
- [TTRL](https://neurips.cc/virtual/2025/loc/san-diego/poster/117645) · [SEAL](https://arxiv.org/abs/2506.10943) · [Absolute Zero](https://neurips.cc/virtual/2025/loc/san-diego/poster/116121) · [ExIt](https://www.arxiv.org/pdf/2509.04575) · [CoVo](https://neurips.cc/virtual/2025/poster/117063) · [Nested Learning](https://neurips.cc/virtual/2025/poster/116123)
- [SuRe](https://www.arxiv.org/pdf/2511.22367) · [GainLoRA](https://cs.nju.edu.cn/lm/en/post/2025-10-11-neurips-2025-accepted-papers/index.html) · [Illusion of Equivalence](https://neurips.cc/virtual/2025/poster/115207)
- [RePO](https://arxiv.org/pdf/2503.07426) · [Difficulty-Targeted Selection](https://arxiv.org/abs/2506.05316) · [Offline RL for Conversation](https://neurips.cc/virtual/2025/loc/san-diego/poster/117620)
