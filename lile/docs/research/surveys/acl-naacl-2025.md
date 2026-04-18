# ACL / NAACL / TACL 2025 — selective loss, personalization, continual SFT

- **Scope**: ACL 2025, NAACL 2025, TACL 2025, Findings, and L2M2 workshop.
- **Compiled**: 2026-04-17 (background agent `a5376dc18269dc8dd`).
- **Filter**: excluded entries already in `sample-efficiency-lit-review.md` (iw-SFT, Rho-1, DFT, GIFT, SFT-GO, DataInf, LESS, GREATS, GAPO, DB-MTL, GCond, CoH, Self-Taught Evaluator, DPO/KTO/IPO, CLoRA).

Tags: **[STRONG]** · **[RELEVANT]** · **[BACKGROUND]** · **[SKIP]**.

---

## Selective / token-level / sequence-level loss weighting

### 1. WIT — On the Effect of Instruction Tuning Loss on Generalization `[STRONG]`
- Khandelwal et al., TACL 2025. <https://aclanthology.org/2025.tacl-1.62/>
- Systematically varies prompt-token vs response-token weights in the SFT objective. Low-to-moderate prompt weight + moderate-to-high response weight beats the standard "mask prompt" recipe across 5 models × 3 datasets × 5 benchmarks, and yields a better starting point for downstream DPO.
- **Why it matters**: directly justifies `lile`'s option to put nonzero weight on prompt/context tokens; simpler knob than Rho-1 excess-loss gating.

### 2. S3FT — Selective Self-to-Supervised Fine-Tuning `[STRONG]`
- Findings NAACL 2025. <https://aclanthology.org/2025.findings-naacl.349/>
- Uses a judge to keep the model's own correct responses and only SFT on the rest. Halves the MMLU / TruthfulQA regression seen under vanilla SFT.
- **Why it matters**: cheap on-policy filter for the LiveLearn replay buffer — skip teaching the model things it already says correctly.

### 3. AlignDistil — Token-Level LM Alignment as Adaptive Policy Distillation `[RELEVANT]`
- ACL 2025 Long. <https://aclanthology.org/2025.acl-long.972/>
- Recasts DPO as token-level KL from an adaptively-extrapolated teacher logit distribution; per-token weights reflect preference strength.
- **Why it matters**: fits `lile`'s selective-loss story with a principled DPO interpretation; complementary to SFT-GO.

## Multi-objective / SFT+DPO balancing

### 4. Balancing the Budget: SFT vs PFT Trade-offs `[STRONG]`
- ACL 2025 Long. <https://aclanthology.org/2025.acl-long.1248/>
- Empirical sweep (100–20k examples/task) of SFT-vs-PFT allocation. SFT dominates low-data regimes; PFT only helps at scale with large models.
- **Why it matters**: directly informs `lile`'s default mixture weights when user feedback is sparse — the daemon's exact regime.

## Critique-and-revise / self-correction

### 5. Confidence vs Critique `[RELEVANT]`
- Yang et al., ACL 2025. <https://aclanthology.org/2025.acl-long.203/>
- Splits self-correction into "confidence" and "critique" axes; reformatting SFT data alone (no RL) beats vanilla SFT on both.
- **Why it matters**: CoH-adjacent; gives a clean SFT-only data recipe `lile` can adopt without adding an RL stage.

### 6. S²R — Self-verify + Self-correct via RL `[BACKGROUND]`
- ACL 2025. <https://aclanthology.org/2025.acl-long.1104/>
- DeepSeek-R1-style multi-turn critique loop trained with RL on math.
- Heavier than `lile`'s target; good pointer for when users explicitly provide revision feedback.

## Online / continual fine-tuning with forgetting mitigation

### 7. GORP — Continual Gradient Low-Rank Projection Fine-Tuning `[STRONG]`
- ACL 2025 Long. <https://aclanthology.org/2025.acl-long.721/>
- Projects per-task LoRA gradients orthogonal to prior-task subspaces. Strong on continual benchmarks.
- **Why it matters**: direct successor to O-LoRA; plausible drop-in `lile` objective alongside CLoRA's A/B regularizer — both cheap, buffer-free.

### 8. HFT — Half Fine-Tuning `[STRONG]`
- ACL 2025 Long. <https://aclanthology.org/2025.acl-long.626/>
- Randomly freezes half the parameters each step; preserves pre-trained knowledge without adapters.
- **Why it matters**: simplest-possible regularizer for task-free continual SFT; implementable as a `lile` optimizer flag in ~20 lines.

### 9. SEE — Sequential Ensemble of Experts `[BACKGROUND]`
- Findings ACL 2025. <https://aclanthology.org/2025.findings-acl.387/>
- Base model + routed per-task LoRA experts.
- Architecturally heavy for `lile`'s single-user daemon; relevant if we later support per-task adapter stacks.

### 10. COPR — Continual Preference Learning with Optimal Policy Regularization `[STRONG]`
- Findings ACL 2025.
- Combines continual learning with preference data by regularizing toward an optimal-policy anchor.
- **Why it matters**: closest published match to `lile`'s core design — online preferences + anti-forgetting in one objective.

## Personalization / few-shot user adaptation

### 11. PROPER — Progressive Learning for Personalized LLMs `[RELEVANT]`
- ACL 2025 Long. <https://aclanthology.org/2025.acl-long.800/>
- Three-stage hierarchical LoRA: population → group (LoRA-MoE) → user-specific adapter.
- **Why it matters**: blueprint for `lile`'s future multi-user mode.

### 12. CHAMELEON — Personalize Your LLM `[RELEVANT]`
- Findings NAACL 2025. <https://aclanthology.org/2025.findings-naacl.407/>
- Self-generates user-specific preference data, then edits representations in two subspaces (personalized / non-personalized) at inference. +40% over baselines on LaMP.
- **Why it matters**: inference-time alternative to LoRA training — low-cost fallback for users with <10 feedback events.

### 13. PLUM — On the Way to LLM Personalization `[STRONG]`
- L2M2 workshop @ ACL 2025. <https://aclanthology.org/2025.l2m2-1.5/>
- Augments user conversations into QA pairs, trains a LoRA with weighted cross-entropy, 81.5% accuracy on 100 convs.
- **Why it matters**: near-identical problem framing to `lile` (sequential conversations, per-user LoRA, weighted CE). Worth a close read for data-augmentation ideas.

## Debiasing / targeted correction

### 14. FairSteer — Inference-Time Debiasing via Dynamic Steering `[RELEVANT]`
- Findings ACL 2025. <https://aclanthology.org/2025.findings-acl.589/>
- Linear probes on activations; subtracts a debiasing steering vector at inference. Bias signatures are >90% linearly separable mid-network.
- **Why it matters**: activation-steering as complement to `lile`'s gradient-based correction — near-instant behavior fixes before a training step has converged.

---

## Synthesis

- **Top additions** to the existing lit review: **WIT** (token weighting), **COPR** (continual + prefs), **PLUM** (per-user LoRA with weighted CE).
- **GORP + HFT** slot next to CLoRA as cheap forgetting mitigators with no replay buffer.
- **Balancing the Budget** is empirical gold for defending `lile`'s SFT-heavy default at low feedback counts.
- EACL 2026 accepted list isn't public yet; the arXiv online-RLHF crop (XPO, SEA, RLTHF) is stronger than anything venue-labeled EACL 2026 so far — flag for the next sweep once the program is posted.
