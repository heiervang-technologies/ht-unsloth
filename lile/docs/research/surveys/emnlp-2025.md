# EMNLP 2025 — user feedback, personalization, RLVR for instruction following

- **Scope**: EMNLP 2025 main + findings; filtered against `lile` citations.
- **Compiled**: 2026-04-17 (background agent `adacd5ff9cd43ba74`).

Tags: **[STRONG]** · **[RELEVANT]** · **[BACKGROUND]** · **[SKIP]**.

---

## Directly actionable for live-training daemon

### 1. User Feedback in Human-LLM Dialogues: A Lens to Understand Users but Noisy as a Learning Signal `[STRONG]`
- Liu, Zhang, Choi (NYU). EMNLP 2025 Main. <https://aclanthology.org/2025.emnlp-main.133/>
- Harvests implicit feedback from WildChat/LMSYS; polarity + content helps on short MTBench-style turns but **hurts** on long WildBench-style queries.
- **Why it matters**: exact "sparse feedback from live users" regime `lile` operates in — empirical warning that naive implicit-feedback SFT is noisy; motivates IW-SFT + KL-guarded objectives.

### 2. FaST — Feature-aware Sampling and Tuning for Personalized Preference Alignment `[STRONG]`
- Thonet et al. (Naver Labs Europe). EMNLP 2025 Main. <https://aclanthology.org/2025.emnlp-main.475/>
- Defines "PPALLI" (personalized preference alignment, limited data per user); introduces DnD + ELIP benchmarks; parameter-efficient feature-aware adapter.
- **Why it matters**: direct per-user LoRA analog of `lile`'s regime; benchmarks are reusable eval harnesses.

### 3. Drift — Decoding-time Personalized Alignments with Implicit User Preferences `[RELEVANT]`
- Findings EMNLP 2025. <https://aclanthology.org/2025.findings-emnlp.324/>
- Training-free baseline: decomposes user preference into weighted attributes, steers frozen LLM at decode with 50–100 examples.
- **Why it matters**: a non-training baseline `lile` must beat. Ablation target: "does LoRA delta outperform decode-time steering at N=100?"

### 4. PRIME — Cognitive Dual-Memory and Personalized Thought Process `[STRONG]`
- EMNLP 2025 Main. <https://aclanthology.org/2025.emnlp-main.1711/>
- Dual memory: episodic (recency-based recall) + semantic (task-oriented LoRA FT). Finds parametric semantic memory beats preference-tuning.
- **Why it matters**: informs `lile`'s `replay_streams`; cold-start results argue for a hybrid retrieval-plus-adapter stance.

### 5. pFedGPT — Hierarchically Optimizing LoRA Aggregation Weights `[BACKGROUND]`
- EMNLP 2025 Main. <https://aclanthology.org/2025.emnlp-main.239/>
- Hierarchical Bayesian optimization over per-module LoRA aggregation for heterogeneous clients.
- Relevant if `lile` ever federates per-user daemons.

## Sample efficiency / selective loss

### 6. LimaCost — Data Valuation for Instruction Tuning `[RELEVANT]`
- Moon et al. (Korea U). Findings EMNLP 2025. <https://aclanthology.org/2025.findings-emnlp.688/>
- Values a datum by how many LIMA points are needed to approximate its gradient; beats NUGGET/SelectIT at a fraction of the cost.
- **Why it matters**: cheap gradient-based valuation as a priority signal for `lile`'s live queue — lighter baseline vs LESS and DataInf.

### 7. Low-Confidence Gold — Refining Low-Confidence Samples `[STRONG]`
- Cai et al. Findings EMNLP 2025. <https://aclanthology.org/2025.findings-emnlp.437/>
- Centroid-clustering + tiny classifier that keeps *low-confidence* samples (hard, informative ones).
- **Why it matters**: inverse of the usual "filter the noise" heuristic — argues `lile` should up-weight low-confidence feedback, not drop it. Testable queue-admission policy.

### 8. MaZO — Masked Zeroth-Order Optimization for Multi-Task FT `[BACKGROUND]`
- Zhang et al. (UCSB/Amazon). EMNLP 2025 Main. <https://aclanthology.org/2025.emnlp-main.935/>
- Addresses gradient-variance + collinearity in multi-task ZO; cites CoBa (EMNLP 2024) as prior.
- Relevant if `lile` ever trains multiple objectives concurrently under memory pressure.

## Critique / RLVR / reasoning

### 9. MultiCritique — Training LMs to Critique With Multi-agent Feedback `[RELEVANT]`
- Lan et al. (Shanghai AI Lab). Findings EMNLP 2025. <https://aclanthology.org/2025.findings-emnlp.78/>
- Multi-agent critique aggregation → SFT → RL; introduces ACU (atomic flaw unit).
- **Why it matters**: CoH/Constitutional-AI descendant producing *structured* critiques; ACU is a reasonable unit for `lile`'s critique-and-revise objective.

### 10. No Need for Explanations — LLMs Learn from Mistakes In-Context `[STRONG]`
- EMNLP 2025 Main. <https://aclanthology.org/2025.emnlp-main.1686.pdf>
- LLMs leverage labeled-wrong examples in-context *without* explanations; matches or beats explicit-critique prompting.
- **Why it matters**: `lile`'s critique channel may not need full NL rationales — a simple "this was wrong" + context could suffice. Strong motivation for an ablation.

### 11. VerIF — Verification Engineering for RL in Instruction Following `[RELEVANT]`
- Peng et al. (THU-KEG). EMNLP 2025 Main. <https://aclanthology.org/2025.emnlp-main.1542/>
- Hybrid rule-based code + QwQ-32B LLM verifier; releases VerInstruct (~22k items); GRPO training.
- **Why it matters**: EMNLP 2025's cleanest RLVR-for-instruction-following recipe. Verifier-construction blueprint if `lile` adds an RL phase.

## Memory / retrieval

### 12. Awesome-RAG-Reasoning (resource paper) `[BACKGROUND]`
- EMNLP 2025 resource track. <https://github.com/DavidZWZ/Awesome-RAG-Reasoning>
- Use as pointer hub into HippoRAG-2 / episodic-RAG literature.

## Negative findings
- No EMNLP 2025 paper on **true online continual SFT from streaming feedback** beyond Liu-Zhang-Choi (#1).
- Continual-learning tutorial <https://aclanthology.org/2025.emnlp-tutorials.7/> surveys space but no new methods.
