# 2026 Synthesis: Test-Time RL, Self-Improvement, o1/R1 Lineage, PRM vs ORM

Prepared for Mei (SVP AI Research). Scope: 2025-2026 literature on reasoning training methods relevant to `lile`. Overlaps with `sample-efficiency-lit-review.md` §5 (GRPO, V-STaR, ReST-EM, AReaL, AsyncFlow, Asynchronous RLHF) are deliberately skipped.

---

## 1. Test-Time RL / RL at inference

**Key papers**
- TTRL: Test-Time Reinforcement Learning — Zuo et al., arXiv:2504.16084 (NeurIPS 2025). https://arxiv.org/abs/2504.16084
- T3RL: Tool Verification for Test-Time RL — arXiv:2603.02203 (2026). https://arxiv.org/abs/2603.02203
- TTC-RL: Learning on the Job (Test-Time Curricula) — arXiv:2510.04786 (ICLR 2026). https://arxiv.org/abs/2510.04786
- Unsupervised RLVR (URLVR) — ICLR 2026 (see TTRL-lineage survey above).
- Amplification Effects in Test-Time RL — MERL TR2026-020 (AAAI 2026). https://www.merl.com/publications/docs/TR2026-020.pdf

**2026 synthesis.** TTRL reliably works: majority-vote pseudo-rewards + GRPO lift Qwen-2.5-Math-7B pass@1 on AIME 2024 by ~211% with *zero* labels. But three failure modes are now well-documented: (a) "false-popular mode collapse" — the majority can be wrong, and RL then reinforces the error (fixed by T3RL's tool-verifier reward shaping, +31.6% on AIME 2024 over TTRL); (b) jailbreak amplification — adversarial test-time data shifts weights in unsafe directions; (c) majority-vote signal degenerates outside verifiable domains (VLAs, open-ended QA). SFT-based TTT has an "initial drop" problem; on-policy RL-TTT is the default in 2026. The honest read: TTRL is real on math/code with self-consistency; everything else is fragile.

**Implication for lile.** Our daemon already trains online with KL-scoped updates — that's TTRL in all but name *if we accept majority-vote pseudo-labels*. Two concrete changes worth considering: (1) add a *verifier-gated reward* path (tools, unit tests, grader model) before committing a sample to the replay stream, mirroring T3RL; (2) add a consensus-drift monitor so the daemon aborts an update when majority answers collapse to a low-entropy attractor (false-popular detector).

---

## 2. Self-improvement loops (generate-then-train)

**Key papers**
- rStar-Math — Guan et al., arXiv:2501.04519 (ICML 2025). https://arxiv.org/abs/2501.04519
- Absolute Zero Reasoner (AZR) — Zhao et al., arXiv:2505.03335 (NeurIPS 2025). Note: the 2504.01441 ID Mei cited does not resolve. https://arxiv.org/abs/2505.03335
- Meta-Rewarding LMs — Wu et al., EMNLP 2025. https://aclanthology.org/2025.emnlp-main.583/
- CREAM: Consistency-Regularized Self-Rewarding — ICLR 2025. (OpenReview Vf6RDObyEF)
- "Superficial Self-Improved Reasoners" — arXiv:2503.02103 (caveat paper). https://arxiv.org/abs/2503.02103

**2026 synthesis.** rStar-Math (Qwen2.5-Math-7B: 58.8% → 90.0% MATH via 4 rounds of MCTS + PPM, beats o1-preview at 7B) and AZR (zero external data, code-executor-verified self-play, SOTA coding/math) are the two flagship wins. Both are *verifiable-domain* stories — code execution or symbolic math. Self-Rewarding LMs (Yuan 2024) saturated in 2-3 iterations; Meta-Rewarding and CREAM both diagnose this as judge-quality stagnation and reward-bias accumulation, fixed by meta-judging and cross-iter consistency. The "Superficial Self-Improved Reasoners" paper (2503.02103) warns that SFT on self-generated traces often memorizes rather than generalizes; model-merging mitigates. **Does it work beyond math/code? Weakly.** Logic-RL transfers to math (+125% AIME); Microsoft's CoR shows math-trained models help science/code. Open-ended domains still lack a verifier, which is the whole bottleneck.

**Implication for lile.** The daemon's current loop is generator-only. Biggest ROI would be adding a V-STaR-style DPO verifier trained on the daemon's *own* correct/incorrect rollouts — cheap (DPO is sample-efficient), uses discarded negatives, and gives us a Best-of-N filter for serving. For any non-verifiable user domain, do not ship a self-rewarding loop without CREAM-style consistency checks.

---

## 3. o1 / DeepSeek-R1 lineage — sample-efficient variants

**Key papers**
- DeepSeek-R1 — arXiv:2501.12948 (Nature 2026). https://arxiv.org/abs/2501.12948
- Kimi k1.5 — arXiv:2501.12599. https://arxiv.org/abs/2501.12599
- Sky-T1-32B / Sky-T1-mini — NovaSky (Berkeley), Jan-Feb 2025 (~$450 / $870 replications). https://novasky-ai.github.io/posts/sky-t1/
- 1-shot RLVR — Wang et al., arXiv:2504.20571. https://arxiv.org/abs/2504.20571
- 1-shot Critique Fine-Tuning (CFT) — arXiv:2506.03295. https://arxiv.org/abs/2506.03295

**2026 synthesis.** The DeepSeek-R1 recipe (GRPO + verifiable reward + optional cold-start SFT) is settled canon; Nature publication in 2026 made it the reference. R1's distillation finding matters: *distilling R1 into Qwen2.5-32B beats running RL directly on the 32B*. Kimi k1.5 independently confirmed "long-context RL + simple GRPO is enough — skip MCTS, value functions, PRMs." For sample-efficient variants: S1/LIMO (~1000 examples) → 1-shot RLVR (literally one example gets Qwen2.5-Math-1.5B ~30% AIME via policy gradient + entropy bonus + "post-saturation generalization") → 1-shot CFT (one problem, many critiques, +15% absolute on math benches in 5 GPU-hours). Reproducibility caveat (Hochlehnert 2025): many RLVR wins are single-seed and may not survive proper eval.

**Implication for lile.** The 1-shot RLVR result directly endorses the daemon's thesis — a handful of high-signal verifiable examples can move a policy substantially. Concretely: (a) keep the entropy-bonus / exploration term in our GRPO objective (1-shot RLVR identifies this as critical); (b) seriously consider a distillation fallback mode — if a user has a strong teacher, distilling is more sample-efficient than on-policy RL on small bases per R1; (c) add multi-seed eval to our harness before claiming wins.

---

## 4. PRMs vs ORMs — sample-efficiency empirics

**Key papers**
- PAVs (Process Advantage Verifiers) — Setlur et al., ICLR 2025 Spotlight, arXiv:2410.08146. https://arxiv.org/abs/2410.08146
- GRPO is Secretly a Process Reward Model — arXiv:2509.21154 (v3 Feb 2026). https://arxiv.org/abs/2509.21154
- Lessons of Developing PRMs in Math Reasoning — arXiv:2501.07301. https://arxiv.org/abs/2501.07301
- PRIME (implicit PRM) — arXiv:2502.01456. https://arxiv.org/abs/2502.01456
- Survey of PRMs — arXiv:2510.08049. https://arxiv.org/abs/2510.08049

**2026 synthesis.** Two results have collapsed the PRM hype: (1) PAVs showed that *properly formulated* PRMs (advantage-shaped, not step-correctness) give 6× sample efficiency and 8× Pass@N over ORMs — so PRMs can work. (2) "GRPO is Secretly a PRM" proved theoretically (and λ-GRPO confirmed empirically) that vanilla GRPO + ORM performs implicit sub-trajectory credit assignment whenever rollouts share prefixes. Combined with PRM downsides — annotation cost, reward hacking, online-update awkwardness — the field has largely dropped explicit PRMs for reasoning RL. Kimi k1.5 explicitly skipped PRMs. The exception: agentic / RAG / tool-use pipelines (ReasonRAG), where prefix overlap is weak so GRPO's free PRM vanishes — there, explicit step rewards still help.

**Implication for lile.** Don't build a PRM. GRPO + a clean ORM is the 2026-consensus default and matches what `lile/objectives/` is already structured around. If we later add agentic / multi-tool objectives where rollouts diverge early, revisit PRMs then. Consider λ-GRPO as a drop-in upgrade — small algorithmic tweak, reportedly faster convergence.

---

## TL;DR for the daemon

1. Add a verifier-gated reward path (T3RL-style) + false-popular drift monitor — protects online-RL loop from self-reinforcing errors.
2. Add V-STaR-style DPO verifier trained on daemon rollouts — turns discarded negatives into Best-of-N filter.
3. Keep entropy bonus in GRPO; 1-shot RLVR confirms it's load-bearing for small-data regimes.
4. Do *not* build a PRM. GRPO+ORM already does process credit assignment.
5. Ship distillation fallback — R1 shows it's more sample-efficient than on-policy RL for smaller bases.
