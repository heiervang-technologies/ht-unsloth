# Sample-Efficiency Literature Review for `lile`

Scope: non-optimizer levers for a live-training LoRA daemon doing online continual learning from sparse user feedback (hours-to-days uptime, Qwen3-class base). Optimizer choice (Lion8bit, Muon, ScheduleFree) is covered separately.

Year focus: 2024–2026. Foundational pre-2024 work cited only when load-bearing.

---

## 1. Replay buffer design for continual LoRA

**Key papers**

- **SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning** (Hazard et al., arXiv 2511.22367, Nov 2025) — picks the most "surprising" (high-loss) sequences before each task; pairs a fast LoRA head with an EMA-updated slow head. <https://arxiv.org/abs/2511.22367>
- **FOREVER: Forgetting-Curve-Inspired Memory Replay for LM Continual Learning** (Feng et al., arXiv 2601.03938, Jan 2026) — schedules replay against a model-centric forgetting curve rather than uniformly; outperforms MixReplay / fixed-interval replay on standard CL benchmarks. <https://arxiv.org/abs/2601.03938>
- **Scalable Strategies for Continual Learning with Replay** (Hickok, arXiv 2505.12512, May 2025) — uses CE on current-task samples and a DER++-with-KD variant on replay (logit standardization + traditional KD instead of raw-logit MSE). <https://arxiv.org/html/2505.12512v1>
- **Combining Replay and LoRA for Continual Learning in NLU (ERI-LoRA)** (Computer Speech & Language, 2024) — class-balanced replay sample selection on top of LoRA on Llama2; small fixed buffers, adaptive class-distribution weighting. <https://www.sciencedirect.com/science/article/abs/pii/S0885230824001207>
- **Improvements to Dark Experience Replay & Reservoir Sampling** (arXiv 2504.20932, 2025; Frontiers in AI 2026) — diagnoses DER's plasticity collapse: reservoir gradually stops accepting new data; proposes adaptive DER weights, blocking inconsistent replays, stratified multi-buffer reservoir. <https://arxiv.org/abs/2504.20932>
- *(Foundational)* **DER / DER++** (Buzzega et al., NeurIPS 2020) — reservoir sampling + logit-distillation replay; still the strongest single-paragraph baseline. <https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf>
- **Wang-ML-Lab CL-of-LLMs Survey** (CSUR 2025) — taxonomy of replay strategies for LLM CL. <https://github.com/Wang-ML-Lab/llm-continual-learning-survey>

**Synthesis.** The 2024–26 literature has *not* converged on a single replay scheme, but it has converged on three principles: (a) plain uniform reservoir is a strong, hard-to-beat baseline but suffers a documented "plasticity collapse" once the stream is long, (b) DER/DER++-style logit replay (regularize current logits to past logits on buffer items) consistently beats label-only replay because it preserves behaviour rather than just labels, and (c) prioritization helps if the priority is *informative* — surprise/loss (SuRe), Maximally-Interfered-Retrieval, gradient coreset, or class-balance — but uniform priority on shuffled streams is the floor everyone benchmarks against. For a live-training daemon, the most defensible default is reservoir + DER++-with-KD on logits, with optional surprise- or MIR-based oversampling once the buffer crosses some size threshold.

**Where the literature is thin/contradictory.** Almost all "continual LoRA + replay" papers are evaluated on *task-incremental* benchmarks (Standard CL Benchmark, SuperNI splits) — not on *streaming* user-feedback. There is no paper I could find that benchmarks replay schemes on a real interactive feedback stream from a deployed model. The DER++ improvement paper (2504.20932) explicitly shows reservoir's flaw under sustained streams, but its fix is benchmarked on RL/regression toys, not LLMs.

---

## 2. Catastrophic forgetting in LoRA adapters

**Key papers**

- **Hierarchical Layer-Wise and Element-Wise Regularization for LLM Finetuning** (arXiv 2501.13669, Jan 2025) — Synaptic-Intelligence-style importance scoring layered with element-wise penalties; specifically targets the LoRA-adapter forgetting case. <https://arxiv.org/abs/2501.13669>
- **Online-LoRA: Task-Free Online Continual Learning via LoRA** (Wei et al., WACV 2025) — explicitly task-free (no task boundaries), drift-detection-triggered LoRA expansion; finds rigid regularizers (EWC++, A-GEM, LODE) underperform in online settings. <https://openaccess.thecvf.com/content/WACV2025/papers/Wei_Online-LoRA_Task-Free_Online_Continual_Learning_via_Low_Rank_Adaptation_WACV_2025_paper.pdf>
- **O-LoRA: Orthogonal Subspace Learning for LM Continual Learning** (Wang et al., EMNLP Findings 2023) — constrain new-task LoRA gradient to be orthogonal to past-task LoRA subspace; the workhorse baseline everyone cites. <https://arxiv.org/abs/2310.14152>
- **CLoRA: Controlled LoRA with Subspace Regularization** (ACL 2025) — adds an orthogonal loss between A and B matrices to the LM loss; cheap, no buffer, no Fisher. <https://aclanthology.org/2025.acl-long.940.pdf>
- **EWC-LoRA: Adapting EWC for PTM-Based Continual Learning** (arXiv 2602.17559, 2026) — argues prior EWC+LoRA combinations regularize each low-rank module separately and get the Fisher wrong; proposes full-dimensional Fisher with shared low-rank update. <https://www.arxiv.org/pdf/2602.17559>
- **OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting** (arXiv 2510.13003, 2025) — orthogonal projection variant; another point in the orthogonal-subspace family. <https://arxiv.org/html/2510.13003>

**Synthesis.** For continual LoRA, the field has roughly converged on a layered defence: (1) LoRA itself is *already* a regularizer (smaller subspace = smaller drift; "LoRA Learns Less and Forgets Less" formalizes this), (2) orthogonal-subspace methods (O-LoRA, CLoRA, OPLoRA) are the cheapest active defence — no buffer, no Fisher, just an orthogonality penalty between current and frozen prior LoRAs, and (3) Fisher/EWC-style penalties are *making a comeback* (EWC-LoRA, hierarchical regularization 2501.13669) once people figured out how to estimate the Fisher correctly under low-rank updates. Online-LoRA (WACV 2025) is the most relevant single paper for `lile` because it is explicitly *task-free* and finds that rigid EWC/A-GEM constraints hurt; its recipe is drift-detection + adapter expansion, not regularization.

**Where the literature is thin/contradictory.** Online-LoRA says rigid regularization hurts; EWC-LoRA says rigid regularization works if you do Fisher right. These are not yet reconciled. Also, every continual-LoRA paper I found assumes *discrete* task boundaries or at minimum a clear distribution-shift event; nobody has a clean recipe for "infinite stream of small drifts from heterogeneous user feedback," which is what `lile` actually faces. The LoRA-delta-magnitude-cap idea (literally clamp the L2 norm of `B @ A`) is widely deployed in practice (LoRA-XS, weight-decay-on-BA in many trainers) but I could not find a 2024–26 paper that systematically studies it as a forgetting mitigation.

---

## 3. Per-sample loss weighting for SFT-from-feedback

**Key papers**

- **Importance-Weighted SFT (iw-SFT)** — Qin & Springenberg, "Supervised Fine-Tuning on Curated Data is RL (and can be improved)", arXiv 2507.12856, Jul 2025 — sequence-level importance weights make SFT optimize a tighter lower bound to the RL objective; trivially extends to quality-scored data. Hits 66.7% on AIME 2024. <https://arxiv.org/abs/2507.12856>
- **Rho-1 / Selective Language Modeling** (Lin et al., 2024) — token-level: train only on "useful" tokens identified by a reference model's excess loss. <https://arxiv.org/abs/2404.07965>
- **DFT: Dynamic Fine-Tuning** (Wu et al., 2025) — dynamically rescales token-level objectives to stabilize gradients; cited as a key recent token-weighting recipe.
- **GIFT: Guided Importance-Aware FT** (arXiv 2509.20863) — entropy-of-the-model's-own-distribution as the per-token weight; concentrate compute on uncertain tokens. <https://arxiv.org/html/2509.20863>
- **SFT-GO: SFT with Group Optimization** (arXiv 2506.15021, 2025) — group tokens by importance (TF-IDF, LLMLingua-2, or Rho-1's reference-model excess loss) and apply a min/max objective across groups. <https://arxiv.org/html/2506.15021v1>
- **Prompt-Loss-Weight study** (Huerta-Enochian et al., 2024, arXiv 2401.13586) — empirical: PLW matters for short-completion data and basically doesn't matter for medium/long completions; small non-zero PLW acts as anti-drift regularization. <https://arxiv.org/html/2401.13586v2/>

**Influence-functions for sample weighting.**
- **LESS** (Xia et al., ICML 2024) — gradient-similarity-to-validation for instruction-tuning data selection; explicitly notes the long-sequence-penalty bias. <https://arxiv.org/abs/2402.04333>
- **DataInf** (Kwon et al., ICLR 2024) — efficient closed-form influence approximation for LoRA-tuned LLMs (computes per-sample influence in seconds, not hours). <https://arxiv.org/abs/2310.00902>
- **GREATS: Online Selection of High-Quality Data** (Wang et al., NeurIPS 2024) — per-iteration online influence-style selection during training. <https://arxiv.org/abs/2405.16089>
- **In2Core** (EMNLP Findings 2024) — coreset selection via influence functions for instruction tuning. <https://aclanthology.org/2024.findings-emnlp.604.pdf>
- **"Do Influence Functions Work on Large Language Models?"** (arXiv 2409.19998, 2024) — sober evaluation: classical IF assumptions break on deep LMs; works *better* on LoRA-tuned models because the parameter space is small enough for tractable iHVP. <https://arxiv.org/abs/2409.19998>

**Synthesis.** For SFT-on-user-rewrites the live menu is: (1) sequence-level — iw-SFT-style importance weighting where the weight is some user-feedback-derived quality score (rating, accept/reject, dwell time); (2) token-level — Rho-1 / DFT / GIFT, all of which boil down to "down-weight tokens the model already gets right or that are catastrophically wrong, up-weight uncertain tokens"; (3) influence-function-driven — DataInf is the only one that's tractable enough on LoRA to plausibly run online. On *clipping*, the only solid theory is from iw-SFT: the importance ratio needs PPO-style clipping to keep the lower bound tight; uncapped importance weights blow up variance. There is no published prescription for the magnitude of the clip in the SFT-from-NL-feedback setting — copy PPO's 0.2 as a default and tune.

**Where the literature is thin/contradictory.** None of the per-sample weighting papers are evaluated under continual-LoRA / streaming conditions. PLW says prompt loss usually doesn't matter; DFT says token weighting matters a lot; both are run on static SFT corpora. Influence functions on LLMs are still actively debated: 2409.19998 is the most honest read and concludes "they work in narrow conditions, not as a black box."

---

## 4. Objective mixing (SFT + KTO + CoH + hinge + contrastive at heterogeneous scales)

**Key papers**

- **Gradient-Adaptive Policy Optimization (GAPO)** (arXiv 2507.01915, 2025) — multiple-gradient-descent (MGDA) for *multi-objective LLM alignment*; the closest 2025 work to lile's actual problem. <https://arxiv.org/abs/2507.01915>
- **DB-MTL: Dual-Balancing for Multi-Task Learning** (arXiv 2308.12029, 2024 update) — log-transform on losses + gradient balancing; reports outperforming PCGrad / GradVac / IMTL-G / CAGrad / Nash-MTL / Aligned-MTL when each is paired with the same log transform. <https://arxiv.org/abs/2308.12029>
- **GCond: Gradient Conflict Resolution via Accumulation-Based Stabilization** (arXiv 2509.07252, 2025) — accumulation-based fix for the "tragic triad" (conflicting directions + magnitude gap + high curvature) that breaks PCGrad/CAGrad on transformer-scale models. <https://arxiv.org/html/2509.07252v1>
- **Beyond Losses Reweighting** (Phan et al., ICCV 2025) — flat-minima-aware multi-task balancing; combines loss gradients with sharpness-aware gradients to find joint flat low-loss regions. <https://openaccess.thecvf.com/content/ICCV2025/papers/Phan_Beyond_Losses_Reweighting_Empowering_Multi-Task_Learning_via_the_Generalization_Perspective_ICCV_2025_paper.pdf>
- **Gradient-Based Multi-Objective Deep Learning Survey** (arXiv 2501.10945, 2025) — taxonomy: gradient-weighting (GradNorm-family) vs gradient-manipulation (PCGrad/GradVac/GradDrop). <https://arxiv.org/abs/2501.10945>
- *(Foundational)* **GradNorm** (Chen et al., 2018) — normalizes gradient magnitudes via dynamic per-task loss scalars. <https://arxiv.org/abs/1711.02257>
- *(Foundational)* **PCGrad** (Yu et al., NeurIPS 2020) — project conflicting gradients onto each other's normal plane. <https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf>

**Synthesis.** The honest 2024–26 verdict is: **gradient-surgery methods (PCGrad, CAGrad, GradVac, MGDA) are theoretically appealing but practically expensive on transformer-scale models** because they need per-task backward passes and per-task gradient storage. The dominant practical recipe in production LLM training remains *running-mean loss scalars* — track an EMA of each objective's loss magnitude and divide each loss by its EMA before summing — sometimes with a log transform (DB-MTL's empirical contribution). For a daemon mixing SFT + KTO + CoH + hinge, the defensible default is: (1) put each objective on its own EMA-normalized scalar, (2) run a log-transform if any scalar is over an order of magnitude bigger than the others, (3) consider per-objective LR groups (cheap, just `param_groups` in PyTorch) rather than per-objective gradient surgery. Reach for PCGrad/CAGrad only if you've measured destructive interference (negative cosine between objective gradients).

**Where the literature is thin/contradictory.** None of the gradient-surgery papers are run on *interactive RLHF-style* setups; they're run on multi-task vision (NYU-v2, CityScapes) or static multi-task NLP. GAPO is the closest analogue but it targets multi-preference DPO, not the SFT/KTO/CoH/hinge cocktail. Also, all the "X is better than PCGrad" papers compare on different backbones, so the rank-ordering is not stable across studies.

---

## 5. Verifiable-reward online learning

**Key papers**

- **RLVR / GRPO** — DeepSeekMath / DeepSeek-R1 (Shao et al., 2024) is the canonical recipe: PPO-style clipping + explicit KL-to-reference + group-relative advantage; binary correctness reward. <https://arxiv.org/abs/2402.03300>
- **GRPO theoretical analysis** (arXiv 2503.06639, 2025) — formal "success amplification" dynamics; shows when GRPO sharpens the existing distribution vs adds new modes. <https://arxiv.org/html/2503.06639v4>
- **RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs** (arXiv 2506.14245, 2025) — Pass@K rises after RLVR even though prior work claimed RLVR only improves sampling efficiency; introduces CoT-Pass@K. <https://arxiv.org/abs/2506.14245>
- **V-STaR: Training Verifiers for Self-Taught Reasoners** (Hosseini et al., COLM 2024) — uses *both* correct and incorrect rollouts: correct ones train the generator (STaR), pairs train a DPO verifier; +4–17% over plain STaR. <https://arxiv.org/abs/2402.06457>
- **ReST-EM** (Singh et al., 2023; widely cited 2024) — formalizes STaR-style filtering as approximate EM; a recent 2025 paper (arXiv 2512.20169) makes the EM derivation rigorous. <https://arxiv.org/abs/2308.08998>
- **AReaL: Asynchronous RL for Language Reasoning** (arXiv 2505.24298, 2025) — fully async streaming RL: rollout workers continuously generate; trainer consumes whatever's ready, with bounded staleness via off-policy correction. 2.77× speedup over sync. <https://arxiv.org/abs/2505.24298>
- **AsyncFlow** (arXiv 2507.01663, 2025) — delayed-parameter-update streaming; lets actors keep generating with old weights for one step while new weights load. <https://arxiv.org/abs/2507.01663>
- **Asynchronous RLHF** (Noukhovitch et al., ICLR 2025) — the foundational "decouple gen and train" paper; quantifies the off-policy-correction tax. <https://openreview.net/pdf?id=FhTAG591Ve>
- **awesome-RLVR** — running curated bibliography. <https://github.com/opendilab/awesome-RLVR>

**Synthesis.** GRPO is the de-facto online RLVR algorithm in 2025. For a *streaming* variant: AReaL and AsyncFlow are the two most relevant 2025 systems papers — they show that GRPO survives bounded staleness (off-policy correction + KL-to-reference together keep the policy on-distribution) and that you can get most of the wallclock win without giving up convergence. For lile-style sparse verifiable-reward feedback ("user accepted the answer", "tests passed", "math answer matched"), the cleanest streamable recipe is: (1) GRPO-style group-relative advantage on whatever batch you can assemble, (2) explicit KL-to-frozen-reference-LoRA to bound drift, (3) AReaL-style off-policy importance correction if rollouts are stale, (4) optional V-STaR-style verifier on the side to re-rank candidates at inference. ReST-EM and STaR are good *offline-rejection-sampling* recipes — they fit the "wait until you have enough confirmed-correct rollouts, then SFT a batch" pattern, which may suit `lile`'s low-throughput stream better than full GRPO.

**Where the literature is thin/contradictory.** There is genuine, current debate over whether RLVR teaches new behaviors or just resamples the base model's distribution (the Tsinghua "efficient samplers" critique vs. the 2506.14245 Pass@K result). For a daemon designed to learn *new* behaviors from sparse feedback, this debate matters. Also: none of the async-streaming RL papers are evaluated under hours-of-uptime, sub-1Hz feedback rates — they all assume high-throughput rollout farms. The ReST-EM streaming variant is a natural research direction but I can't find a published one.

---

## 6. Learning from natural-language critique (semantic feedback)

**Key papers**

- **Chain of Hindsight (CoH)** (Liu, Sferrazza, Abbeel, ICLR 2024) — concatenate `[bad answer] [critique] [good answer]` and SFT the model to produce the good answer conditioned on the bad one + critique; then at inference prepend a "good" tag. <https://arxiv.org/abs/2302.02676>
- **Constitutional AI / RLAIF** (Bai et al., 2022; widely deployed in 2024) — model self-critiques + revises against natural-language principles; revisions become SFT data, then RLAIF for the polish. <https://arxiv.org/abs/2212.08073>
- **Self-Taught Evaluator** (Wang et al., Meta AI, Oct 2024) — model-generated critiques used as preference signal; default RLAIF recipe in 2024.
- **RLHF-Book Ch. 13 — CAI & AI Feedback** (Lambert, 2024–25) — current-best-practice guide; explicit on the LLM-as-judge calibration risk. <https://rlhfbook.com/c/13-cai>
- **Chain-of-X Survey** (arXiv 2404.15676, 2024) — taxonomy of CoH-family hindsight/critique/revision methods. <https://arxiv.org/html/2404.15676v2>

**Synthesis.** "Take an `nl_critique` and turn it into a gradient step without collapsing into the critique style" is essentially the CoH problem statement. The 2024 best practice has three pieces: (1) **CoH-style sequence formatting** — train on `[draft, critique, revision]` triples so the model learns the *operation* of critique-then-revise, not just the surface style of the critique; (2) **inference-time tag conditioning** — at deploy time prepend a "well-aligned" tag and *strip* the critique; CoH's original recipe; (3) **dual reward / dual loss** — pair the CoH SFT loss with a KL-to-pre-critique-reference penalty (analogous to RLHF's KL term) to bound how far the policy can move per critique. Constitutional AI's two-stage recipe (SFT on self-revisions, then RLAIF on the revised model) is the production template. The collapse failure mode (model starts emitting verbose critique-style prose unprompted) is empirically real but under-documented in the literature; CAI papers note it implicitly via their dual helpfulness/harmlessness objective.

**Where the literature is thin/contradictory.** I found no 2024–26 paper that *specifically* studies "model collapses into critique style after CoH training" with concrete metrics; it's folklore. There is also no settled recipe for *online* CoH (one critique at a time, gradient step immediately) — all CoH/CAI work is batched. For lile, the safest design is offline-style: accumulate critiques in the replay buffer, periodically do mini-SFT batches with the CoH format + KL-to-reference, never single-step on a single critique.

---

## 7. Sample efficiency of QLoRA / LoRA continual updates

**Key papers**

- **LoRA Without Regret** (Thinking Machines Lab, 2025) — central sample-efficiency claim: *when key details are right, LoRA matches FullFT sample efficiency and final performance*; the details are (a) target all linear layers, (b) optimal LR is ~10× higher than FullFT, (c) batch size matters (LoRA tolerates large batches less well), (d) LoRA capacity becomes the bottleneck only when the dataset is pretraining-scale. <https://thinkingmachines.ai/blog/lora/>
- **LoRA Learns Less and Forgets Less** (Biderman et al., COLM 2024) — quantifies the trade-off: LoRA underperforms FullFT on aggressive new-domain learning but forgets the source domain dramatically less; MLP modules carry more rank than attention; rank=16 with all-linear targeting is the empirical sweet spot for instruction tuning. <https://arxiv.org/abs/2405.09673>
- **Practical Tips for Finetuning LLMs Using LoRA** (Raschka, 2023, widely cited as the 2024 default) — `α = 2r` heuristic, all-linear target modules, optimal LR around 3e-4 for LoRA. <https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms>
- **QLoRA** (Dettmers et al., NeurIPS 2023) — 4-bit base + LoRA + paged optimizers; the foundational sample-efficiency claim that 4-bit doesn't measurably hurt downstream performance vs 16-bit. <https://arxiv.org/abs/2305.14314>
- **QA-LoRA** (Xu et al., ICLR 2024) — fixes the QLoRA quantization-vs-adaptation degree-of-freedom mismatch; merges back to a quantized model losslessly. <https://openreview.net/forum?id=WvFoJccpo8>
- **QDyLoRA** (EMNLP Industry 2024) — dynamic-rank QLoRA; choose effective rank per task without retraining. <https://aclanthology.org/2024.emnlp-industry.53.pdf>

**Synthesis.** For a live-training daemon the empirical recommendations are now reasonably stable: (1) **target all linear modules** (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) — attention-only underperforms by a measurable margin; (2) **rank 8–32 with `α = 2r`** is the sample-efficient regime for behaviour change without overfitting; rank 256 only helps if you have pretraining-scale data; (3) **LR ~10× higher than FullFT** (typically 1e-4 to 5e-4 for Qwen3-class), and lower for very-low-rank runs; (4) **small batches** — LoRA's sample efficiency degrades with large batches more sharply than FullFT, contra what you'd naively expect; (5) **QLoRA's 4-bit base costs ~33% memory and 39% wall-clock** vs 16-bit LoRA with no measurable quality loss for instruction tuning. None of the studies give concrete "samples-to-behaviour-change" numbers for sparse user feedback, but Biderman et al. and LoRA-Without-Regret both report ~hundreds-to-low-thousands of high-quality samples to durably move task-specific behavior.

**Where the literature is thin/contradictory.** No published numbers on samples-to-behavior-change for *online single-user* feedback specifically (vs static benchmark fine-tuning). Also, LoRA Without Regret claims "LoRA matches FullFT sample efficiency" while LoRA Learns Less claims the opposite — the difference appears to be (a) all-linear vs Q/V-only targeting, (b) LR tuning, and (c) dataset size, but the two papers don't directly reconcile.

---

## Cross-cutting honest call-outs

- **The biggest gap in the literature for `lile`'s problem is the streaming, sparse-feedback, single-user setting.** Almost everything cited above is benchmarked on either (a) static SFT corpora, (b) discrete CL benchmarks with known task boundaries, or (c) high-throughput rollout-farm RL. The `lile`-specific regime (one user, sub-1Hz feedback, hours of uptime, mixed implicit/explicit signals) has no published baseline.
- **Replay-vs-orthogonal is not yet decided** for continual LoRA. Replay (DER++, FOREVER) and orthogonal subspaces (O-LoRA, CLoRA) are both strong; they have not been benchmarked head-to-head in the same paper on the same backbone with the same evaluation.
- **Influence functions on LoRA are the most underexplored *practical* tool** I came across. DataInf is cheap enough to run online and would give per-feedback-item importance weights that are theoretically grounded. I could not find a paper that combines DataInf with a streaming LoRA daemon — that gap is a genuine research opportunity.
- **No paper I found cites a confirmed `lile`-style daemon as a baseline.** The closest analogues are Online-LoRA (task-free, but vision) and AReaL (streaming, but high-throughput RL).

---

## Citations I could not confirm and have NOT included

- I checked SuRe (arXiv 2511.22367) and FOREVER (arXiv 2601.03938) directly via WebFetch and they exist; both have future-dated arXiv IDs which is consistent with the April 2026 cutoff.
- I did *not* cite "CCPD v2" or "PPO-from-NL-critique" because I could not find an authoritative paper with that exact name in 2024–26; if you have a specific reference please send it and I'll fold it in.
- "DPO-with-verifier" as a single named method does not have one canonical paper; V-STaR is the closest formalization (DPO-trained verifier on top of a STaR generator).
