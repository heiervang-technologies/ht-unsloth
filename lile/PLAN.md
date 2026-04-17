# LiveLearn (`lile`) — Design & Development Plan

**Status:** v0.1 draft, living document
**Codename:** LiveLearn / `lile` (CLI + pip package)
**One-line pitch:** A single-model, always-on local training & inference daemon that learns from live traffic with configurable, stackable objectives — built on top of `ht-unsloth`.

---

## 1. North Star

> One mutable model. Always serving. Always trainable. Any objective, any time, via API.

**The user experience we want:**

```bash
# Start the daemon
lile serve --model qwen3-8b --adapter live --port 8000

# In another terminal — normal inference
curl -X POST localhost:8000/v1/chat/completions -d '{...}'

# At any point, send a learning request
curl -X POST localhost:8000/v1/train -d '{
  "objective": "sft",
  "batch": [...],
  "kl_anchor": "base",
  "lr": 1e-5
}'
# → 200 OK, enqueued. Next /chat/completions reflects it.
```

The daemon holds **one model state**. That state is mutable, versionable, savable at any time, and reproducible from a trajectory log.

---

## 2. Decision: what do we build off of?

This is the highest-stakes decision. Everything downstream depends on it.

### 2.1 The shortlist, honestly evaluated

| Option | Verdict | Why |
|---|---|---|
| Solo torch + FastAPI + custom Triton | ❌ | Reinvents Unsloth's kernels. 2-year project before we match what Unsloth ships today. |
| Fork vLLM / llama.cpp + add autograd | ❌ | vLLM's speed comes from memory layouts actively hostile to autograd (PagedAttention, CUDA graphs). llama.cpp has no real training path (ggml autograd is a toy). Tearing out the load-bearing walls. |
| Fork TRL directly | ❌ | TRL is a trainer library that *uses* Unsloth/vLLM. Lower in the stack than we want — we'd be re-implementing Unsloth's kernels to catch up. |
| Fork transformers | ❌ | Too slow. Unsloth already monkey-patches it; no reason to own that code. |
| Fork nanoGPT / nanochat / modded-nanoGPT | ❌ | Beautiful research codebases, not production. No LoRA, no 4-bit, no serving, no GRPO. Years of work. |
| Fork verl / ROLL / OpenRLHF | ❌ (for this hardware) | Built for datacenter-class memory. HybridEngine resharding assumes you can hold two sharded weight copies in VRAM — false on 1-2× RTX 3090. |
| **Fork Unsloth (our `ht-unsloth`)** | ✅ | Already the best-in-class for consumer-GPU LoRA/QLoRA/FP8 training. TRL-compatible trainers (SFT, GRPO, GSPO, DPO) out of the box. Handles the hardest low-level detail — dequant-merge-requantize for 4-bit bases — correctly. |

### 2.2 Why Unsloth is the right base, concretely

- **Kernel moat.** Hand-written Triton for LoRA + rotary + attention on Ampere. Nobody else has this for consumer GPUs.
- **Memory math actually works on 24GB.** Qwen3-8B FP8 GRPO fits in ~16GB with headroom for rollouts and context. Qwen3-14B with 4-bit + 16-bit LoRA fits comfortably. Qwen3-30B-A3B MoE is reachable with 2× 3090s.
- **Correct 4-bit merge path.** Unsloth's `save.py` already implements dequant → fp32 merge → optional requantize. This is the pattern that matters for our "progressively merge adapters into base" feature — reinventing it would be a multi-week footgun marathon (see §6).
- **TRL-compatible trainers.** SFT, GRPO, GSPO, DPO work today. REINFORCE and policy-gradient variants slot into the same abstraction.
- **Monkey-patch architecture.** Unsloth doesn't fork transformers; it patches it. This means our fork stays surgical — we override what we need and inherit upstream progress.
- **Active development.** February 2026 release: 12× faster MoE training, 35% less VRAM, embedding support, ultra-long RL context. We want to ride this, not compete with it.

### 2.3 What we are NOT inheriting

Unsloth is a **training library**, not a daemon. It has no persistent server, no trajectory buffer, no adapter routing, no stackable-objective API, no replay, no save-on-demand versioning. That's **all the stuff we build** — and it's the valuable part.

### 2.4 The fork strategy (matches your existing `ht-unsloth` workflow)

Keep the invariant you already have: **few clean commits on top of upstream**. We split our work across two repos:

1. **`ht-unsloth`** (the fork): only changes that *must* live inside Unsloth — new trainers, kernel tweaks, hooks we need for live serving. Keep each commit rebasable.
2. **`lile`** (the new repo): the daemon, API, adapter manager, trajectory store, control plane. Imports `ht-unsloth` as a pinned dependency. This is ~80% of the actual product code.

Rule of thumb: **if a change could plausibly be upstreamed to Unsloth, it belongs in `ht-unsloth`. Otherwise it belongs in `lile`.** This keeps the fork lean and makes upstream rebases tractable.

---

## 3. The core mental model

### 3.1 One model, one state

At any instant, the daemon owns exactly one "live model," which is the composition:

```
live_model = base_weights ⊕ merged_deltas ⊕ active_lora_adapter
```

- **`base_weights`**: the immutable starting point (e.g. Qwen3-8B Q4_K_M). Never modified directly.
- **`merged_deltas`**: a full-precision delta tensor representing everything we've baked in from prior LoRAs via progressive merging. Lives in CPU RAM between merges; applied to base on load.
- **`active_lora_adapter`**: the currently-training LoRA. Hot in GPU. This is what gradients flow into.

When the user says "merge," `active_lora_adapter` gets dequant-merge-requantized into `merged_deltas`, then reset to zero, and training continues on a fresh adapter. This gives us the progressive merge story without ever doing full-weight training on a 4-bit base (which is the wrong thing to do — see §6).

**The `live` state is saveable at any instant** as `(base_ref, merged_deltas.safetensors, active_adapter.safetensors, trajectory_log_offset)`. Reproducible, versioned, diff-able.

### 3.2 Full fine-tuning mode

Optionally, the user can run in **full-FT mode** (no LoRA): `live_model = base_weights` with the weights themselves trainable in 16-bit. This only fits for small models (≤3B on 24GB) or with aggressive offloading, but it's a supported configuration for users who want it. Internally it's just a different adapter strategy — the rest of the system is unchanged.

### 3.3 Objectives are stackable

The API accepts objectives as a **list** per batch, and optionally per sample:

```json
{
  "batch": [
    {"input": "...", "target": "...", "objectives": [{"sft": {"weight": 1.0}}]},
    {"input": "...", "reward": 0.8, "objectives": [{"grpo": {"group_id": "g7"}}]}
  ],
  "batch_objectives": [
    {"kl_anchor": {"target": "base", "weight": 0.1}}
  ]
}
```

Semantics: per-sample objectives define per-sample loss terms; batch objectives are added once per step. The trainer composes them as a weighted sum. If a configuration is invalid (e.g. GRPO sample without group_id), the request is rejected — we don't silently do the wrong thing.

### 3.4 The compute queue

Large batches exceed our memory-per-step budget. The daemon doesn't reject them — it chunks them and **enqueues** them onto a compute buffer. Training steps drain the queue asynchronously between inference requests (or on dedicated GPU if you run 2× 3090s in split mode).

Key property: **a batch committed to the queue is guaranteed to be reflected in the model before any POST'd inference request that arrives after the training request's `commit_token` was returned.** This is the "POST a batch, next inference sees it" promise. Implementation: inference requests check the queue's commit cursor before dispatch.

---

## 4. Architecture (high level)

<!-- Sketch; will be expanded in §7 with module breakdown. -->

```
                       ┌─────────────────────────────────────┐
                       │              lile daemon            │
                       │                                     │
  OpenAI-compat API ──▶│  Router  ──▶  Inference Engine      │
  /v1/chat/completions │     │         (Unsloth fast_generate│
                       │     │          or vLLM sidecar)     │
                       │     │                               │
  Learning API    ────▶│  Queue  ──▶  Training Engine        │
  /v1/train            │            (ht-unsloth trainers)    │
                       │                                     │
  Control API     ────▶│  Controller  ──▶  Adapter Manager   │
  /v1/state/*          │                   Trajectory Store  │
                       │                   Snapshot Manager  │
                       └─────────────────────────────────────┘
                                      ▲
                                      │
                             ┌────────┴────────┐
                             │   GPU state     │
                             │   base + Δ + a  │
                             └─────────────────┘
```

Components in detail come in §7. The important topology choices:

- **Single-process for 1× 3090.** Inference and training share one CUDA context; we swap between them with Unsloth's weight-sharing path. Adds some step-to-generate latency; avoids inter-process weight sync.
- **Split-process for 2× 3090s.** One GPU runs serving (vLLM sidecar), one runs training. Weight sync over shared memory every N training steps. This is the locallama sweet spot.

---

## 5. Learning methods — v0 scope

All of these are expressible as composable objective functions operating on a batch:

- **SFT** — standard next-token CE loss. The baseline. Works day one.
- **Policy gradient / REINFORCE** — loss = `-Σ log π(a|s) · R`. Needs rollouts + reward signal.
- **GRPO (and family: GSPO, DAPO, REINFORCE++)** — group-relative policy optimization. Unsloth has this; we wrap it.
- **DPO** — paired preference learning. Included for completeness but rarely the right choice for online use (requires pairs).
- **KTO** — binary desirable/undesirable learning. The natural fit for thumbs-up/down UI signals. See §5b.1.
- **Chain of Hindsight (CoH)** — learning from natural-language critiques and rewrites. See §5b.2.
- **KL anchoring** — L2 / forward-KL / reverse-KL penalty to a reference model (base, EMA, snapshot). Not a standalone objective but the most important modifier for avoiding forgetting (the "RL's Razor" finding).
- **Distillation** — KL to a teacher model's logits. Teacher can be a larger frozen model or a prior snapshot of ourselves (self-distillation).
- **Prompt baking** — a novel-to-lile trainer. Given `(system_prompt, query, response)` triples, teach the model to produce `response` from `query` alone. Implementation: standard SFT loss but the system_prompt is withheld at training time; we can optionally distill the *with-prompt* logits into the *without-prompt* forward pass for stability.

**Stackability examples:**

- SFT + KL anchor to base (safe fine-tuning)
- GRPO + KL anchor to EMA (online RL without drift)
- Distillation + SFT (student learns teacher + matches gold)
- Prompt baking + distillation from self-with-prompt (the theoretically cleanest formulation)

---

## 5b. Binary feedback (KTO) and feedback-guided correction

A core use case for a live daemon is learning from the user's in-line reactions: thumbs-up/down buttons, "no, more like X," full rewrites. v0 needs first-class paths for all three.

### 5b.1 KTO for binary feedback — yes, include it

**Recommendation: KTO is a first-class objective in v0.** It is not superseded and is arguably more relevant in 2026 than when it launched, because the constraint it solves is exactly ours: you get one signal per interaction, not paired preferences.

What the literature actually says, evenly:

- KTO needs **only a binary desirable/undesirable signal per example** — no pairs. This is the thumbs-up/down shape of real UI feedback. Most other preference methods require you to show the user two responses and ask which they prefer; KTO doesn't.
- Contextual AI's original result: KTO matches or exceeds DPO performance without using preference data, tested across 1B–30B. Independent reproductions (Hugging Face's pref-tuning study) have generally found DPO slightly ahead *when paired preference data exists*, but KTO ahead when you only have binary signals — which is our situation.
- KTO has a useful stability property: KTO handles contradictory preferences from different humans better than DPO. In a case where two humans have opposite preferences, DPO might satisfy one while making both worse off. KTO, on the other hand, avoids changing the policy in the presence of such contradictions. For a personal assistant learning from a single user this matters less; for locallama users training from multiple people's interactions it matters a lot.
- One honest limitation to bake into defaults: KTO's objective overly focuses on maximizing the likelihood of positive examples. In practice this means you may want to weight undesirable examples more heavily (KTO exposes `desirable_weight` and `undesirable_weight` hyperparameters). We should expose both and default to something like 1.0/1.5 rather than 1.0/1.0 based on community findings.

**Integration into lile:**

- Add `kto` as an objective type in the registry.
- API shape: each sample carries `"label": "desirable" | "undesirable"` instead of a paired target.
- The most common real-world source is a thumbs button on the frontend. We document the pattern: the daemon receives `(prompt, response, thumb)` triples, adds them to the trajectory log, and periodically batches them into a KTO step.
- KTO composes cleanly with KL anchoring (it has a built-in KL term already, but stacking with our base-anchor works too) and with SFT (KTO + SFT on positives alone is a useful recipe).
- TRL has a `KTOTrainer`; Unsloth's TRL integration makes this close to free.

### 5b.2 Feedback-guided correction — yes, this is a real research area and we should expose it

"Feedback-guided correction" isn't one algorithm, it's a family. For our purposes three variants matter, from simplest to most ambitious.

**Variant A: Rewrite-as-SFT (the baseline).** User dislikes a response, provides the correct one. We SFT on `(prompt → corrected_response)`. Mechanically trivial — it's just SFT with user-sourced targets. What makes it interesting is *which* adapter it gets logged against and *how* it's mixed with other data. Often the simplest thing works: log rewrites into a high-priority buffer, weight them 3–5× in the next training step, done.

**Variant B: Chain of Hindsight (CoH).** From Liu, Sferrazza, and Abbeel (2023) — the technique is: convert all types of feedback into sequences of sentences, which are then used to fine-tune the model, allowing us to take advantage of the language comprehension capabilities of language models. We condition the model on a sequence of model generations paired with feedback. By doing so, the model is trained to generate outputs based on feedback, while learning to identify and correct errors. Concretely you build training sequences like: `"Prompt: X. Bad response: Y. Feedback: 'too verbose.' Good response: Z."` and train next-token CE on the whole thing. At inference you can prepend a hint or not — the model has internalized what "too verbose" means relative to its own failure modes.

CoH is appealing for us because it ingests **natural-language feedback** — "be more concise," "don't apologize so much," "wrong, the answer is 42" — which is the dominant form of real human feedback. Users don't thumbs-button; they complain in words.

**Variant C: SCoRe — self-correction via multi-turn RL.** From the ICLR 2025 paper: SCoRe, a multi-turn RL approach for teaching LLMs how to correct their own mistakes. To the best of our knowledge, SCoRe is the first approach to attain significantly positive intrinsic self-correction: relative to base Gemini models, our method attains an absolute 15.6% gain on self-correction for reasoning problems from MATH and an absolute 9.1% gain on coding problems from HumanEval. This is heavier — you train the model to produce a first attempt, look at its own/external feedback, and then produce a better attempt. It's a GRPO-style multi-turn trainer.

SCoRe is more than we need for v0, but it's worth knowing the shape because the trajectory store we're already building naturally produces SCoRe-shaped data (the user often sends a follow-up when they disliked the first answer — that's a correction trajectory for free). We should design the trajectory schema so SCoRe is a drop-in later.

**A note on what to avoid:** recent work demonstrates that naïvely prompting LLMs for self-correction can degrade performance. The failure mode is specifically *prompt-only* self-correction without training — "are you sure? try again" makes models worse on average. We're fine because we're *training* on correction signals, not just prompting for them, but it's worth flagging in docs: "turn on self-correction prompting" is not a feature; learned correction is.

### 5b.3 The unified feedback pipeline for lile

Here's how the three feedback shapes collapse into one clean design:

1. **Every response gets a `response_id`.** Returned in the API, referenceable for up to N days.
2. **One feedback endpoint: `POST /v1/feedback`.** Payload variants:
   ```
   {"response_id": "...", "kind": "binary", "value": "up" | "down"}
   {"response_id": "...", "kind": "rewrite", "better_response": "..."}
   {"response_id": "...", "kind": "nl_critique", "critique": "too verbose"}
   {"response_id": "...", "kind": "nl_critique_with_rewrite", "critique": "...", "better_response": "..."}
   ```
3. **The daemon decides how to train on it**, based on configured objectives:
   - `binary` → KTO sample.
   - `rewrite` → high-weight SFT sample, optionally paired with original as a DPO-style sample if `pair_with_original: true`.
   - `nl_critique` → CoH sample (original + critique, no rewrite). CoH works with critiques alone; it learns to avoid the criticized behavior.
   - `nl_critique_with_rewrite` → CoH sample using all three fields. Richest signal; Liu et al. show this is where CoH wins most clearly.
4. **All feedback lands in the trajectory log first, always.** Training is a separate concern. This lets us replay, re-weight, or apply new methods to old feedback when a better algorithm lands next year.

This gives us one UI surface (thumbs + "suggest a rewrite" + "what was wrong?") that maps to three algorithms without the user caring which one fired.

### 5b.4 Adding the preferred-response endpoint — one more feedback shape, one more route

The user may also have a concrete better response in mind — not a critique, just "here's how you should have answered." This is the highest-information feedback shape of all, and it deserves a first-class API:

```
{"response_id": "...", "kind": "preferred", "better_response": "..."}
```

Semantically this is `(x, y⁻, y⁺)` — a genuine preference pair. The obvious thing to reach for is DPO, and DPO is indeed the *name* of the endpoint behavior. But the implementation underneath should route through the same CCPD v2 machinery (see §5c.11) rather than vanilla DPO, for reasons laid out in §5c.16 below. Same endpoint, better gradient pathway, user doesn't care.

We also distinguish two sub-cases:

- **Pure rewrite.** `kind: "preferred"` with `better_response` only. Treated as a 2-candidate preference update.
- **Critique + rewrite.** `kind: "nl_critique_with_rewrite"`. We get *both* the semantic direction and the concrete target. This is the richest signal and the one we optimize for hardest (see §5c.16).

---

## 5c. A proposed π-only feedback-guided objective — **Critique-Conditional Policy Distillation (CCPD)**

*This section is speculative research design rather than a settled plan — but it's mathematically grounded and aligned with 2025–2026 directions. If it works, it's lile's novel contribution. If parts don't work, they're additive to the CoH/KTO baselines.*

### 5c.1 The problem, stated precisely

Given: a live policy π_θ, a prompt x, a response y that π generated, and a natural-language critique c from the user. No reward model, no teacher, no paired preference, no user-provided rewrite. Just (x, y, c).

Find: a gradient update to θ that makes π more likely to produce critique-satisfying responses to *similar* prompts in the future — without an external verifier, without collapse, and sample-efficient enough that one (x, y, c) tuple produces a meaningful update.

The key constraint: **the only thing that knows what c means is π itself.** An LLM of reasonable capability can read "too verbose" and produce a less verbose version of y. That latent capability is the lever we have.

### 5c.2 The core observation — the critique *is* the reward, implicitly

There's a recent thread of work using the log-ratio between two policy configurations as a reward signal, without training a separate reward model. Self-Rewarding PPO does this by using a reward function designed as the log policy ratio between the SFT model and the pretrained base model. This function serves as an implicit reward signal, using the pretrained policy as a baseline and the SFT policy as a target. DPO's entire derivation rests on the same reparameterization trick.

Apply that trick to critiques. Define a critique-conditional reward:

```
r_c(x, y) := β · [log π(y | x, c) − log π(y | x)]
```

In words: how much more likely does π make the response y when we show it the critique c, vs when we don't? Intuition:

- If c is "too verbose" and y is short, c raises the likelihood of y. `r_c > 0`.
- If c is "too verbose" and y is rambling, c lowers the likelihood of y. `r_c < 0`.
- If c is orthogonal ("what's the weather?") and y is about cooking, c barely moves the likelihood. `r_c ≈ 0`.

This is just Bayes: `log π(y|x,c) − log π(y|x) = log π(c | x, y) − log π(c | x)` (up to a constant, modulo the direction we condition). Either formulation works — the second is often more numerically stable because critiques are short.

**Why this isn't crazy:** a modern instruction-tuned π at 7B+ scale is genuinely competent at reading a critique and evaluating whether a response honors it. That's the competence we're mining. The same competence that makes LLM-as-a-judge work makes this work — we're just using π as its own judge, on its own generations, with respect to a user-provided criterion.

**Why this beats training a reward model on critiques:** a reward model would need a dataset of (response, critique, score) triples. We have one critique at a time. π is already trained.

### 5c.3 The full objective

Given (x, y⁻, c) — prompt, bad response that earned the critique, critique — and an optional self-generated refinement y⁺ ∼ π(·|x, c):

**Step 1 (decode): generate a critique-guided refinement.** At feedback time, sample `y⁺ ∼ π(·|x, c)`. This is cheap — one forward pass of generation. π reads the critique and produces what it *thinks* is a better response. This y⁺ is our self-teacher.

Optionally use a sharper decoding for y⁺: low temperature, or best-of-N with self-scoring via the r_c signal above. This burns inference for sample quality, which is the right tradeoff — the update affects all future responses.

**Step 2 (loss): a three-term objective.** Let π_old be the policy at feedback receipt time (we freeze it; no gradient flows through it). Let π_θ be the current trainable policy.

```
L_CCPD = L_distill + α · L_contrastive + γ · L_KL
```

where:

- **L_distill** — the prompt-baking term. SFT on y⁺ *without* the critique in context:

  ```
  L_distill = −E_{t} [ log π_θ(y⁺_t | x, y⁺_{<t}) ]
  ```

  This bakes the critique-guided improvement into the weights. π learns to produce y⁺-style responses from x alone, without needing c at inference. This is the "context distillation" / "prompt baking" you mentioned — the critique is the transient context we want to compile in.

- **L_contrastive** — the rejection term. Lower the likelihood of y⁻ (the bad response), anchored by the critique-conditional reward:

  ```
  L_contrastive = −E [ σ(β · [log π_θ(y⁺|x) − log π_θ(y⁻|x)] − Δ_c) ]
  ```

  where `Δ_c = β · [log π_old(y⁺|x,c) − log π_old(y⁻|x,c)]` is the *target margin computed under π_old with the critique visible*. This is the novel piece. It says: the gap between y⁺ and y⁻ under the *critique-free* policy should equal the gap between y⁺ and y⁻ under the critique-aware policy. We're asking π to internalize the discriminative power that the critique gave it.

  Mathematically this is a DPO-style sigmoid loss with the target margin not at zero, but at whatever margin π_old would have predicted *if it could see the critique*. The critique is the teacher, and it's the same model reading the critique differently.

- **L_KL** — the anchor. Standard KL-to-reference to prevent drift:

  ```
  L_KL = KL(π_θ(·|x) ‖ π_ref(·|x))
  ```

  π_ref is whatever you want it anchored to — base, EMA, snapshot. This is the RL's Razor lever from earlier in the plan.

**Why three terms and not one:** L_distill alone tends to overfit to y⁺ (which may itself be imperfect). L_contrastive alone is DPO-shaped and unstable with self-generated pairs (y⁺ and y⁻ both came from π). L_KL alone doesn't encode the feedback. Together: L_distill pulls toward the critique-satisfying region, L_contrastive pushes away from the critiqued region with calibrated force, L_KL keeps us honest globally.

### 5c.4 Why π_old is useful (and when it's strictly needed)

π_old (the policy snapshot at feedback-receipt time) appears only in computing `Δ_c` — the target margin. You could in principle compute it with π_θ and let gradients flow through both sides, but this is unstable: the target and the student become the same moving thing.

Freezing a π_old copy for the duration of one feedback-batch's gradient steps is cheap — it's just "don't update these weights for 1-10 steps." Same model instance, different `requires_grad` flag. No separate model in VRAM.

We **do not** need π_old for the L_distill or L_KL terms (L_KL uses π_ref which is a different thing — the global anchor, typically the base model).

### 5c.5 The reasoning-trace infilling variant

You mentioned reasoning-trace infilling. Here's how it slots in, and I think it's the sharper version of this objective for any task with chain-of-thought structure:

Instead of sampling a single y⁺, decompose into `y = (trace, answer)`. When the critique targets a reasoning error ("you divided wrong at step 3"):

**Step 1 (infill): mask the erroneous step and infill with critique-conditioned decoding.** That is, regenerate only the span of trace that the critique targets, conditioned on (x, prefix of trace up to error, c). This is cheaper, more focused, and gives you token-level credit assignment for free — the gradient flows only into the tokens that actually changed.

**Step 2 (loss): same three terms, but y⁺ and y⁻ differ only in the infilled span.** This dramatically reduces variance because everything not-critiqued is held constant. The L_distill term becomes nearly a surgical edit to specific reasoning behavior rather than a blunt rewrite.

This is structurally similar to span-level credit assignment in Text2Grad (Wang et al., 28 May 2025), where span-level alignment between critique phrases and policy output is leveraged for local gradient updates, but π-only — we're doing the span alignment via decoding rather than via a separate alignment model.

### 5c.6 How this relates to Critique-GRPO

There's a concurrent paper worth knowing about: Critique-GRPO directly adopts a 'shaping function' used to reweight gradients to amplify learning from correct, unfamiliar refinements and to penalize incorrect ones. They propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration.

Critique-GRPO **requires numerical rewards alongside critiques** — it's a hybrid. CCPD as proposed here is strictly critique-only, using the implicit reward from critique-conditional likelihood ratios in place of Critique-GRPO's scalar rewards. If CCPD works, it's a strict generalization: Critique-GRPO becomes the special case where your critique is "correct" / "incorrect" and you happen to have a verifier.

The other distinction: Critique-GRPO is designed for verifiable domains (math, code). CCPD is designed for the open-ended personal-assistant domain where verifiability is absent.

### 5c.7 What could go wrong — honest enumeration

1. **π might not be a competent critic of itself.** If π doesn't understand "too verbose" well enough to generate a less verbose y⁺, the whole stack falls apart. Mitigation: this is an empirical question that scales with base model capability — works for Qwen3-8B+, probably fails for 1B models. We should benchmark this explicitly before committing.
2. **r_c can be gamed by length.** Long responses have lower log-likelihood in general, so any critique moving response length systematically moves r_c in a way that looks like signal but isn't. Mitigation: length-normalize the log-likelihood, as is standard in DPO variants.
3. **The L_contrastive target margin Δ_c has weird edge cases** when the critique barely moves likelihoods. When `|Δ_c|` is small, we're asking π to learn a tiny distinction from a noisy signal. Mitigation: only apply L_contrastive when `|Δ_c| > τ` for some threshold; fall back to L_distill only when the critique is weakly informative.
4. **Self-generated y⁺ can be worse than y⁻.** π conditioned on critique doesn't always produce a better response — sometimes it misreads the critique. Mitigation: optional sanity check: compute `r_c(x, y⁺) > r_c(x, y⁻)` before committing the update. If the critique doesn't prefer y⁺ over y⁻ under π, skip the sample. This is π-only self-verification at the cost of one extra forward pass.
5. **The prompt-baking term may over-anchor to one example.** If we L_distill on y⁺ too aggressively, we memorize the specific refinement rather than the critique's direction. Mitigation: the KL anchor term does this work; also the whole point of collecting multiple critiques over time is that L_distill averages over many y⁺.

### 5c.8 Implementation sketch

```python
def ccpd_loss(x, y_neg, c, pi_theta, pi_old, pi_ref,
              beta=0.1, alpha=0.5, gamma=0.05, tau=0.3):
    # Step 1: sample self-refinement
    y_pos = pi_theta.generate(x, critique=c, temperature=0.3)  # no grad

    # Sanity check: does the critique actually prefer y_pos over y_neg under pi?
    r_pos = logprob(pi_old, y_pos, x, c) - logprob(pi_old, y_pos, x)
    r_neg = logprob(pi_old, y_neg, x, c) - logprob(pi_old, y_neg, x)
    if r_pos - r_neg < tau:
        return None  # skip; critique not informative here

    # L_distill: SFT on y_pos without critique in context
    L_distill = -logprob(pi_theta, y_pos, x).mean()

    # L_contrastive: DPO-style with critique-conditional target margin
    delta_c = beta * (r_pos - r_neg)  # computed under pi_old
    logratio_pos = logprob(pi_theta, y_pos, x) - logprob(pi_ref, y_pos, x)
    logratio_neg = logprob(pi_theta, y_neg, x) - logprob(pi_ref, y_neg, x)
    L_contrast = -F.logsigmoid(beta * (logratio_pos - logratio_neg) - delta_c).mean()

    # L_KL: global anchor
    L_kl = kl_divergence(pi_theta, pi_ref, x).mean()

    return L_distill + alpha * L_contrast + gamma * L_kl
```

One call to generate (for y⁺), four forward passes for log-probs (π_θ on y⁺, y⁻; π_ref on y⁺, y⁻), two under π_old for the target margin. Roughly 2× a DPO step per sample. Sample efficiency wins pay for this.

### 5c.9 What to ship, and when

**v0.2 (phase 2 of the roadmap):** Ship CCPD as an experimental objective behind a feature flag. Benchmark against:
- CoH alone (critique as context, no self-refinement)
- Vanilla SFT on self-generated y⁺ (ablation: L_distill only)
- KTO if user also provides binary label

**The right benchmark for "does this work":** a held-out set of (prompt, model response, critique) triples where we know what the critique-satisfying answer looks like. Measure whether one CCPD update generalizes — i.e., does applying the critique to sample N improve behavior on samples N+1..N+k for semantically similar critiques? This is the sample-efficiency claim.

**If it works**, this becomes lile's headline capability: "send a critique, get a learned correction, measurable in one update." If it doesn't — specifically if the π-is-its-own-judge assumption fails for small models — we fall back to CoH + KTO, which are already robust.

### 5c.10 Correction prompted by Razin et al. (2025) — and the resulting cleaner design

**Razin, Lin, Yao & Arora (2025) — "Why is Your Language Model a Poor Implicit Reward Model?"** (arXiv:2507.07981) directly pressures the load-bearing assumption of CCPD's contrastive term as originally written. I owe an honest revision, and it turns out the revision makes the proposal stronger, not weaker.

**Their result in one sentence:** the `log π(y|x) − log π_ref(y|x)` parameterization that DPO uses — the "implicit reward model" or IM-RM — generalizes worse than putting a linear head on the hidden representations (the "explicit reward model" or EX-RM), *even when both are trained on identical data with identical loss and identical base LM*. They rule out the intuitive "LLMs aren't good verifiers" explanation explicitly.

**The mechanism that matters for us:** gradient updates to an IM-RM flow through the unembedding and preferentially adjust token-identity probabilities. This biases learning toward surface-form cues — specific tokens appearing in good vs bad responses — rather than the semantic content encoded in hidden states. EX-RMs, by routing gradients through a representation bottleneck, implicitly regularize toward semantics. Razin et al. explicitly flag this as a plausible cause of the DPO-vs-RLHF gap: DPO suffers from a reliance on superficial token-level cues.

**Consequence for CCPD as written in §5c.3:** the L_contrastive term is DPO-shaped. It inherits the IM-RM surface-cue bias. Keeping it as written would build our headline capability on top of a known weak foundation.

**But — and this is the key observation — the paper's critique is about the gradient pathway, not the reward signal.** The scalar log-ratio is fine as a ranking/weighting signal; it's using it as the *differentiated quantity* that creates the pathology. And given the user's permission for auxiliary sampling (especially memory-neutral sampling), we have a cleaner fix available.

### 5c.11 CCPD v2 — auxiliary sampling + detached rewards

The fix: use `r_c` as a **detached scalar reward** on **auxiliary-sampled rollouts**, with the gradient flowing through the policy's log-prob of its own generations. This is the EX-RM-style pathway — through hidden states, not through the unembedding as a log-ratio.

**Why auxiliary sampling is almost free on our hardware.** On a 3090 serving a 7-14B model, the paged-attention KV pool is already allocated for serving. Generating k extra short completions at feedback time reuses that pool — incremental VRAM cost is a few MB for a batch of short sequences, not gigabytes. Wall-clock cost: k × (a few hundred ms). This is the right tradeoff: burn inference once, get a better gradient update that affects all future responses. It also composes cleanly with the compute-queue design in §3.4 — auxiliary sampling is queued alongside the training step that uses it.

**The revised objective.** Given (x, y⁻, c) — prompt, critiqued response, critique — and a small auxiliary rollout budget k (say 4–8):

**Step 1 (auxiliary sampling).** Sample k refinement candidates under `π_old` with no grad:
```
Y⁺ = { y⁺_i ~ π_old(·|x, c) : i = 1..k }
```
Temperature 0.7-1.0 for diversity. Add y⁻ to the candidate set as a known-bad anchor. No training memory used.

**Step 2 (scoring — detached).** For each candidate y, compute under `π_old`, no grad:
```
r_c(y) = β · [log π_old(y|x, c) − log π_old(y|x)] / |y|
```
Length-normalize by token count to kill length bias. These are scalars — **detached from the graph.** The gradient pathway Razin et al. warn against is absent.

**Step 3 (rank and filter).** Compute advantages as centered ranks, not raw scores:
```
A(y) = rank(r_c(y)) − (k+1)/2    # mean zero, range ±(k−1)/2
```
Ranking rather than raw scores further insulates us from surface-cue sensitivity — what matters is ordering, not absolute value. Drop the sample entirely if `max A − min A < τ`: the critique failed to discriminate among π's generations, so there's nothing to learn.

**Step 4 (the loss).**
```
L_CCPD_v2 = L_policy + α · L_distill + γ · L_KL

L_policy   = −E_y [ A(y) · log π_θ(y | x) ]             # REINFORCE with critique advantage
L_distill  = −E_{y_i ∈ top-m Y⁺} [ log π_θ(y_i | x) ]   # SFT on top-m ranked samples
L_KL       = KL(π_θ(·|x) ‖ π_ref(·|x))
```

Three things change structurally from v1:

1. **Gradient flows through `log π_θ(y|x)` weighted by a scalar advantage.** This is the EX-RM-style pathway — through hidden representations — that Razin et al. show generalizes well. The log-ratio is gone from the differentiated loss; it only enters as a scalar weight.

2. **k auxiliary rollouts give us a distribution**, not a point estimate. One critique + k rollouts is a much richer signal. The variance of the rank-based advantage is much lower than a single (y⁺ vs y⁻) comparison.

3. **L_distill uses top-m samples, not just one y⁺.** A bad draw of y⁺ no longer contaminates the distill term — we distill from the critique-ranked best.

### 5c.12 Why this is better than v1 on every axis

- **Generalization.** The Razin et al. pathology is avoided by construction: no gradient flows through the log-ratio parameterization. L_policy is REINFORCE, L_distill is SFT — both are gradient shapes empirically robust in 2025-2026 RL-for-LLMs work.
- **Sample efficiency.** k rollouts per feedback event produce a richer advantage, and rank-based advantages have lower variance than pairwise margin targets.
- **Robustness.** Ranking is invariant to absolute scale of `r_c` — critiques that barely move log-ratios still produce useful ordering when they discriminate. The `τ` filter drops cases where ordering itself is uninformative.
- **Memory.** Zero additional training memory. Auxiliary rollouts live in the serving KV pool. Detached scoring uses forward-only compute. Only the final backward pass uses training memory, and it's the size of a standard GRPO step with group k.
- **Degradation.** With k=1, α=1, γ=0 it's vanilla SFT on self-refinement. With k=8, α=0, γ=small it's critique-rewarded REINFORCE. The full objective interpolates.

### 5c.13 The reasoning-trace infilling variant under v2

Infilling gets cleaner under v2. Instead of sampling k full refinements, sample k infills of the critiqued span:

```
Y⁺_infill = { (prefix, infill_i, suffix) : infill_i ~ π_old(·|x, prefix, c) }
```

Score each with the same detached `r_c`, rank-advantage on the infilled tokens only. Credit assignment is surgical *and* the gradient pathway is Razin-safe. This is the version I'd ship first for any chain-of-thought task — the memory cost of k short infills is negligible and the credit assignment is nearly perfect.

### 5c.14 Implementation sketch (revised)

```python
@torch.no_grad()
def score_and_rank(pi_old, x, candidates, c, beta=0.1):
    scores = []
    for y in candidates:
        ll_with_c    = logprob(pi_old, y, x, c) / len(y)
        ll_without_c = logprob(pi_old, y, x)     / len(y)
        scores.append(beta * (ll_with_c - ll_without_c))
    scores = torch.tensor(scores)
    ranks = scores.argsort().argsort().float()           # 0..k-1
    advantages = ranks - (len(ranks) - 1) / 2            # centered
    return advantages, scores

def ccpd_v2_loss(x, y_neg, c, pi_theta, pi_old, pi_ref,
                 k=6, alpha=0.3, gamma=0.05, tau=0.5,
                 distill_top_m=2):
    # Step 1: auxiliary rollouts (memory-free, costs wall-clock)
    Y_pos = pi_old.generate(x, critique=c, n=k, temperature=0.9)   # no grad
    candidates = Y_pos + [y_neg]

    # Steps 2-3: detached scoring and rank advantages
    advantages, scores = score_and_rank(pi_old, x, candidates, c)
    if advantages.max() - advantages.min() < tau:
        return None  # critique non-informative; skip

    # Step 4a: REINFORCE with critique advantage
    logprobs = torch.stack([logprob(pi_theta, y, x) / len(y) for y in candidates])
    L_policy = -(advantages.detach() * logprobs).mean()

    # Step 4b: SFT on top-m critique-ranked samples
    top_idx = advantages.argsort(descending=True)[:distill_top_m]
    top_m = [candidates[i] for i in top_idx]
    L_distill = -torch.stack([logprob(pi_theta, y, x) for y in top_m]).mean()

    # Step 4c: KL anchor to base/EMA/snapshot
    L_kl = kl_divergence(pi_theta, pi_ref, x).mean()

    return L_policy + alpha * L_distill + gamma * L_kl
```

Training-graph forward passes: k+1 (for L_policy) + m (for L_distill) + 1 (for L_KL) ≈ 9 with k=6, m=2. One backward. This is comparable to a GRPO step with group size 6-8 — which Unsloth already budgets for on consumer GPUs. **No additional memory overhead vs GRPO.**

### 5c.15 Summary of what changed in v2

- **Dropped:** the DPO-shaped contrastive term with critique-conditional target margin. Elegant on paper, vulnerable to the exact pathology Razin et al. document.
- **Added:** auxiliary sampling (k rollouts per feedback, memory-neutral via shared KV pool), detached scalar rewards, rank-based advantages, REINFORCE-style gradient pathway.
- **Kept:** prompt-baking distillation (L_distill), KL anchor (L_KL), critique-as-implicit-reward intuition (as a scalar weight, not a differentiated loss), the reasoning-trace infilling variant (now cleaner), the π-only constraint.

The final shape: **ranked-advantage REINFORCE on auxiliary critique-guided rollouts, plus SFT distillation on the top-ranked samples, plus KL anchoring.** The critique-as-reward signal is preserved where it's useful (as a ranker) and removed from where it fails (as a differentiated loss). If π_old's critique-rankings are reliable — and Razin et al.'s critique does not touch ranking reliability, only gradient pathway — this achieves the sample-efficiency goal while sidestepping the generalization gap.

### 5c.16 The preferred-response endpoint — DPO-shaped API, CCPD v2 internals

When the user supplies a concrete `y⁺` rewrite, the natural instinct is vanilla DPO. Don't. Two things stop us:

1. **Razin et al. (2025) applies directly.** A user-provided `(y⁺, y⁻)` pair fed into DPO is exactly the IM-RM gradient pathway the paper warns about. The generalization gap is real even for clean preference data.
2. **The "3D-Properties" finding** from a separate concurrent line of work (Yan et al., 2025): DPO's implicit reward modeling suffers from Drastic Drop in rejected response likelihood, Degradation into response suppression, and Dispersion effect on unseen responses. In an online-learning setting this is particularly toxic — we're updating continuously, so "degradation into response suppression" compounds over feedback events.
3. **Apple's April 2025 findings** (arXiv:2409.03650 v2): across five out-of-domain settings, DPO has a mean drop in accuracy of 3% and a maximum drop of 7% versus explicit reward models. These findings highlight that DPO's implicit reward model has limited generalization ability and substantiates the integration of an explicit reward model in iterative DPO approaches. Not catastrophic, but meaningful — and iterative/online DPO is exactly our regime.

**The routing that keeps the good UX and drops the bad gradient.** The `preferred` endpoint accepts `(x, y⁻, y⁺)` and dispatches to CCPD v2 with these substitutions:

- **Skip Step 1 of CCPD v2 (auxiliary sampling) partially.** We already have `y⁺` from the user. Optionally sample k−2 additional candidates via `π_old(·|x)` to fill out the candidate set. Memory-neutral as before. The user's `y⁺` is guaranteed to sit at the top of the rank; `y⁻` at the bottom.
- **Replace Step 2 (scoring).** Don't score `y⁺` and `y⁻` via `r_c` — we know their rankings from supervision. Score the auxiliary rollouts via a proxy: their log-likelihood under π_old (plain, no critique), or their similarity to `y⁺` under some cheap metric (token overlap, or cosine similarity of mean embeddings). The auxiliary samples fill in the middle of the rank.
- **Step 3 (rank advantages) and Step 4 (loss) are unchanged.** REINFORCE with rank advantages, SFT distillation on top-m (which now always includes the user's `y⁺`), KL anchor.

**When the user provides critique + rewrite:** best case. `y⁺` anchors the top of the rank by supervision; auxiliary rollouts are sampled `~π_old(·|x, c)` as in regular CCPD; the critique-derived `r_c` fills in the middle. This is the richest training signal the system can produce from one feedback event.

**Why this works where DPO fails.** We've replaced the differentiated log-ratio with a rank-based REINFORCE gradient pathway. The user's preference information is preserved — their `y⁺` *is* the top-rank sample, driving the largest positive advantage, and the SFT term distills it directly. What we've dropped is the specific parameterization that causes surface-cue overfitting and response-suppression dynamics.

**Preservation of the DPO intuition for users who insist.** We expose vanilla DPO as an opt-in objective via `{"objective": "dpo"}` in the training API for users who want to reproduce DPO results or who have specific reasons. The `preferred` endpoint's default routes to CCPD v2 because we think it's the better default. This matches how the whole system is designed — path of least resistance is the safer default; long-term optimal path is accessible.

**API summary** — all four feedback shapes collapse to one internal objective family:

| UI/API shape | Data | Routes to | Auxiliary sampling |
|---|---|---|---|
| `binary` (thumbs) | `(x, y, up/down)` | KTO | None |
| `nl_critique` | `(x, y⁻, c)` | CCPD v2 | k ~ π(·\|x, c) |
| `preferred` | `(x, y⁻, y⁺)` | CCPD v2 (user-seeded rank) | k−2 ~ π(·\|x) |
| `nl_critique_with_rewrite` | `(x, y⁻, c, y⁺)` | CCPD v2 (user-seeded rank, critique-scored middle) | k−2 ~ π(·\|x, c) |

This is the point of the API: **one endpoint shape, flexible online learning, every feedback type routes to a gradient-safe training path, no vanilla DPO unless explicitly asked for.** The user thinks in terms of their feedback; the system picks the objective.

---

## 5d. The method tier menu — compute/capability Pareto curve

The CCPD v2 family in §5c is the capability end of a spectrum. In production we want methods at multiple points along the compute-vs-capability curve, selectable per-request or chosen by the system based on queue pressure and feedback priority. This section sketches the full menu.

**The two axes we're trading off:**
- **Time-to-ready:** wall-clock between "feedback received" and "model stepped, inference resumed."
- **Gradient quality:** Razin-safety (is the gradient pathway through hidden states or through unembedding log-ratios?), credit assignment granularity, update variance, and generalization.

A third axis, **signal richness** (binary < critique < rewrite < critique+rewrite), is determined by the user's input, not the method. Richer signals *allow* richer methods but don't require them.

Legend: 🆕 = novel-to-lile contribution. 📖 = adapted from cited literature. 🔧 = engineering composition of known parts.

### Tier 0 — Deferred (<100ms, no gradient update)

**T0.1 Log-and-batch.** 🔧
- Write `(x, y, feedback)` tuple to the append-only trajectory log. Return 200 immediately. No training. A background task drains the buffer on schedule (idle periods, or when buffer exceeds threshold).
- **Use when:** throughput matters more than responsiveness; feedback is speculative or noisy; user isn't expecting immediate model change.
- **Gradient quality:** N/A — we haven't updated yet. When we do, it's via one of the higher tiers.

### Tier 1 — Fast (1-5s, one forward + one backward, no auxiliary sampling)

**T1.1 Weighted SFT on chosen response.** 📖 (standard)
- For `preferred` or `nl_critique_with_rewrite` feedback. Loss = `−log π_θ(y⁺|x)`, weight = user-configurable (default 3× typical SFT sample). Optional mild rejection: add `λ · log π_θ(y⁻|x)` with λ small (~0.05).
- **Gradient quality:** Razin-safe (gradient flows through policy log-prob, not log-ratios). Uncalibrated — doesn't know *how much* better y⁺ is.
- **Time:** ~1-3s for one forward+backward on a 7-14B with LoRA.

**T1.2 KTO single-step.** 📖 Ethayarajh et al. (2024)
- For `binary` feedback. TRL's `KTOTrainer` in online mode. Per-sample step.
- **Gradient quality:** KTO's own analysis; generally well-behaved, one known weakness is overweighting positive examples (§5b.1).
- **Time:** ~1-3s, similar to T1.1.

**T1.3 Chain-of-Hindsight single-step.** 📖 Liu, Sferrazza & Abbeel (2023)
- For `nl_critique` feedback. Format the sample as `"Prompt X. Bad response Y. Feedback 'c.' Good response: [end]"`. SFT on the whole sequence. If no `y⁺` is provided, CoH still trains on the `(x, y⁻, c)` prefix — it learns to *associate* the critique with the bad response, which transfers.
- **Gradient quality:** Razin-safe (plain SFT gradient). Doesn't actually produce an improved response — teaches the association. Good for bulk/noisy critique data; weak for sharp correction.
- **Time:** ~1-3s.

**T1.4 Rejection-sampling SFT.** 📖 ReST (Gulcehre et al., 2023), RFT (Yuan et al., 2023)
- No feedback needed. Periodically sample n responses to past prompts from `π_old`, filter by some criterion (heuristic, rule, or judge), SFT on the winners. Runs in the background as a "polish" pass.
- **Use when:** idle GPU, no live feedback queue. Supplementary objective.
- **Time:** depends on batch size; typically 10-60s per cycle but fully backgrounded.

### Tier 2 — Balanced (5-15s, small auxiliary budget k=2-4)

**This is the expected default for interactive feedback.**

**T2.1 CCPD v2 (light).** 🆕 §5c.11
- k=2-4 auxiliary rollouts. For `nl_critique`: sample from `π_old(·|x,c)`. For `preferred`: user's `y⁺` + 1-3 samples from `π_old(·|x)`. Rank via `r_c` (detached). REINFORCE with rank advantage + SFT on top-m + KL anchor.
- **Gradient quality:** Razin-safe by construction. Rank-based further insulates from surface cues. Calibrated: richer signal than T1.
- **Time:** ~5-10s (k short rollouts at a few hundred ms each + k+3 training-graph forwards + backward).

**T2.2 Contrastive SFT with margin.** 🔧 derived from RPO (2024), D2PO (Singhal et al., 2024)
- Middle ground between T1 and T2.1. No auxiliary sampling; use only the user's `(y⁻, y⁺)`. Loss = `−log π(y⁺|x) + λ · max(0, log π(y⁻|x) − log π(y⁺|x) + margin)`. Hinge loss with explicit margin, not log-ratio reparameterization.
- **Gradient quality:** Razin-safer than DPO — the gradient on y⁺ goes through its log-prob directly (EX-RM pathway), only the rejection term has log-ratio character, and the hinge clips it when the margin is satisfied.
- **Use when:** `preferred` feedback, aux sampling budget not available.
- **Time:** ~2-4s.

### Tier 3 — Rich (15-60s, k=8+ auxiliary samples, fine-grained credit assignment)

**T3.1 CCPD v2 (full) with trace infilling.** 🆕 §5c.13
- k=8+ auxiliary rollouts. For reasoning tasks with chain-of-thought: sample infills of the critiqued span only. Everything else from T2.1, but the gradient flows only through the infilled tokens — near-surgical credit assignment.
- **Gradient quality:** same Razin-safety as T2.1, dramatically lower variance on critique-targeted behaviors, almost no side effects on untargeted behaviors.
- **Use when:** task has structure (CoT, code, multi-step), feedback is sharp ("your step 3 is wrong").
- **Time:** ~20-40s.

**T3.2 SCoRe-style multi-turn correction.** 📖 Kumar et al. (ICLR 2025)
- When the user sends a follow-up *after* the critiqued response, the pair is itself a correction trajectory. Train in multi-turn GRPO fashion: given `(x, y⁻)`, reward π for producing a y⁺ that is judged (by `r_c`, or by the user's follow-up response as implicit target) to be better.
- **Gradient quality:** Full GRPO pathway, Razin-safe. The ICLR paper reports an absolute 15.6% gain on self-correction for MATH and 9.1% on HumanEval relative to the Gemini base. Open question whether those gains transfer to open-ended chat — they were measured on verifiable domains.
- **Use when:** natural conversation flow produced a correction trajectory; high-value feedback event.
- **Time:** ~30-60s for a small multi-turn batch.

### Tier 4 — Deliberate (minutes, runs backgrounded or off-peak)

**T4.1 Trajectory replay with re-weighting.** 🔧
- The buffer accumulates feedback over hours/days. Offline pass: re-sample, re-weight (recent more heavily, high-agreement more heavily, contradictions downweighted), run one of T2/T3 methods on the composed batch. This is how we catch up after deferred (T0) samples.
- **Use when:** overnight, or whenever the live queue is quiet.
- **Time:** minutes.

**T4.2 Self-distillation from snapshot.** 📖 self-distillation literature, e.g. Born-Again Networks (Furlanello et al., 2018) adapted for LLMs
- Sample a batch of prompts. Generate responses under a *frozen snapshot* of the current best adapter state. Train the live model toward the snapshot's logits (KL minimization). Regularizes drift, consolidates recent learning, reduces variance.
- **Use when:** model has been updated a lot recently; want to stabilize before next update burst. Periodic maintenance pass.
- **Time:** minutes for a meaningful batch.

**T4.3 Progressive-merge consolidation.** 🆕🔧 (§3.1)
- The progressive merge from §3.1, triggered automatically when the active LoRA accumulates enough updates. `dequant → merge → requantize`-safe via Unsloth's `fast_dequantize` path. Reset active adapter; continue.
- **Gradient quality:** preservation only — not a new gradient. Critical for long-horizon operation without adapter-count blowup.
- **Time:** seconds to minutes depending on model size; blocks training but not inference.

### Selection policy

The system chooses a tier based on:
1. **Explicit request.** API accepts `"tier": "fast" | "balanced" | "rich"` override.
2. **Feedback shape.** Binary → T1.2 default. Critique alone → T2.1 default. Rewrite → T2.1 or T1.1 based on queue. Critique+rewrite → T3.1 if CoT task, else T2.1.
3. **Queue pressure.** If >N pending feedback events, demote the default by one tier (rich→balanced, balanced→fast). This keeps the daemon from falling behind.
4. **Idle polish.** T1.4 (rejection SFT), T4.1 (replay), T4.2 (self-distill) run backgrounded when the live queue is empty.

This gives us **continuity across the Pareto curve** — every feedback event gets *some* update, and the system degrades gracefully from rich to fast under load rather than dropping samples or blocking the user.

### Menu summary

| Tier | Time | Method | Feedback | Gradient | Origin |
|---|---|---|---|---|---|
| T0.1 | <100ms | Log-and-batch | any | deferred | 🔧 |
| T1.1 | 1-3s | Weighted SFT | rewrite | Razin-safe | 📖 |
| T1.2 | 1-3s | KTO single-step | binary | mostly-safe | 📖 |
| T1.3 | 1-3s | CoH single-step | critique | Razin-safe | 📖 |
| T1.4 | bg | Rejection SFT | none | Razin-safe | 📖 |
| T2.1 | 5-10s | CCPD v2 (light) | critique/rewrite | Razin-safe + rank | 🆕 |
| T2.2 | 2-4s | Hinge contrastive | rewrite | mostly-safe | 🔧 |
| T3.1 | 20-40s | CCPD v2 + infill | critique, CoT | Razin-safe + surgical | 🆕 |
| T3.2 | 30-60s | SCoRe multi-turn | correction traj | Razin-safe | 📖 |
| T4.1 | min | Replay + reweight | any (buffered) | depends on inner | 🔧 |
| T4.2 | min | Self-distill | none | KL-based, safe | 📖 |
| T4.3 | min | Progressive merge | n/a | preservation | 🆕🔧 |

**Innovation claimed by lile:** the CCPD v2 family itself (§5c), the unified routing layer that puts rewrites through ranked-advantage REINFORCE instead of DPO (§5c.16), the trace-infilling variant for surgical credit assignment on CoT tasks (§5c.13/T3.1), the always-on tier-selection control plane, and the progressive-merge residual store (§3.1). Everything else is composition of well-understood prior work — which is what we want for a production system.

---

## 6. The 4-bit merge gotcha (critical)

**You cannot naively merge a 16-bit LoRA into a 4-bit base and expect it to work.** The LoRA was trained against dequantized bf16 weights; quantizing the merged result to NF4 introduces rounding errors the LoRA never saw. Results range from "slight quality loss" to "complete garbage" depending on adapter magnitude.

The correct sequence, which Unsloth already implements in `save.py`:

1. `fast_dequantize` base weights → fp32.
2. Merge: `W_merged = W_dequantized + scaling × (B @ A)` via in-place `addmm_`.
3. Cast back to target dtype (bf16 for keeping in RAM, NF4 if we re-quantize).

**Implications for lile:**

- Our `merged_deltas` must be **stored in bf16 or fp32**, not re-quantized to NF4 every time. NF4 is for the base only.
- The "live inference path" reads base (NF4) and applies merged_deltas (bf16) as a residual at forward time. Slightly slower than pure NF4 but correct.
- If the user explicitly wants a fully-requantized merged checkpoint (for smaller VRAM or GGUF export), we expose that as a separate `/v1/state/export` operation — but the live daemon state never uses that path for training.

This one constraint drives a lot of the architecture. Getting it wrong means silent quality degradation.

---

## 7. Module breakdown (to expand)

*This section is intentionally sparse in v0.1 — we'll flesh it out after we've confirmed the top-level decisions above.*

- `lile.server` — FastAPI app, routes, auth, streaming
- `lile.engine.inference` — generation path (Unsloth `fast_generate` or vLLM sidecar)
- `lile.engine.train` — wraps ht-unsloth trainers; implements the objective composer
- `lile.state` — the live model state (base + merged_deltas + active_adapter)
- `lile.queue` — compute buffer, chunking, commit tokens
- `lile.adapters` — LoRA pool management, hot-swap, progressive merge
- `lile.trajectory` — append-only log, replay, snapshot reconstruction
- `lile.snapshot` — save/restore/diff
- `lile.objectives` — SFT, PG, GRPO, KL, distillation, prompt-baking; pluggable registry
- `lile.controller` — the single writer serializing train/merge/save/load on the GPU

---

## 8. Open questions (resolve before §7 expansion)

1. **Inference engine:** Unsloth's `fast_generate` (simpler, shares weights) vs a vLLM sidecar (faster, two-process). For 1× 3090 default, fast_generate is likely right. For 2× 3090, vLLM sidecar on the serving GPU almost certainly wins.
2. **Default base models to support at launch.** Recommend: Qwen3-8B, Qwen3-14B, Qwen3-30B-A3B (MoE, for 2× 3090 users), gpt-oss-20b. All are first-class in Unsloth.
3. **Full-FT mode scope for v0.** Include or defer? Inclination: include but mark experimental; it's trivial once LoRA mode works.
4. **Reward model/judge.** For GRPO, rewards can come from (a) user-provided scalars in the API, (b) an in-process judge model, (c) a rule-based function. Probably expose all three with (a) as the baseline.
5. **Staleness policy.** You said it's low priority, but we need *something* — at minimum, a "max in-flight gradient steps without a base-KL check" guardrail.
6. **Packaging.** `pip install lile` with GPU-detect auto-install? Docker image? Both?
7. **`r_c` ranking reliability at 7B-14B scale.** The linchpin empirical question for §5c CCPD v2. See §11 — this one has to be answered *first*.

---

## 9. Phased roadmap

**Phase 0 — Decisions (this doc).** Lock name, base, module boundaries, API shape.

**Phase 1 — Skeleton + T1 methods.**
- `lile serve` stands up FastAPI + Unsloth fast_generate + a single LoRA adapter.
- `/v1/train` accepts SFT batches, chunks them, trains, guarantees next-request visibility.
- T1.1 (weighted SFT), T1.2 (KTO), T1.3 (CoH) objectives.
- Snapshot + restore work. Trajectory log is append-only JSONL.
- `/v1/feedback` endpoint routing binary/critique/rewrite/critique+rewrite to T1 defaults.

**Phase 2 — Tier selector + composer + KL anchoring.**
- Objective registry + composer with stackable loss terms.
- KL-to-base, KL-to-EMA, KL-to-snapshot.
- Distillation trainer.
- Tier-selection control plane (explicit override, shape default, queue-pressure demotion).
- T4.1 (replay), T4.2 (self-distill) backgrounded.

**Phase 3 — CCPD v2 (the innovation core).**
- Auxiliary sampling path with shared KV pool.
- T2.1 (CCPD v2 light) implementation.
- Detached `r_c` scoring, rank-advantage REINFORCE, top-m SFT distillation.
- T2.2 (hinge contrastive) as aux-sampling-free fallback.
- **The benchmark gate:** validate `r_c` ranking reliability before shipping (see §8 Q7, §11).

**Phase 4 — Rich tier + trace infilling.**
- T3.1 (CCPD v2 with trace infilling) for CoT tasks.
- T3.2 (SCoRe-style multi-turn) when correction trajectories are available.
- Auto-detection of CoT structure to route between T2.1 and T3.1.

**Phase 5 — Progressive merge + adapter pool.**
- `merged_deltas` residual store.
- `/v1/state/merge` endpoint + T4.3 auto-consolidation.
- Multi-adapter routing (per-user, per-task).

**Phase 6 — Polish for locallama release.**
- GGUF export.
- One-command install.
- Example recipes + docs.
- Two-GPU split-process mode.

---

## 10. What "done" looks like for v1.0

1. `pip install lile && lile serve --model qwen3-8b` just works on a single 3090.
2. OpenAI-compatible chat completions endpoint.
3. `/v1/train` endpoint accepting SFT, GRPO, policy gradient, distillation, and prompt-baking, with stackable per-sample and per-batch objectives.
4. Progressive merge works without quality regression (benchmark: GSM8K, MMLU, HumanEval before/after merge of N adapters).
5. Snapshot + restore is byte-exact.
6. Two-GPU mode splits serving and training cleanly.
7. Documentation good enough that a locallama user can set up a personal assistant that learns from their use in under 30 minutes.

---

## 11. The next uncertainty to zoom in on: does `r_c` actually rank?

Everything in §5c and most of §5d Tier 2/3 rests on one empirical claim: that **the implicit critique reward `r_c(y) = β·[log π_old(y|x,c) − log π_old(y|x)] / |y|` produces reliable *rankings* over π_old's own generations**, even when it fails as a differentiated loss (which is the Razin et al. finding we've accepted).

This is the single highest-leverage uncertainty in the plan. If rankings are reliable, CCPD v2 is real and lile has a defensible innovation story. If rankings are noise at our target model scales (7B-14B), we fall back to a pure T1 + T4 system, which works but doesn't have a headline feature.

**Why this matters more than the other open questions in §8.** The inference engine choice (Q1) is a well-defined engineering decision with known tradeoffs. The merge-residual latency concern (footnote to Q6) is measurable in a day. Default base model (Q2) can be chosen from a shortlist. The full-FT scope (Q3) can be deferred. All of these are *after* the ranking question, because they don't change direction if we're right but require a complete rearchitecture if CCPD v2 doesn't work.

**The experiment, concretely.**

**Inputs.**
- Pre-trained Qwen3-8B and Qwen3-14B (the likely v1.0 defaults).
- A held-out set of 200-500 `(prompt, response)` pairs where the response has a known deficiency (too verbose, factual error, wrong format, etc.).
- A matching set of 200-500 critiques — ideally a mix of short ("too verbose") and long ("this misunderstands the question because..."). Can be synthesized by a stronger model (GPT-4-class) or hand-labeled.
- For each `(x, y, c)`, a ground-truth "better" response `y*` — also synthesizable or human-labeled.

**Procedure.**
1. For each (x, y⁻, c), sample k=8 candidates `Y⁺ ~ π(·|x, c)` from the target model.
2. Score each candidate via `r_c`, detached.
3. Rank the candidates by `r_c`.
4. For each candidate, compute a ground-truth quality score: semantic similarity to `y*`, or preference judgment by a held-out stronger LLM judge, or (if the critique is verifiable) a rule-based check.
5. Measure **Spearman rank correlation** between `r_c` rank and ground-truth rank, per (x, c).
6. Aggregate.

**Decision thresholds.**
- **Spearman > 0.5 mean:** ranking is reliable. Ship CCPD v2 as the T2.1 default.
- **Spearman 0.2-0.5:** ranking is noisy but non-random. Use CCPD v2 only with k≥8 (larger k averages out noise) and keep T2.2 (hinge contrastive, no ranking needed) as the primary T2 method.
- **Spearman < 0.2:** ranking is unreliable at this scale. CCPD v2 falls back to CCPD-v1-style single-sample SFT on self-refinement. Most of the innovation claim collapses; we still have a solid T1-only system.

**Ablations worth running simultaneously** (cheap, same infrastructure):
- Spearman as a function of model size (8B vs 14B vs 30B MoE). We'd expect a monotone improvement.
- Spearman as a function of critique length (short phrase vs paragraph). Length-normalization in `r_c` should reduce sensitivity, but the experiment tells us by how much.
- Spearman as a function of k. We expect the top/bottom ends of the rank to be more reliable than the middle — if so, rank-based advantage can be made robust by only trusting the extremes.
- `r_c` computed forward (`log π(y|x,c) − log π(y|x)`) vs backward (`log π(c|x,y) − log π(c|x)` via Bayes). Check whether one is consistently better.

**Estimated cost.** 500 prompts × 8 candidates × 2 forward passes (with/without critique) on Qwen3-8B = 8000 forward passes. Roughly 30-60 minutes on a single 3090. The experiment is cheap relative to its impact on the project direction.

**Why this should be run before Phase 3 (CCPD v2 implementation) begins.** If we run it after building the full training path, we're committed to CCPD v2 code even if rankings are unreliable. Running it as a standalone probe before implementation gates the phase.

**If results are positive**, the Phase 3 work is motivated and we have preliminary evidence to include in any future writeup of the method.

**If results are negative**, we save months of implementation work on a method that wouldn't have worked, and we still ship a respectable T1 + T4 system (which is strictly better than what users can assemble from TRL + Unsloth today).

---

## Next step

Run the §11 ranking-reliability experiment. Everything downstream — Phase 3 planning, the §8 open-question resolution, the §7 module breakdown — is better-informed after we have this one number.

If the experiment lands positive, we come back and resolve §8 Q1 (inference engine) and expand §7 with confidence. If it lands negative, we rescope the plan before sinking further effort — which is exactly what we want surfaced before, not after, building the CCPD v2 stack.
