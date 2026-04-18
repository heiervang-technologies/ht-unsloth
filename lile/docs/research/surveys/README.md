# `lile` research surveys

This folder holds **literature surveys** that motivate the design and PR backlog for the `lile` LiveLearn daemon. Each file is a compact, annotated scan of a specific venue or theme — built from focused research passes, curated against what's already in `lile/docs/research/sample-efficiency-lit-review.md` and `optimizer-sample-efficiency.md`.

The surveys are **motivation documents, not bibliographies**. Every entry is tagged:

- **[STRONG]** — directly changes a design decision or unlocks a new PR candidate.
- **[RELEVANT]** — worth reading; informs an existing decision but doesn't force a change.
- **[BACKGROUND]** — context only; not immediately actionable.
- **[SKIP]** — recorded so we don't re-scan it later.

## Index

| Survey | Scope | Status | Feeds into |
|---|---|---|---|
| [`neurips-2025-sample-efficient.md`](neurips-2025-sample-efficient.md) | NeurIPS 2025 — TTT, self-improvement, continual LoRA, off-policy RL | complete | synthesis §forgetting, synthesis §RL, optimizer §4 |
| [`acl-naacl-2025.md`](acl-naacl-2025.md) | ACL / NAACL / TACL 2025 — selective loss, personalization, continual SFT | complete | lit-review §token-weighting, synthesis §personalization |
| [`meta-learning-llm-adaptation.md`](meta-learning-llm-adaptation.md) | Meta-learning for LLMs 2024–2026 — MAML, hypernet LoRA, TTRL | complete | synthesis §meta-init, potential new PR: T2L/HyperLoRA probe |
| [`optimizer-landscape-2025-2026.md`](optimizer-landscape-2025-2026.md) | 8-bit optimizer stability, AdEMAMix, Muon, SF-AdamW caveats | complete | optimizer §2 table, §3 diff sketch, §4 PR D/E |
| [`datainf-streaming-lora.md`](datainf-streaming-lora.md) | Influence functions for LoRA — DataInf, LESS, GREATS gap | complete | synthesis §per-sample-weighting, speculative `lile/teach/datainf_probe.py` |
| [`8bit-optimizer-stability.md`](8bit-optimizer-stability.md) | bnb version guidance, block size, stability regressions | complete | `pyproject.toml` pin, optimizer §keep-as-fallback |
| [`icml-2025.md`](icml-2025.md) | ICML 2025 — FLOW, ConfPO, Multi-Ref KL, PILAF, Flat-LoRA, pretraining-inject scaling law | complete | optimizer §2, synthesis §forgetting, synthesis §active-query |
| [`iclr-2025-2026.md`](iclr-2025-2026.md) | ICLR 2025 + 2026 — In-Place TTT, Meta-UCF, Uni-DPO, Spurious Forgetting | complete | synthesis §TTT, synthesis §hypernet, synthesis §forgetting |
| [`emnlp-2025.md`](emnlp-2025.md) | EMNLP 2025 — implicit-feedback noise, PPALLI, PRIME, LimaCost, VerIF | complete | synthesis §user-feedback, synthesis §personalization |
| [`colm-2025.md`](colm-2025.md) | COLM 2024/25 — LoRI, OCRM, collaborative self-play, D2PO, Self-Rewarding | complete | synthesis §RL, synthesis §LoRA-sparsity |
| [`test-time-rl-2025.md`](test-time-rl-2025.md) | Test-time RL / self-improvement — TTRL, SEAL, rStar-Math, PAVs | complete | synthesis §RL, `production-implementation-roadmap.md` §verifier-gated |

## How to use these docs

1. **Before opening a new PR**, grep the surveys for the technique. If a survey already cites prior art either supporting or killing the idea, cite it in the PR description.
2. **When writing motivation sections** of design docs (`optimizer-sample-efficiency.md`, `sample-efficiency-synthesis.md`), pull `[STRONG]` findings into the body and put the survey on the References list.
3. **When a `[STRONG]` finding has no PR yet**, add it to the "open threads" section of the nearest design doc so it doesn't rot on the shelf.
4. **Shipping targets** — `[STRONG]` findings from these surveys are mapped to concrete PRs in [`../production-implementation-roadmap.md`](../production-implementation-roadmap.md). That document is the authoritative translation of research into code.

## Authorship

Surveys are compiled from dispatched research-agent passes; each file names the scope + date + agent ID at the top. Findings are cross-checked against the existing lit review before being labeled `[STRONG]`.
