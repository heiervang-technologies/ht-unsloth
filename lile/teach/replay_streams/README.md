# replay_streams

Canonical, versioned feedback streams used as the A/B workload for every sample-efficiency experiment in `lile/docs/research/`.

The goal of a replay stream is not to approximate production traffic. It is to be a **falsifiable, reproducible feedback trajectory** that any optimizer / replay / forgetting PR can be A/B'd against under the eval harness (`lile/docs/research/eval-harness.md`).

---

## Streams

| File | Events | Kind mix | Domain mix | Owner | Status |
|---|---|---|---|---|---|
| `mixed_500.jsonl` | 500 | 40% binary / 30% nl_critique / 20% rewrite / 10% preferred | 25% each: math / code / common-sense / general | claude-opus (%4) | in progress, ETA 2026-04-23 |

Decision-of-record: the composition was negotiated with Mei (%30) and recorded in `optimizer-sample-efficiency.md` §7 / `sample-efficiency-synthesis.md` §3.

---

## `mixed_500.jsonl` — spec

### Event schema

One JSON object per line. Fields match `FeedbackRequest` at `lile/server.py:83` + `Controller.feedback_to_batch` at `lile/controller.py:308-387`. Each event is a *record*, not a train spec — the controller does the routing.

Required fields (all events):

- `response_id` — `"mixed_500-<idx>"`, monotonic.
- `feedback_kind` — one of `"binary" | "rewrite" | "preferred" | "nl_critique"`.
- `prompt` — user message, single-turn.
- `response` — the cold-model's reply (see §generation).
- `domain` — one of `"math" | "code" | "common-sense" | "general"`. Not consumed by the controller; used by the harness to slice regressions by domain.

Kind-specific fields:

- `binary`: `value ∈ {"up", "down"}`.
- `rewrite`: `better_response: str`, `weight: float = 3.0` (matches `controller.py:358` default).
- `preferred`: `chosen: str`, `rejected: str` (hinge objective).
- `nl_critique`: `critique: str`; optionally `better_response` to upgrade the record to `nl_critique_with_rewrite` (CoH good-path).

### Composition (exact counts)

Total 500 events. Composition is exact, not approximate — the generator checks counts before writing.

| Kind | Count | Objective hit | Rationale |
|---|---|---|---|
| binary | 200 | KTO | Dominant real-world feedback shape. Drives `v` under AdamW. |
| nl_critique | 150 | CoH | Stresses concern #2 (objective-mixing); surfaces PR B's falsifiable win. |
| rewrite | 100 | weighted_sft (w=3.0) | Signals the high-information semantic path. |
| preferred | 50 | hinge (margin=1.0) | Edge case; covered but not emphasized. |

By domain, each kind is split 25/25/25/25. So exactly 50 binary+math, 50 binary+code, ...; 37.5 nl_critique+math (round to 37/38/38/37 to total 150); etc. The generator rounds by largest-remainder.

### Ordering

Shuffled with `seed=42` at the top level of the file. Not domain-clustered — the point of this stream is to realistic-ally stress the optimizer's ability to interleave objectives and domains.

### Source datasets for prompts

| Domain | Source | License |
|---|---|---|
| math | [GSM8K](https://huggingface.co/datasets/gsm8k) `train` | MIT |
| code | [HumanEval](https://huggingface.co/datasets/openai_humaneval) `test` (there is no train split); [MBPP](https://huggingface.co/datasets/mbpp) `train` as fallback | MIT / CC |
| common-sense | [HellaSwag](https://huggingface.co/datasets/hellaswag) `train` | MIT |
| general | [MMLU](https://huggingface.co/datasets/cais/mmlu) diverse subjects (history, physics, biology, law) | MIT |

Prompts are sampled deterministically (seed=42) and trimmed to single-turn user messages. No multi-turn, no system prompts.

**Disjointness constraint.** The eval harness (`eval-harness.md`) uses the *same* datasets for evaluation. To prevent train-on-eval contamination, `mixed_500.jsonl` prompts are drawn from an index range disjoint from the eval harness's `--limit 250` slice. Specifically: eval uses indices `[0, 250)`; the stream uses indices `[1000, 1000+N)` where N is the per-domain budget. This is enforced by the generator and asserted in a smoke test.

### Response generation

Responses are produced by the **cold Qwen3 base model** — no LoRA adapter, no streamed feedback history. Generation parameters:

- `temperature = 0.7` — enough variance to produce the full spectrum of up/down-worthy responses; not so much that responses become incoherent.
- `top_p = 0.95`.
- `max_tokens` per domain: math=512 (GSM8K needs chain-of-thought space), code=768, common-sense=128, general=256.
- One response per prompt; no best-of-N.
- Seed: 42 forwarded into `torch.manual_seed` / `torch.cuda.manual_seed_all` and the FastLanguageModel generation call.

### Label generation

- **binary (value up/down)**: ground-truth comparison where available (GSM8K exact-match on final numeric answer; HumanEval+ test execution for code; HellaSwag label match). For common-sense / general where ground truth is fuzzy, binary labels fall back to a rubric check by a teacher model (Claude Opus 4.7 via API, or `gpt-4o-mini` as fallback). The rubric for "up" is binary and explicit: factual correctness + non-refusal. Teacher labels are cached.
- **preferred (chosen/rejected)**: two responses sampled at `temperature=0.7`; chosen is the one that passes ground truth (math/code) or the teacher rubric (common-sense/general). If both pass or both fail, the pair is dropped and re-sampled.
- **rewrite (better_response)**: generated by the teacher model with the prompt "Rewrite this response to be more correct/helpful without changing format" applied to the cold response. For math/code the teacher is instructed to produce a correct solution; for common-sense/general the teacher is instructed to write the ideal response the user would prefer.
- **nl_critique (critique)**: generated by the teacher model with the prompt "Write a 1-3 sentence critique of this response explaining what is wrong or could be improved." Critique style matches the CoH paper rubric (Liu et al., 2023): concrete, non-vacuous, specific to the response. 20% of `nl_critique` events additionally get a `better_response` from the teacher, upgrading them to `nl_critique_with_rewrite` (the CoH good-path variant).

### Non-goals

- **Not a benchmark.** n=500 is a workload, not a test set.
- **Not production traffic.** Real users do not submit 40% binary + 30% nl_critique — nl_critique is rare. The stream over-samples nl_critique deliberately because it is the objective that exercises concern #2 (see `optimizer-sample-efficiency.md` §1 concern #2).
- **Not diverse across models.** Responses come from one base model (Qwen3). Generalization to other bases is a separate question.

---

## Reproduction

```bash
# 1. Ensure daemon is up with the cold base model (no adapter, no history).
#    See lile/DESIGN.md for cold-start.

# 2. Build the stream.
cd /home/me/ht/forks/ht-unsloth
uv run python -m lile.teach.replay_streams.build_mixed_500 \
    --endpoint http://127.0.0.1:8768/v1 \
    --teacher-model claude-opus-4-7 \
    --out lile/teach/replay_streams/mixed_500.jsonl \
    --seed 42

# 3. Verify shape.
uv run python -m lile.teach.replay_streams.validate mixed_500.jsonl
```

The file is committed to the repo (~1 MB at n=500) so A/Bs are reproducible without re-running the teacher. Regeneration is only needed if the composition spec changes.

### Teacher API cost estimate

Rough back-of-envelope for claude-opus-4-7 at published rates:

- 150 nl_critique + 100 rewrite + 50 preferred re-samples + 200 binary labels (for common-sense/general only, ~100 prompts) = ~500 teacher calls.
- Avg ~800 input + 400 output tokens per call.
- ≈ $5-10 per full regeneration.

Budget-neutral. Cheaper still with `gpt-4o-mini` as the teacher; `claude-opus-4-7` is the default for label quality.

---

## Related streams (deferred)

These are not built yet. Listed so they don't show up as "missing" in future reviews.

- `binary_500.jsonl` — pure-KTO stream for baseline "concern #2 is invisible" evidence in Mei's PR B writeup.
- `rehearsal_pulse_50.jsonl` — 50-sample skill-targeted batch for the rehearsal loop (`sample-efficiency-synthesis.md` §4). One per verifiable task (hellaswag / arc-easy / arc-challenge / gsm8k / humaneval_plus).
- `long_2000.jsonl` — 2000-event stream for PR D (ScheduleFree) long-stream discrimination. Straight composition-scaled extension of `mixed_500`.

If you build one of these, add it to the table at the top and give it the same spec-README treatment.
