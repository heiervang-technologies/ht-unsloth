# lile design notes — 2026-04-18

Working notes captured during a thinking-aloud design session. Not spec, not decided — input to later design docs / issues / PRs.

## 1. CPU RAM is fair game for non-training ops

- `state.py:144-146` already keeps `merged_deltas` in CPU bf16 between merges.
- Snapshot save/load goes through safetensors (mmap).
- No cap enforced on CPU RAM.
- Implication: caching both cold-baseline and trained adapter tensors in CPU RAM to make on/off swaps near-instant is within the invariant. Useful for scaling-curve A/Bs where one cold baseline amortizes across N cursors.

## 2. Training scripts — where do they live?

Two options discussed:

**A. Repurpose Recipe Studio tab.** Its graph/executions abstractions map to multi-step flows (`train-N → snapshot → eval → repeat`). But current nodes are data-flow oriented, so it's a real rework, not a veneer.

**B. New "Runs" panel.** Shells out to named scripts in `lile/teach/` and streams stdout. Lighter, more faithful to existing pattern where scripts drive the daemon.

Not decided. User leaning toward B implied but not final.

## 3. RLAIF, not distillation

Current `lile/teach/tutor/run.py` is SFT-on-gold (tutor produces full correct answers, student SFTs on them).

User wants: **tutor-as-critic with varying feedback intensity** (hints → partial corrections → full rewrites).

Two signal paths:
- **SFT-on-corrected** (simplest): student generates → tutor critiques + emits correction at sampled intensity → POST the pair to `/v1/train` as SFT. New script `lile/teach/rlaif/run.py`, no new daemon primitives.
- **Preference pairs** (richer): tutor emits winner/loser pairs → DPO objective. Needs a DPO objective wired into `lile/objectives/` (KTO exists, DPO doesn't).

Open question: which signal path.

## 4. Surgical correction via negative-token SFT

User's example:
> Rule: we don't talk about Voldemort.
> Prompt: "The antagonist is"
> Model's `argmax(logits) = "Voldemort"`.
> After one feedback, model should answer: "The antagonist is he who shall not be named."

Mechanism: **unlikelihood training** (Welleck et al. 2019) applied surgically at a single position.

- Push `logit("Voldemort" | "The antagonist is ")` down.
- KL anchor over the rest of the vocab so nothing else drifts.
- Pair with a **positive teacher** at the same position ("he who shall not be named") so the model learns what *should* go there, not just what shouldn't — otherwise runner-up token may be equally wrong.

**Trigger refinement (2026-04-18):** fire the correction not only when the bad token is *argmax*, but whenever it is *a threat* at a position — i.e. `rank(bad_token, logits_t) < K` **or** `p(bad_token | prefix_t) > P`. Argmax-only is too narrow:
- Misses near-miss cases where the model would have picked the bad token ~30% of the time.
- Doesn't generalize across prefixes — one rule ("never generate Voldemort") should apply wherever the token is a likely candidate, not just where it topped the distribution.

Loss: sum the unlikelihood push-down over all positions where the trigger fires, KL anchor everywhere else. This converts a single feedback into a batched, prefix-generalized correction.

Status: no unlikelihood objective in `lile/objectives/` yet (we have sft, hinge, kto, ccpd, coh, ntp, kl). Would be a new `objectives/unlike.py` + daemon exposes it as a new `objective="unlike"` primitive.

Primitive shape (proposal):
```
POST /v1/train
{
  "objective": "unlike",
  "prefix": "The antagonist is ",
  "bad_token_id": 12345,           // the token to push down
  "good_token_id": 54321,          // optional: positive teacher
  "margin": ...,
  "kl_weight": ...
}
```

Script owns when/why to call it; daemon owns execution + safety.

## 5. Primitive design principles

Primitives are a contract. Right primitives → scripts stay thin. Wrong primitives → scripts reach around the API (importing internals, monkey-patching, forks).

**Criteria:**
- **Atomic** — one grad step / one snapshot / one generation.
- **Orthogonal** — no duplication.
- **Expressive** — open-ended script space.
- **Versioned** — breaking change = version bump.

**Expressivity test:** can we express SFT + DPO + unlikelihood + eval + scaling-curve + RLAIF loops without the daemon knowing any of those words? If yes, primitives are orthogonal.

Every new script that wants a feature is evidence: either we have the right primitives, or we're missing one.

## 6. Dataset training — client-side, not server-side

Two options discussed:

**A. Server-managed background job.** `POST /v1/dataset/train {dataset_uri, epochs, ...}` → server iterates internally, client polls for progress.
- Bloats primitive set: every training style (SFT, DPO, curriculum, RLAIF, unlike) becomes a server-side concept.
- Violates "daemon owns primitives, scripts own logic."

**B. Client-side long-running script** (selected).
- Script runs under tmux / nohup / systemd.
- Sends `/v1/train` batches atomically.
- Daemon's queue + mode_lock interleaves training with concurrent `/v1/chat/completions` so user can still talk to the model mid-train.
- Resilience via snapshots: script dies → relaunch from last commit cursor.
- Matches existing `tutor/run.py` pattern.

**Server-side things worth adding:**
- Observability — e.g. `/v1/commits` SSE stream of per-step losses so any client can attach and watch.
- Training loop itself stays in userland.

## Summary — what we'd add / build

Short list surfaced so far:
1. `objectives/unlike.py` — surgical negative-token SFT with positive teacher + KL anchor.
2. Daemon primitive `objective="unlike"` on `/v1/train`.
3. `lile/teach/rlaif/run.py` — tutor-as-critic loop (SFT-on-corrected path as the MVP).
4. Observability: `/v1/commits` SSE or similar.
5. Studio UI: either repurpose Recipe Studio or new "Runs" panel (undecided).
6. (Optimization) CPU-RAM cache of multiple adapter tensors for instant on/off swap in scaling-curve A/Bs.

None of these are decided — this is a session capture.

## 7. Shipped this session (architect role)

- **`lile/objectives/unlike.py`** — surgical unlikelihood objective. Single-position negative-token loss with rank-K or prob-P trigger; optional positive teacher; composes with KL anchor at batch level.
- **Registered `"unlike"` in `OBJECTIVES`** (`lile/objectives/__init__.py`).
- **Tests:**
  - `lile/tests/test_unlike_trigger.py` — 8 unit tests on `_should_trigger` (rank-only, prob-only, both, boundary conditions, error cases). All green.
  - `lile/tests/test_unlike_loss.py` — 5 e2e stub-model tests: triggered → positive loss; non-triggered → zero; positive teacher fires independent of trigger; mixed-batch trigger counting; gradient flows into model. All green.
- **Next:** wire into daemon `/v1/train` dispatch (client sends `{objective: "unlike", samples: [...]}`) — already routed via `OBJECTIVES` registry through `TrainEngine.step`, so no server-side change needed unless we want a schema-validated request model.

### Post-review fixes (Mei, 2026-04-18)

1. **KL-anchor schema composition.** `kl._sample_text` previously did `prompt = s["prompt"]`; unlike samples carry `"prefix"`. Fixed with a one-line fallback `s.get("prompt") or s.get("prefix") or ""`. Tests extended in `test_kl_scope.py` (3 new cases: prefix-only, prompt-wins-when-both, empty-dict-safe). Keeps both schemas honest without forcing unlike callers to duplicate fields.
2. **smoke_objectives entry.** Added three `unlike` steps to `smoke_objectives.py`: pure push-down, with-positive-teacher, and `unlike + kl_anchor` batch-level composition (verifies the schema fallback on a real model end-to-end).
3. **GLOSSARY.md row.** Added `unlike` to the Razin-safety table: `depends` — with positive teacher it's SFT-family safe; pure-unlike requires KL anchor (not optional).
4. **Docstring clarification.** Unlike module docstring now spells out the Razin-safety condition and the prompt/prefix schema fallback.
5. Deferred (non-blocking): vectorize trigger_mask. Only matters at B > 32; daemon path is typically B ∈ [1, few].

## 8. Cross-reference — related specs

- **PR L safety follow-ups** (Mei, 2026-04-18): `lile/docs/research/pr-specs/ttrl-safety-followups.md` — verifier-gate / consensus-drift / adversarial-prompt-guard required before `ttrl_pseudo_reward=True` flips live. L.1a + L.1b + L.2 + eval-gate upgrade are the internal gate; L.3 gates external. Orthogonal to the unlike objective track.
- **Razin-safety sharpening** (Cleo/Mei, 2026-04-18, in flight): `lile/docs/research/proofs/razin-safety-sharpened.md`. Counterexample Mei verified by hand: V=3, p=(0.1, 0.89, 0.01), SFT target=0, η=1 → q=(0.396, 0.588, 0.016). Small non-target grew in *absolute* mass (0.01→0.016), not just relative share. Mechanism: dominant non-target crowded, partition function shrinks, small non-target gains. The "SFT cannot shift mass to unintended outputs" reading has a hole. Cleo's deliverables: B (formalize the sharpened claim, characterize which non-targets can grow), A (one-step LR bound for unlike + KL anchor).

## 9. KL anchor scope for unlike (decision, 2026-04-18)

Triggered by Cleo's counterexample above. The Razin-safety leak is at the *target position*: small-prior tokens can grow in absolute mass when a dominant non-target is crowded. Prompt-only KL anchor (current `scope="prompt"`) does not touch target-position logits, so does not close that gap.

**Decision:** the principled default anchor for unlike is **scope="target_position" with exclude_token_ids=[bad, good]** — anchor at the single target position over the vocab complement of the two surgery tokens. Uniform anchor at target position would fight the intended surgery; masking the surgery tokens out resolves the tension cleanly.

**Primitive-set implication:** `kl_anchor` gains one new scope (`"target_position"`) and one new argument (`exclude_token_ids: list[int] | None`). Not a new primitive — an extension of an existing one, keeping the composition boundary clean. Tracked as PR G follow-up.

**Composition shape (proposal):**
```json
{
  "objective": "unlike",
  "samples": [{"prefix": "...", "bad_token_id": B, "good_token_id": G, ...}],
  "batch_objectives": [
    {"name": "kl_anchor", "scope": "target_position",
     "exclude_token_ids": [B, G], "weight": w}
  ]
}
```

**Tiered preconditions (Mei, 2026-04-18):** unlike.py enforces three severity tiers on pure-unlike calls (samples with no `good_token_id`):

1. **No `kl_anchor` at all** → **error**. This is the Razin-unsafe configuration Cleo's B names (Sauron silently grows). Escape hatch: `allow_unanchored=True` in the batch spec for research / adversarial-testing workflows that want raw push-down.
2. **`kl_anchor` present, scope="prompt"** → **warn and proceed**. Anchor exists, just on the wrong slice — likely a "forgot to set scope" case. Surgery still lands; the brake is weaker than ideal but not silent-growth.
3. **`kl_anchor` scope="target_position" but missing the surgery tokens in the exclude set** → **warn and proceed**. The anchor fights the surgery at the bad/good positions, producing a muddier gradient but not a correctness failure.

Non-pure-unlike calls (with `good_token_id` set) are SFT-family dominant and not subject to the tier-1 error — warnings still apply at the same granularity.

**Per-sample `exclude_token_ids` via schema fallback.** `kl_anchor` with `scope="target_position"` reaches into each sample's `bad_token_id` and `good_token_id` to derive the exclude set, mirroring the `prefix`/`prompt` schema-fallback pattern. This keeps the `batch_objectives` entry clean (`{"name": "kl_anchor", "scope": "target_position", "weight": w}`) — zero new schema surface on the caller side. An explicit `exclude_token_ids` at batch level is allowed and unions with the per-sample derivation, preserving primitive orthogonality for non-surgery workflows.

**Downstream:** Cleo's A (one-step LR bound for unlike + anchor) is tied to this scope; B (characterization of which non-targets can grow without an anchor) is independent. GLOSSARY `unlike` row holds at "depends — pure-unlike requires KL anchor" until Cleo's razin-safety-sharpened.md lands, then rewritten with the target-position-anchor qualifier.

## 10. Observability: /v1/commits SSE — next primitive after unlike dispatch

**Status:** lifted to highest-priority backend work in the architect queue. Pushed by Mei and agreed.

**Why now:**
1. **Byte-determinism live detection.** Current compare-json is offline — we only catch flip count after a full eval run (hours). An SSE stream of commit events with per-step loss gives live drift detection the moment a flip happens, cutting diagnosis time from runs to seconds.
2. **RLAIF real-time feedback loop.** Tutor-as-critic (`lile/teach/rlaif/run.py`) needs low-latency commit confirmation to issue the next batch. Polling `/v1/state` is wrong-grained; SSE is the right primitive.
3. **Multi-client observability.** Any number of clients can attach (debugger, dashboard, another script) without adding load on the hot path; server emits once, fans out.

**Primitive shape (draft):**
```
GET /v1/commits/stream  → text/event-stream
  each event:
  data: {"cursor": N, "ts": ..., "objective": "...", "loss": ..., "components": {...}}
```

One atomic event per commit cursor advance. No filtering, no replay buffer (clients handle their own backfill via `/v1/state/snapshot` if needed). Emitted from the same single-writer path that advances the cursor, so ordering is guaranteed.

**Implementation sketch:** an `asyncio.Queue` per connected client, fed from the commit-cursor writer in `Controller`. Drop-on-full (bounded queue) with a warning header — we never block the training path on a slow client.

**Not in scope:** chat-event SSE, trajectory replay, filter expressions. Those are userland concerns and can layer on top of the basic cursor stream.

## 11. Defaults calibration sweep — rank_below / prob_above

**Status:** queued, launches after trained_500_det finishes (GPU freed).

**Problem:** `default_rank_below=5` and `default_prob_above=0.1` are educated-guess defaults in `unlike.py`. Good enough for a smoke test but unvalidated against the actual household-correction use case.

**Sweep design:**
- **Model:** Qwen3-0.6B (matches smoke scale, fast iteration).
- **Corpus:** ~20 hand-picked household-correction prefixes, each paired with a `bad_token` (the mistake we want to surgically remove) and optionally a `good_token` (positive teacher). Structured in `lile/teach/rlaif/calibration_corpus.jsonl`. Domains: proper-name avoidance ("Voldemort"), factual corrections ("capital of Australia is Sydney" → "Canberra"), safety-adjacent tonal ("I hate" at bad positions).
- **Sweep grid:** `rank_below ∈ {1, 3, 5, 10, 20, ∞}` × `prob_above ∈ {0.01, 0.05, 0.1, 0.3, ∞}` (∞ = disable that criterion).
- **Metrics per cell:**
  - **Trigger rate** on the calibration corpus (what fraction of samples fire)
  - **False-fire rate** on a held-out neutral-prompt set (should be ~0 — firing where the bad token wasn't a threat)
  - **Post-correction drift** via KL anchor components on a neutral eval set
  - **Single-shot correction success** (does one step make the bad token stop being argmax)
- **Output:** `lile/docs/research/unlike-defaults-calibration.md` with the recommended defaults and a tuning table for callers who want non-default trade-offs.
- **Gate:** defaults change only if the sweep shows a cell that strictly dominates the current (5, 0.1) on both trigger-rate + false-fire-rate. Otherwise current defaults hold and the doc publishes the table as reference.

Does not block unlike primitive dispatch work — SSE and dispatch schema are higher priority.

## 12. Primitive-set summary (post-decisions 2026-04-18)

After the decisions above, the primitive set for the near-term is:

- `/v1/train` with `objective ∈ {sft, weighted_sft, kto, coh, hinge, cppo, ccpd_v2, ntp, unlike}` + `batch_objectives` composition (kl_anchor extended with `scope="target_position"` + `exclude_token_ids`).
- `/v1/chat/completions` (existing).
- `/v1/state/snapshot/{save,load}` (existing).
- `/v1/commits/stream` (new, priority-1).

**What's not in the primitive set (intentional):** dataset iterators, epoch management, curriculum, RLAIF loop state, scaling-curve orchestration, any tutor model. All userland, all shipping as scripts in `lile/teach/`.

That's the full contract. Anything a future script needs beyond this is evidence the primitive set is missing something — not evidence the daemon should be extended.
