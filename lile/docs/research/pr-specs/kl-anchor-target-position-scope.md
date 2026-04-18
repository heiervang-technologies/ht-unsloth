# kl_anchor — target-position scope + schema-fallback exclude

**Author:** claude-opus (live-learn-architect)
**Date:** 2026-04-18
**Status:** proposed
**Priority:** P1 — blocks unlike tiered preconditions (task #19); unblocks unlike safe default
**Task:** #15
**Depends on:** nothing (PR G already extended kl_anchor with `scope` param; this is scope extension)
**Blocks:** #19, #17 (calibration sweep validates against this scope)

---

## Problem

Current `kl_anchor` has `scope ∈ {"prompt", "full_sequence"}`. For the `unlike` objective (surgical single-position push-down), neither scope closes the Razin-safety leak Cleo's B characterizes: at the surgery position, any tail token with `p_j < M_p(η)` can grow in absolute mass. Prompt-only scope doesn't touch target-position logits at all; full-sequence fights the intended surgery uniformly.

Decision (design-notes-2026-04-18 §9): the principled shape is `scope="target_position"` with the surgery tokens `{bad_token_id, good_token_id}` excluded from the anchor domain. Anchor the complement of the vocab-at-target-position, leaving the two surgery tokens free to move.

## Scope

Extend `kl_anchor` with:
- New enum value: `scope="target_position"`.
- New optional param: `exclude_token_ids: list[int] | None`.
- Per-sample schema fallback: when `scope="target_position"` and `exclude_token_ids` is not set, derive the exclude set per-sample from `{sample["bad_token_id"], sample["good_token_id"]}` (either may be absent; filter `None` out). Mirrors the `prefix`/`prompt` schema-fallback pattern landed in kl.py already.
- Explicit `exclude_token_ids` at batch level UNIONs with the per-sample derivation.

## Contract

### Input

```json
{
  "objective": "unlike",
  "samples": [{"prefix": "...", "bad_token_id": B, "good_token_id": G}],
  "batch_objectives": [
    {"name": "kl_anchor", "scope": "target_position", "weight": w}
  ]
}
```

The above is equivalent to `exclude_token_ids=[B, G]` at the batch-objective level (schema-derived).

### Computation

1. Forward `pi_ref` (ref model) and `pi_theta` (trainee) over the batched prefix tokens.
2. Gather logits at each sample's last real token position (same as unlike's existing path; share the indexing helper).
3. Mask out the per-sample exclude token IDs in both distributions.
4. Renormalize both distributions over the un-excluded vocab.
5. KL( pi_ref_masked || pi_theta_masked ) — the standard anchor direction.
6. Scale by `weight`, add to the batch loss.

### Implementation notes

- Re-use the forward pass from `unlike` when both land in the same batch. Hook into the existing logits tensor rather than re-forwarding. Token-position indexing is the same.
- For the excluded tokens, use `logits.masked_fill(mask, -inf)` before softmax so the KL compute is clean.
- No gradient flows through `pi_ref` — already handled by the existing anchor path under `disable_adapter()`.

## Tests

1. **Schema fallback parity.** `scope="target_position"` with no explicit `exclude_token_ids` and samples carrying `bad_token_id=10, good_token_id=20` produces the same loss as the same call with explicit `exclude_token_ids=[10, 20]`. Pin in `test_kl_target_position.py`.
2. **Explicit-list union.** Per-sample derives `{10, 20}`, explicit batch-level adds `[30]`, the union `{10, 20, 30}` is applied. Loss matches manual computation.
3. **No-exclude fallback.** Samples without `bad_token_id`/`good_token_id` + no explicit exclude → exclude set empty → standard target-position anchor over full vocab (should still run; just a different safety profile).
4. **Gradient zero on excluded tokens.** Backward through the anchor loss produces zero gradient on the logits of excluded token IDs at the target position. Pinned because the guarantee is the whole point.
5. **Existing `scope="prompt"` / `scope="full_sequence"` unchanged.** Regression guard.

## Non-goals

- No new objective. kl_anchor stays a single composition primitive.
- No per-sample scope — scope is batch-level; the *exclude set* is per-sample via fallback.
- No dynamic exclude based on rank/prob at runtime. Keep the primitive static; dynamic surgery-token selection belongs in userland if it's ever needed.

## Rollout

Single PR to `lile/objectives/kl.py` + new test file. No route changes, no schema migration (existing callers unaffected; new branch is opt-in via scope string).
