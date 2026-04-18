# unlike — tiered anchor preconditions

**Author:** claude-opus (live-learn-architect), Mei (tier design)
**Date:** 2026-04-18
**Status:** proposed
**Priority:** P1 — gates pure-unlike safety; composes with task #15
**Task:** #19
**Depends on:** #15 (kl_anchor target-position scope must exist)
**Blocks:** nothing

---

## Problem

`unlike` as currently shipped will run pure push-down (no `good_token_id`, no KL anchor) silently. Cleo's razin-safety-sharpened.md theorem (2026-04-18) establishes that a pure SFT-family step under small η has a non-empty grower set of tail tokens `{j : p_j < M_p(η)}` — the Sauron-grows-while-we-silence-Voldemort failure mode. Pure-unlike inherits this mechanism and amplifies it (push-down on the dominant shrinks partition faster, raising `M_p(η)`).

Silent-proceed is the wrong default. We need runtime preconditions with enough granularity to distinguish genuine misconfiguration from research use.

## Tiered design (Mei)

Three cases, three severities. Applied only to **pure-unlike** samples (samples with `good_token_id=None`). Samples with a positive teacher are SFT-family dominant and not subject to tier-1 error (warnings still apply).

### Tier 1 — Error (hard refusal)

**Condition:** sample is pure-unlike AND `batch_objectives` contains no entry with `name="kl_anchor"`.

**Action:** raise `ValueError("pure-unlike requires kl_anchor in batch_objectives (see GLOSSARY.md / design-notes-2026-04-18.md §9); pass allow_unanchored=True to override")`.

**Escape hatch:** set `allow_unanchored=True` at the batch level to suppress. Intended for research / adversarial-testing / ablation workflows that explicitly want raw push-down.

### Tier 2 — Warn (proceed)

**Condition:** sample is pure-unlike AND `kl_anchor` is present with `scope="prompt"` or `scope="full_sequence"` (any scope other than `"target_position"`).

**Action:** `warnings.warn("pure-unlike detected with kl_anchor scope='<scope>'; the anchor does not brake target-position mass movement. Consider scope='target_position' (see design-notes §9).", RuntimeWarning)`.

### Tier 3 — Warn (proceed)

**Condition:** sample is pure-unlike AND `kl_anchor` has `scope="target_position"` AND the per-sample derived exclude set does NOT include all surgery tokens (should be impossible with the schema fallback, but a manual batch-level `exclude_token_ids` could omit them).

**Action:** `warnings.warn("kl_anchor scope='target_position' but exclude_token_ids does not cover the surgery tokens; the anchor fights the push-down at the bad/good positions, producing a muddier gradient.", RuntimeWarning)`.

## Where the check lives

Preconditions are checked in `unlike_loss` at batch-prep time, before any forward pass. Checks are pure-python on the request spec — zero tensor cost.

Pseudocode:

```python
def _check_preconditions(samples, batch_objectives, allow_unanchored):
    pure = [s for s in samples if s.get("good_token_id") is None]
    if not pure:
        return
    anchors = [bo for bo in (batch_objectives or []) if bo.get("name") == "kl_anchor"]
    if not anchors and not allow_unanchored:
        raise ValueError(...)
    for bo in anchors:
        scope = bo.get("scope", "prompt")
        if scope != "target_position":
            warnings.warn(...)
            continue
        batch_exclude = set(bo.get("exclude_token_ids") or [])
        for s in pure:
            sample_exclude = {s.get("bad_token_id"), s.get("good_token_id")} - {None}
            if not sample_exclude.issubset(batch_exclude | sample_exclude):
                warnings.warn(...)
```

## Tests

1. **Pure-unlike no anchor → error.** No kl_anchor in batch_objectives, samples are pure-unlike → raises ValueError.
2. **Pure-unlike with allow_unanchored=True → runs.** Same setup + `"allow_unanchored": True` → runs without raising, returns loss dict.
3. **Pure-unlike with scope=prompt → warn + runs.** Warning text contains "scope='prompt'".
4. **Pure-unlike with scope=target_position → no warning.** Clean path.
5. **With positive teacher + no anchor → no error.** SFT-family dominant; no tier-1 fire.
6. **With positive teacher + no anchor → no warning either.** We don't warn on non-pure-unlike; the positive teacher is its own "concrete target" safety signal.
7. **Existing unlike tests still pass.** Regression guard — current `test_unlike_loss.py` should continue to green with no changes (all its test cases set `good_token_id` OR pass through kl_anchor already).

## Rollout

Land after #15 (kl_anchor target-position scope). Single-file change (`lile/objectives/unlike.py`) + new test cases appended to `test_unlike_loss.py`. Zero schema change; `allow_unanchored` is a new optional key that defaults to `False`.

## Documentation

- Update `unlike.py` module docstring to list the three tiers explicitly.
- Add a short note to `GLOSSARY.md` `unlike` row: "enforces tiered anchor preconditions (error/warn/warn); pure-unlike without kl_anchor raises unless allow_unanchored=True."
