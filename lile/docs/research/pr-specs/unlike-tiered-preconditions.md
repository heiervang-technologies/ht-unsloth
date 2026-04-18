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

Four cases, four severities. Tiers 1–3 apply only to **pure-unlike** samples (samples with `good_token_id=None`); samples with a positive teacher are SFT-family dominant and not subject to tier-1 error, though tier-4 (LR-floor) still applies because the unsafe-small-η regime is a consequence of the positive-teacher side, not the push-down side.

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

### Tier 4 — Warn (proceed) — small-η safety floor

**Condition:** the effective LR for this `unlike` step is below a hard-coded heuristic floor of `5e-5`. Applies to **all** unlike samples (pure-unlike AND positive-teacher), because Cleo's razin-safety-sharpened.md theorem shows the unsafe-small-η failure mode lives on the SFT-on-good side, not the push-down side. Effective LR is `cfg.per_objective_lr.get("unlike", cfg.default_lr)`.

**Action:** `warnings.warn(f"unlike dispatched with effective_lr={effective_lr:g} < 5e-5 (known-unsafe regime — the positive-teacher side can push p_bad UP; see unlike.py docstring and GLOSSARY). Override via per_objective_lr={{\"unlike\": 5e-5}} or higher. This heuristic floor will be replaced by Cleo's closed-form eta_min when task A lands.", RuntimeWarning)`.

**Why a runtime warn rather than a hard error or a config-default flip:**
- **Not an error:** a 1e-5 LR with unlike is not *guaranteed* to fire the grower-set collision on `p_bad` — it's a distribution-conditional risk. Erroring would refuse every legitimate lower-LR experiment.
- **Not a config-default flip:** `cfg.default_lr=1e-5` is the SFT default and calibrated for that family. Flipping globally affects every objective. Flipping just `per_objective_lr["unlike"]` as a built-in is fragile because any user override that omits `"unlike"` silently re-opens the trap. The dispatch-time warn catches every path.
- **Upgrade path:** when task A's closed-form `eta_min(p_bad, p_good, V, ...)` lands, Tier 4 upgrades from a static `5e-5` heuristic to a per-sample `eta_min` check. The warning surface stays the same; only the threshold is swapped.

## Where the check lives

Preconditions are checked in `unlike_loss` at batch-prep time, before any forward pass. Checks are pure-python on the request spec — zero tensor cost. Tiers 1–3 run on the samples-vs-batch_objectives structure; Tier 4 reads the effective LR for the `"unlike"` key from the `TrainEngine` / config and runs independently of sample purity.

Pseudocode:

```python
_UNLIKE_LR_HEURISTIC_FLOOR = 5e-5  # upgrades to Cleo A's eta_min when it lands

def _check_preconditions(samples, batch_objectives, allow_unanchored, effective_lr):
    # Tiers 1-3: pure-unlike anchor shape checks.
    pure = [s for s in samples if s.get("good_token_id") is None]
    if pure:
        anchors = [bo for bo in (batch_objectives or []) if bo.get("name") == "kl_anchor"]
        if not anchors and not allow_unanchored:
            raise ValueError(...)  # tier 1
        for bo in anchors:
            scope = bo.get("scope", "prompt")
            if scope != "target_position":
                warnings.warn(...)  # tier 2
                continue
            batch_exclude = set(bo.get("exclude_token_ids") or [])
            for s in pure:
                sample_exclude = {s.get("bad_token_id"), s.get("good_token_id")} - {None}
                if not sample_exclude.issubset(batch_exclude | sample_exclude):
                    warnings.warn(...)  # tier 3

    # Tier 4: small-η safety floor — applies to pure AND positive-teacher samples.
    if effective_lr < _UNLIKE_LR_HEURISTIC_FLOOR:
        warnings.warn(...)  # tier 4
```

`effective_lr` is passed in from `TrainEngine.step` so the primitive doesnt reach into the config directly. For AdamW the nominal LR is the right first-order proxy (Cleo's theorem is SGD-exact; AdamW is a lower-bound heuristic, cross-references the safety_monitor spec's Known-approximation block).

## Tests

1. **Pure-unlike no anchor → error.** No kl_anchor in batch_objectives, samples are pure-unlike → raises ValueError.
2. **Pure-unlike with allow_unanchored=True → runs.** Same setup + `"allow_unanchored": True` → runs without raising, returns loss dict.
3. **Pure-unlike with scope=prompt → warn + runs.** Warning text contains "scope='prompt'".
4. **Pure-unlike with scope=target_position → no warning.** Clean path.
5. **With positive teacher + no anchor → no error.** SFT-family dominant; no tier-1 fire.
6. **With positive teacher + no anchor → no warning either.** We don't warn on non-pure-unlike for tiers 1-3; the positive teacher is its own "concrete target" safety signal. Tier 4 still applies (see test 8).
7. **Existing unlike tests still pass.** Regression guard — current `test_unlike_loss.py` should continue to green with no changes (all its test cases set `good_token_id` OR pass through kl_anchor already). Pass `effective_lr >= 5e-5` in the existing tests' dispatch wrapper so no tier-4 noise fires on the regression suite.
8. **Tier 4 fires on small-η positive-teacher sample.** Sample with `good_token_id` set, no pure-unlike issues, `effective_lr=1e-5` → single RuntimeWarning whose message contains "effective_lr=1e-05" and "known-unsafe regime". No error. Step completes.
9. **Tier 4 does not fire at the floor.** `effective_lr=5e-5` exactly → no warning. `effective_lr=5.1e-5` → no warning. Strictly-below-floor condition.
10. **Tier 4 + Tier 1 can fire together.** Pure-unlike + no anchor + `effective_lr=1e-6` → ValueError (tier 1); tier 4 check is structured to run only when tier 1 did not raise, so the user sees the more urgent error first. Verify tier 4 does NOT warn before the raise.

## Rollout

Land after #15 (kl_anchor target-position scope). Single-file change (`lile/objectives/unlike.py`) + new test cases appended to `test_unlike_loss.py`. Zero schema change; `allow_unanchored` is a new optional key that defaults to `False`.

## Documentation

- Update `unlike.py` module docstring to list the four tiers explicitly.
- Add a short note to `GLOSSARY.md` `unlike` row: "enforces tiered preconditions (error/warn/warn/warn); pure-unlike without kl_anchor raises unless allow_unanchored=True; effective_lr < 5e-5 emits a known-unsafe-regime warning (tier 4 floor heuristic, upgrades to Cleo A's eta_min when that lands)."

## Follow-up (not this PR)

- **Per-objective safe-LR registry.** If a second objective grows a theoretical safe-LR floor (e.g. a future DPO variant), Tier 4's hardcoded `_UNLIKE_LR_HEURISTIC_FLOOR` generalizes to a registry `_OBJECTIVE_SAFE_FLOORS: dict[str, float]` consulted by `TrainEngine.step`. Deferred until N=2 — premature for one objective.
- **Tier 4 upgrade:** when Cleo's task A produces the closed-form `eta_min(p_bad, p_good, V, λ_kl, ε)`, Tier 4's floor becomes per-sample rather than global, and the warning message includes the exceeded predicted bound.
