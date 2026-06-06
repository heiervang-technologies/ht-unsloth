# Rebase resumption guide

Working notes from the 2026-06-01 rebase rehearsal. Captures the conflict patterns observed when replaying 16 HT commits onto `upstream/main` (~396 commits ahead). Use this when picking up the actual catch-up rebase blocked by the `HAI_GH_PAT` 403 (see `HT-CHANGELOG.md` Unreleased).

## Before you start

1. Pre-flight tag: `git tag pre-sync-$(date +%Y%m%d-%H%M)` on current `ht` HEAD and push it. This is the rollback anchor.
2. Make a fresh worktree (never rebase directly on `ht`): `git worktree add -b chore/rebase-YYYY-MM-DD .worktrees/ht-rebase ht`.
3. `cd .worktrees/ht-rebase && git fetch upstream main`.
4. Open this guide in another pane.

## Conflict map (from 2026-06-01 merge-tree preview)

25 files truly conflict. 81% in `studio/`. Distribution:

- **Studio frontend (16 files)**: chat integration, training types/config, training UI, sidebar, locale files
- **Studio backend (6 files)**: `main.py`, `routes/__init__.py`, `routes/auth.py`, `routes/training.py`, `requirements/studio.txt`, `core/training/worker.py`
- **Root configs (5 files)**: `pyproject.toml`, `README.md`, `.gitignore`, `.github/workflows/stale.yml` (modify/delete), `studio/setup.sh`
- **unsloth core (1 file)**: `unsloth_cli/commands/studio.py`

`unsloth/kernels/utils.py` is touched upstream (2 commits) but NOT in the conflict set — `matmul_lora` (line 1027) is untouched. The contract test at `tests/test_matmul_lora_contract.py` stays green through the rebase.

## Resolution patterns by category

### Pattern A: HT added an enum value, upstream also added one

**Example**: `TrainingMethod` in `studio/frontend/src/types/training.ts`.
- Upstream: added `"cpt"` (Continued Pretraining)
- HT: added `"prompt-baking"` (bakery integration)
- **Resolution**: keep both. Final shape `"qlora" | "lora" | "full" | "cpt" | "prompt-baking"`.
- **Side-effect**: every `Record<TrainingMethod, X>` in the codebase needs both keys. Hit `studio/frontend/src/features/training/lib/training-methods.ts` (`BACKEND_TRAINING_TYPE`, `TRAINING_METHOD_LABELS`) and `studio/frontend/src/features/export/constants.ts` (`METHOD_LABELS`) and `model-section.tsx` (`METHOD_DOTS`).

### Pattern B: HT added a feature toggle, upstream rewrote the surrounding code

**Example**: `UNSLOTH_DISABLE_AUTH` short-circuit in `studio/backend/routes/auth.py`.
- Upstream: added rate-limiting via `_bucket_key` / `_login_blocked`; rewrote docstrings.
- HT: added DISABLE_AUTH early-return in `auth_status` and `login`.
- **Resolution**: keep upstream's docstrings + new features, ADD HT's short-circuit BEFORE upstream's logic. Order matters — disabled-auth must bypass rate-limiting, not be subject to it.

### Pattern C: HT extended a function signature, upstream extended it differently

**Example**: `studio/frontend/src/lib/vram.ts`.
- Upstream: `estimateLoadingVram(params, method, modelId)` — added a 3rd arg.
- HT: `estimateLoadingVram(params, method)` + new `checkMultiGpuVramFit(est, perGpuGb, count)`.
- **Resolution**: take upstream's 3-arg call (the `modelId` arg is meaningful), keep HT's `checkMultiGpuVramFit` call (multi-GPU is the HT feature).

### Pattern D: HT helper function dead code, upstream added a real helper

**Example**: `studio/frontend/src/app/auth-guards.ts`.
- Upstream: added `authRedirect()` helper, used 4 times in the file.
- HT: added `checkPasswordChangeRequired()`, never used anywhere.
- **Resolution**: keep upstream's `authRedirect()`, drop HT's unused helper. Grep for usages before dropping anything.

### Pattern E: HT's minimal version was a strict subset of upstream's later expansion

**Example**: `studio/setup.sh` STUDIO_HOME override.
- HT: `STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth/studio}"`
- Upstream: full block with `UNSLOTH_STUDIO_HOME` + `STUDIO_HOME` alias, whitespace strip, tilde expansion, write-check validation.
- **Resolution**: take upstream's block, drop HT's. Functionally identical for the simple case; safer for edge cases.

### Pattern F: HT added a new field/interface, upstream added unrelated fields nearby

**Example**: `studio/frontend/src/features/chat/types/api.ts`.
- Upstream: added many new fields (`cancel_id`, `provider_id`, `enable_prompt_caching`, …, `fast_mode`).
- HT: added `after_commit_token` field + a whole new `LileResponseMeta` interface.
- **Resolution**: keep ALL upstream fields, append HT's field at the end of the interface, append HT's new interface after the existing one closes.

### Pattern G: HT-only files that upstream doesn't have

**Example**: `lile/**` (no longer in this repo post-relocation), `Dockerfile`, `HT-CHANGELOG.md`.
- **Resolution**: no conflict by definition. Git applies HT's commits cleanly.

### Pattern H: Files HT deleted, upstream modified

**Example**: `.github/workflows/stale.yml` (HT deleted upstream's stale-issue bot).
- **Resolution**: `git rm <file>` then `git rebase --continue`. Don't second-guess — HT's removal was intentional (recorded in HT-CHANGELOG 2026-03-18).

### Pattern I: Both sides modified the same i18n keys / locale files

**Example**: `studio/frontend/src/i18n/locales/{en,zh-CN}.ts`.
- If HT only ADDS keys (no key conflict): merge both sides' additions.
- If HT/upstream rename or replace the same key: take upstream's name + HT's value (or merge value semantically). Update calls in the calling .tsx files to match.

## Resolution order

Confirmed counts from the 2026-06-01 actual rebase (not just rehearsal):

| Commit | Conflict files | Notes |
|---|---|---|
| 1: 2777b86fe `fix(studio/data)` | 0 | Auto-merge |
| 2: 98ba737a3 `feat(studio): HT branding + fork-sync CI` | 2 | Patterns E, G — `studio/setup.sh`, `unsloth_cli/commands/studio.py` |
| 3: d0d027c9a `feat(studio): multi-GPU + Docker + UNSLOTH_DISABLE_AUTH` | 4 | Patterns B, C, D, H — auth.py, auth-guards.ts, vram.ts, stale.yml (rm) |
| 4: 27781af7a `feat(studio): prompt baking` | 7 | Pattern A everywhere — types/training.ts, config.ts, constants.ts, model-section, params-section, mappers, training-config-store. Side-effect: training-methods.ts + en.ts + zh-CN.ts also need additions |
| 5: 36c784b67 `feat(studio/ht): HT mascot` | 0 | Auto-merge |
| 6: 0cb41b63d `feat(lile): LiveLearn daemon + Studio integration` | 8 | Biggest commit. Pattern F + chat integration. Took `--theirs` for 5 chat files (commit 13 reshapes them anyway — confirmed byte-identical to origin/ht after full rebase) |
| 7: dc045d7a4 `docs(ht): README` | 1 | README.md — took `--theirs` (commit 12 reshapes it for lile-relocation) |
| 8-9: lile follow-ups + RLVR | 0 each | All `lile/**` paths; upstream never touched |
| 10: 4c04c7fd8 `chore: remove lile` | 1 | pyproject.toml — kept upstream's `[tool.pytest.ini_options]` for security testpath (this section's drift was already resolved in commit 6) |
| 11-12: docs + chore | 0 each | Auto-merge |
| 13: 1afa1407d `chore: gitignore lile leftovers` | 1 | `.gitignore` — took commit 13's broader `/lile/` + `/lile_data/` + `/.playwright-mcp/` ignores; the lile_data ignores from commit 6 are obsolete |
| 14-16: capsule mode + tests | 0 each | New files only |

Actual conflict commits: **7 out of 16** (commits 2, 3, 4, 6, 7, 10, 13). The lile-relocation commits (8-9, 11-12) cleanly delete/restore lile-only files. Total wall-clock for the rebase: ~25 min with these patterns memorized.

## Commit 6 specifically (lile introduction)

The biggest commit. 6 conflicts in chat-related Studio frontend files. **Important**: commit 13 (hybrid lile capsule, PR #55) re-writes most of what commit 6 added. So when resolving commit 6:

- For chat-runtime-store, chat-adapter, chat-settings-sheet, app-sidebar, thread.tsx: keep HT's lile-toggle code intact (Pattern F-style merges). Don't agonize over edge cases — commit 13 will reshape it shortly.
- For types/api.ts: merge cleanly; the `after_commit_token` field and `LileResponseMeta` interface survive into commit 13 unchanged.

## After each commit

```bash
git add <resolved-files>
git rebase --continue
```

Then in another shell:
```bash
cd /home/me/ht/forks/ht-unsloth && pytest tests/test_matmul_lora_contract.py -v
```
The contract test runs in <1 s and confirms the kernel didn't drift. Re-run after each commit that touches `unsloth/kernels/`.

## After completing the rebase

1. `pytest tests/test_matmul_lora_contract.py -v` — kernel contract.
2. `cd studio/backend && pytest tests/test_lile_route.py -v` — Studio-lile-optional contract.
3. Push branch: `git push -u origin chore/rebase-YYYY-MM-DD` (your own creds; the `HAI_GH_PAT` 403 only affects the bot's `origin main` push, not feature branches).
4. Open PR. Both CI workflows (`unsloth-kernel-contract`, `studio-tests`) should turn green.
5. After merge: tag `ht-YYYY-MM-DD`, push tag. Bump agi's `unsloth` pin in a follow-up PR there. Bump `matrix.lile_ref` in `.github/workflows/studio-tests.yml` to the same tag.

## Rollback

If something goes catastrophically wrong post-merge, the `pre-sync-YYYYMMDD-HHMM` tag from step 1 of "Before you start" is the anchor. `git reset --hard pre-sync-YYYYMMDD-HHMM` on `ht` + force-push if needed (requires push to `main`/`ht`, which needs `HAI_GH_PAT` or admin override).
