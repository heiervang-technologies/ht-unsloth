# Studio ↔ lile integration contract

HT Studio (this fork) integrates with [lile](https://github.com/heiervang-technologies/agi) — the LiveLearn live-training daemon — but does not depend on it. The integration is **opt-in** and **runtime-detected**, not install-time-required.

## The contract, in one paragraph

Studio backend MUST import and serve cleanly in a Python interpreter where the `lile` package is absent. Lile's presence is probed at request time via `routes.lile.lile_available()`. When lile is absent and no external daemon URL is configured, the `/api/lile/capsule/*` endpoints return `{running: false, mode: "unconfigured"}` instead of raising. The frontend renders this as an "offline" state with a hint to set `LILE_DAEMON_URL` or install the package.

## The four modes

`/api/lile/capsule/status` reports one of four modes via the `mode` field:

| Mode | Triggered by | Who owns the daemon's lifecycle |
|---|---|---|
| `external` | `LILE_DAEMON_URL` (or `LILE_HOST` + `LILE_PORT`) set, daemon reachable at `/health` | The process that launched the daemon. Studio's Stop button is disabled. |
| `spawned` | `lile_available()` is `True`, Studio launched the daemon as a subprocess | Studio. Stop sends SIGTERM, escalates to SIGKILL after 15 s. |
| `external-unreachable` | URL configured, but `/health` doesn't respond | Whoever launched it (and it's broken). |
| `unconfigured` | No URL set, lile package not installed | Nobody. Studio shows install hint. |

The frontend types this as `CapsuleMode` in `studio/frontend/src/features/lile/api/types.ts` — keep that union in sync with backend additions.

## How spawn-or-connect resolves

`capsule/start` is the only route with a branching policy. It tries:

1. **External probe.** If `LILE_DAEMON_URL` (or legacy host+port) is set, GET `/health` with a 500 ms timeout. If 200, return `mode=external` and we're done.
2. **Spawned check.** If we previously spawned a subprocess and `_is_alive(pid)` is still true, reuse it.
3. **Spawn locally.** If `lile_available()` returns True, exec `python -m lile.console.launch` with `LILE_HOST=127.0.0.1`, port from `LILE_PORT` (default 8768). Poll `/health` for up to 60 s.
4. **Give up.** Return `mode=external-unreachable` with the install hint.

This ordering is deliberate: an externally-managed daemon always wins (someone is operating it; we shouldn't shadow it with a subprocess we also have to manage).

## Adding new lile-only code paths in Studio

If you add a route, route helper, or background task that depends on lile:

- Do NOT add `import lile` at module top.
- Call `routes.lile.lile_available()` at request time and degrade gracefully when it returns False.
- Add a test cell to the `studio-tests` workflow if your code path has lile-installed-only behavior that needs CI coverage.
- Don't add a second try/except-import probe somewhere else; extend `lile_available()` (or the route that needs it) instead — the contract is "one canonical check, in one place."

## CI

`.github/workflows/studio-tests.yml` runs the lile route tests in a matrix:

- **`lile-absent`**: only Studio deps installed. Asserts `lile` package is genuinely not importable, then runs `tests/test_lile_route.py`. This is the contract: tests must pass with lile uninstalled.
- **`lile-installed`**: pip-installs `lile @ git+https://github.com/heiervang-technologies/agi@<ht-YYYY-MM-DD>` (the cross-repo pin tag). Same test suite must pass.

When you bump the cross-repo pin in `HT-CHANGELOG.md`, also bump `matrix.lile_ref` in the workflow.

## Cross-repo pin coupling

`heiervang-technologies/agi`'s `pyproject.toml` pins `unsloth @ git+https://github.com/heiervang-technologies/ht-unsloth@ht-YYYY-MM-DD`. Studio's CI pins lile at the same `ht-YYYY-MM-DD` tag. **One tag, both directions** — that's the simplest discipline for keeping the two repos in lockstep.

The flow per upstream-sync:

1. ht-unsloth rebases onto upstream/main. Tag `ht-YYYY-MM-DD`.
2. agi bumps its `unsloth` pin to the new tag.
3. ht-unsloth's `.github/workflows/studio-tests.yml` bumps `matrix.lile_ref` to the same new tag.
4. CI confirms both repos still play nicely.

## Why this exists

Earlier iterations of HT-Studio imported `lile` at module top. That broke every Studio install that didn't also want live-learning. The contract documented here removes that coupling, and the CI matrix makes regressions noisy.
