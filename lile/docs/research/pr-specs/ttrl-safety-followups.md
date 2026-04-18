# TTRL Safety Follow-Ups — PR L.1 / L.2 / L.3 Spec

**Context**: PR #34 landed vanilla TTRL (`lile/teach/ttrl_mv.py`, flag-off default, eval-gate deferred). It correctly implements the roadmap PR L claim but does NOT yet include the three safety mechanisms that `surveys/test-time-rl-2025.md` §1 flagged as required for production flag-on. These follow-ups need explicit PR scope before `cfg.ttrl_pseudo_reward=True` ships in any path.

---

## What PR #34 got right

- **Razin-safe by construction**: SFT on a model-generated target, not a preference margin. Can only shift mass toward the concrete winner string; cannot reweight to arbitrary unintended outputs.
- **Verifier-filtered prompts**: `select_verifier(prompt)` gates which prompts are eligible for TTRL. Prompts with no registered verifier are skipped entirely.
- **`min_count ≥ 2` floor** on plurality: k=4 with one vote each returns `None` rather than training on the first-extractable rollout. Matches the roadmap's "train on *agreement*" spirit.
- **Per-prompt cap** (`max_per_prompt=3`) + **seen-bump-on-failure**: bounded exposure, no loop-on-broken-submit.
- **Flag-off default** + **eval-gate explicitly deferred**: no surprise activation; gate is a known follow-up.

This is the correct shape for "vanilla TTRL as a v0." The gaps below are **not PR #34 bugs** — they are the survey's load-bearing additions that need to land before the flag flips on.

---

## Gap 1 — Verifier-correctness gate on the majority winner (PR L.1)

**Source**: `surveys/test-time-rl-2025.md` §1 — T3RL (arXiv:2603.02203) adds verifier-correctness shaping on top of TTRL; reports **+31.6% on AIME 2024 over vanilla TTRL**.

**What's missing**: PR #34 uses the verifier as a *filter* (`select_verifier(prompt)` — only fire on claimable prompts) and as an *equivalence-key generator* (majority-vote over `extract_answer` output for math, sandboxed stdout for code). It does NOT call the verifier's existing `verify(domain, prompt, candidate) -> bool | float | None` correctness method at all. The majority winner is SFT'd on even when a correctness verdict is available.

**Actual current verifier shapes** (verified by reading `lile/objectives/verifiers/_math.py` and `_code.py`):
- `_code.verify`: **already a real correctness check** when the prompt embeds an `Expected:` stanza — runs the candidate's code in the existing sandbox and returns `True` iff `stdout == expected`. For prompts without an `Expected:` stanza, returns `None`.
- `_math.verify`: currently returns `True` on extractability (`extract_answer(candidate) is not None`), **not** on correctness — no reference answer is consulted. Adequate for filter / equivalence use but not for the T3RL gate.

**Proposed scope**:

- **L.1a (~30 LOC, cheap)** — wire the existing `verify()` call into `TTRLScheduler._maybe_run_one` between the majority vote and `submit_train`. If `verify(domain, prompt, winner) is False`, skip the submit and bump a new `skipped_verifier_reject` stat. If `None`, current behaviour preserved (proceed with SFT) so unlabeled prompts still feed the loop — but log the rate. Immediate win on code prompts with `Expected:` stanzas.

- **L.1b (~60 LOC, deeper)** — upgrade `_math.verify` so that when the prompt embeds a reference (e.g. `#### 42`, `\boxed{42}`, `Expected: 42`), it returns real correctness (`extract_answer(candidate) == extract_answer(prompt_reference)`) rather than mere extractability. For prompts without an embedded reference, keep returning `None` (unlabeled regime — outside L.1's scope). Recognizing embedded references unlocks ~every labelled benchmark prompt (GSM8K, MATH-500) the daemon sees during eval.

- **L.1c (separate future PR, larger)** — add a grader / solver path for the true unlabeled-math regime: either a sympy symbolic path (cheap, covers arithmetic and polynomial), or a small LLM-judge verifier (expensive, general). Out of scope for L.1; tracked for when unlabeled math TTRL becomes the bottleneck.

**Eval gate contribution**: `skipped_verifier_reject` rate becomes a monitorable signal. Expected delta on the deferred GSM8K held-out gate: survey's T3RL delta is +31.6% AIME over vanilla TTRL — even a fraction is load-bearing, and L.1a+b covers every labelled prompt path without any new dependency.

---

## Gap 2 — Consensus-drift monitor (false-popular attractor) (PR L.2)

**Source**: `surveys/test-time-rl-2025.md` §1 — "false-popular mode collapse" is the first of the three documented TTRL failure modes. Survey recommended a "consensus-drift monitor so the daemon aborts an update when majority answers collapse to a low-entropy attractor."

**What's missing**: PR #34 tracks `labels_enqueued`, `skipped_no_majority`, `skipped_empty_stdout` — all per-event counters. There is no **rolling-window** signal over the *distribution* of majority answers. If the model enters a false-popular attractor (e.g., answer "42" wins majority on 8 of the last 10 prompts across unrelated questions), TTRL keeps reinforcing it with no detection.

**Proposed scope (~50 LOC)**:
- Ring buffer in `TTRLScheduler` holding the last N (default 50) `(prompt_domain, winner_key)` pairs.
- On each successful vote, compute Shannon entropy of winner-keys within-domain over the window.
- If entropy drops below a per-domain floor (tune on smoke — initial heuristic: `H < 0.3 * H_max` where `H_max = log2(window_size)`), abort submit for that event, log `skipped_drift_detected`, and surface via `/v1/state/stats`.
- Config knobs: `ttrl_drift_window`, `ttrl_drift_entropy_floor`. Default permissive; tighten post-smoke.

**Eval gate contribution**: rate of `skipped_drift_detected` becomes a circuit-breaker metric for the deferred GSM8K gate. If drift events fire frequently in a live session, that alone is reason to keep the flag off.

---

## Gap 3 — Adversarial-prompt guard (PR L.3, lower priority)

**Source**: `surveys/test-time-rl-2025.md` §1 — "jailbreak amplification: adversarial test-time data shifts weights in unsafe directions."

**What's missing**: PR #34 trusts the inference trajectory log as the prompt source. If a user submits an adversarial / jailbreak prompt that the verifier happens to claim (e.g., a math problem with a prompt-injection payload), the k=4 rollouts and the SFT winner may embed the adversarial framing. Razin-safety bounds the damage (SFT can't reweight preferences), but we'd still be concretely moving mass toward an adversarial response pattern.

**Proposed scope (~40 LOC)** (explicitly *after* L.1 + L.2 land; lowest priority):
- Simple heuristic gate at prompt-selection time: a configurable deny-list of substrings / regex patterns (jailbreak canaries, instruction-override tokens), a max prompt-length cap, optional embedding-distance check against a small adversarial-prompt exemplar set.
- Not a replacement for real safety review; just a low-floor tripwire.

---

## Recommended sequencing

1. **L.1 (verifier-correctness gate)** — highest leverage per survey (+31.6% AIME); cheap; unblocks the deferred GSM8K eval gate to actually show lift instead of washing.
2. **L.2 (consensus-drift monitor)** — required circuit-breaker for flag-on. Cheap; composable with L.1.
3. **Deferred GSM8K eval gate** — tighten from roadmap's "no >5pp regression on other tasks" to also require: `majority-wrong rate < X`, `drift-detection rate < Y`, `verifier-agreement rate > Z`. Numbers land empirically post-L.1+L.2 smoke.
4. **L.3 (adversarial guard)** — after the above. Nice-to-have; not blocking flag-on for internal evals, **is** blocking for any external-user-facing TTRL deployment.

`cfg.ttrl_pseudo_reward=True` should not flip in any live path until at least L.1 + L.2 + the upgraded eval gate are in place.

---

## Cross-references

- `surveys/test-time-rl-2025.md` §1 — TTRL failure modes and recommended fixes (T3RL, CREAM).
- `lile/teach/ttrl_mv.py` — PR #34 implementation; extension points are the stat buckets in `__init__` and the vote-to-submit path in `_maybe_run_one`.
- `lile/objectives/verifiers/` — existing verifier plugin point; L.1 extends the protocol rather than replacing it.
- Razin-safety framing (`lile/GLOSSARY.md`) — already correctly applied in PR #34; none of the follow-ups change the safety class.
