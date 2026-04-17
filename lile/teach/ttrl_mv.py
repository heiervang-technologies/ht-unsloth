"""TTRL-style majority-vote pseudo-reward — roadmap PR L.

When the compute queue is idle and a recent inference prompt is claimed by a
registered verifier, sample ``k`` fresh rollouts from the live model,
majority-vote over the verifier-extracted answers, and enqueue an SFT step
on the rollout that matches the majority answer. Turns unlabeled prompts
into a steady low-priority training signal.

The roadmap (``production-implementation-roadmap.md#PR-L``) nominated
``lile/objectives/ttrl_mv.py`` as the module path, but architecturally this
is a *scheduler* — the rollout-sampling step must run in inference mode,
outside the queue worker — so it lives alongside :class:`IdleReplayScheduler`
in the teach/ layer. The training signal itself still flows through the
existing ``sft`` objective; TTRL is a pseudo-label generator, not a new loss.

Flag-gated (``cfg.ttrl_pseudo_reward=False``) and **eval-gate deferred**.
The roadmap's gate — "on GSM8K held-out, TTRL must not regress on any other
harness task by >5pp; kill if it does" — requires the ``lile[eval]``
harness to be CI-promoted first (not today; see the follow-up
harness-promotion PR). Until then, this module ships behind the flag with
smoke coverage only.

Razin-safety: we train SFT on a model-generated target string, not a
preference margin. SFT can only shift mass toward the concrete target;
it cannot re-weight mass to arbitrary unintended outputs (the Razin-safe
bound from ``lile/GLOSSARY.md``). The verifier filter ensures we only
TTRL prompts whose answer is *extractable* — correctness of that answer
is a separate concern that the deferred eval gate is designed to detect.
"""
from __future__ import annotations

import asyncio
import collections
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..objectives.verifiers import select as select_verifier
from ..objectives.verifiers._code import _run_sandboxed, extract_code
from ..objectives.verifiers._math import extract_answer as extract_math_answer

if TYPE_CHECKING:
    from ..controller import Controller

log = logging.getLogger(__name__)


@dataclass
class TTRLPolicy:
    k_rollouts: int = 4
    idle_threshold_s: float = 30.0
    poll_interval_s: float = 2.0
    max_per_prompt: int = 3        # lifetime cap per inference offset
    min_prompts: int = 3           # don't fire until the log has N candidates
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95

    @classmethod
    def from_config(cls, cfg: Any) -> "TTRLPolicy":
        return cls(
            k_rollouts=getattr(cfg, "ttrl_k_rollouts", 4),
            idle_threshold_s=getattr(cfg, "ttrl_idle_threshold_s", 30.0),
            poll_interval_s=getattr(cfg, "ttrl_poll_interval_s", 2.0),
            max_per_prompt=getattr(cfg, "ttrl_max_per_prompt", 3),
            min_prompts=getattr(cfg, "ttrl_min_prompts", 3),
            sampling_temperature=getattr(cfg, "ttrl_temperature", 0.8),
            sampling_top_p=getattr(cfg, "ttrl_top_p", 0.95),
        )


def _rollout_key(domain: str, rollout: str) -> str | None:
    """Normalize a rollout into its domain-specific equivalence key.

    ``math`` uses :func:`verifiers._math.extract_answer` (``####``, boxed,
    answer-is, last-number fallback). ``code`` runs the fenced snippet in
    the verifier sandbox and keys on stdout — two rollouts that print the
    same thing are equivalent regardless of source.
    """
    if domain == "math":
        return extract_math_answer(rollout)
    if domain == "code":
        code = extract_code(rollout)
        if code is None:
            return None
        try:
            return _run_sandboxed(code)
        except Exception:
            return None
    return None


def majority_vote(
    rollouts: list[str], domain: str,
) -> tuple[int, str] | None:
    """Return ``(winning_rollout_index, equivalence_key)`` or ``None``.

    Plurality wins — a strict majority is not required. Ties break on first
    occurrence so the result is deterministic given the input order.
    ``None`` when every rollout's key was None (nothing extractable).
    """
    keys = [_rollout_key(domain, r) for r in rollouts]
    counts = collections.Counter(k for k in keys if k is not None)
    if not counts:
        return None
    winner, _ = counts.most_common(1)[0]
    idx = next(i for i, k in enumerate(keys) if k == winner)
    return idx, winner


class TTRLScheduler:
    """Async scheduler that turns idle-queue prompts into SFT pseudo-labels."""

    def __init__(self, controller: "Controller", policy: TTRLPolicy) -> None:
        self.controller = controller
        self.policy = policy
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._seen: dict[int, int] = {}  # inference-offset -> ttrl count
        self.stats = {
            "idle_checks": 0,
            "rollouts_sampled": 0,
            "labels_enqueued": 0,
            "skipped_no_majority": 0,
            "skipped_empty": 0,
            "last_offset": -1,
        }

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("TTRL scheduler already started")
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="ttrl-mv")
        log.info("TTRL scheduler started (policy=%s)", self.policy)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(
                    self._task, timeout=self.policy.poll_interval_s * 2 + 1,
                )
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    # ------------------------------------------------------------------ main loop
    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.policy.poll_interval_s,
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                self.stats["idle_checks"] += 1
                if not self.controller.queue.is_idle_for(self.policy.idle_threshold_s):
                    continue
                await self._maybe_run_one()
        except asyncio.CancelledError:
            log.info("TTRL scheduler cancelled")
            raise
        except Exception:
            log.exception("TTRL scheduler crashed")
            raise

    # ------------------------------------------------------------------ core step
    async def _maybe_run_one(self) -> None:
        choice = self._pick_prompt()
        if choice is None:
            self.stats["skipped_empty"] += 1
            return
        offset, prompt, domain = choice
        rollouts = await self._sample_rollouts(prompt)
        self.stats["rollouts_sampled"] += len(rollouts)
        vote = majority_vote(rollouts, domain)
        if vote is None:
            self._seen[offset] = self._seen.get(offset, 0) + 1
            self.stats["skipped_no_majority"] += 1
            return
        idx, _key = vote
        winner = rollouts[idx]
        spec = {
            "objective": "sft",
            "samples": [{"prompt": prompt, "response": winner}],
            "_ttrl_mv": True,
            "_ttrl_offset": offset,
        }
        try:
            await self.controller.submit_train(spec)
            self._seen[offset] = self._seen.get(offset, 0) + 1
            self.stats["labels_enqueued"] += 1
            self.stats["last_offset"] = offset
        except Exception:
            log.exception("TTRL submit_train failed (offset=%d)", offset)

    # ------------------------------------------------------------------ pieces
    def _pick_prompt(self) -> tuple[int, str, str] | None:
        """Most-recent verifier-claimed inference prompt under the per-offset cap.

        Returns ``(offset, prompt, domain)`` or ``None`` when the log has
        fewer than ``policy.min_prompts`` qualifying entries.
        """
        candidates: list[tuple[int, str, str]] = []
        cap = self.policy.max_per_prompt
        for offset, rec in self.controller.trajectory.iter_with_offsets(
            kinds={"inference"},
        ):
            if self._seen.get(offset, 0) >= cap:
                continue
            prompt = rec.get("prompt") or ""
            if not prompt:
                continue
            domain = select_verifier(prompt)
            if domain is None:
                continue
            candidates.append((offset, prompt, domain))
        if len(candidates) < self.policy.min_prompts:
            return None
        return candidates[-1]

    async def _sample_rollouts(self, prompt: str) -> list[str]:
        """``k`` sequential ``controller.generate`` calls at sampling temperature."""
        messages = [{"role": "user", "content": prompt}]
        rollouts: list[str] = []
        for _ in range(self.policy.k_rollouts):
            try:
                result = await self.controller.generate(
                    messages,
                    temperature=self.policy.sampling_temperature,
                    top_p=self.policy.sampling_top_p,
                )
            except Exception:
                log.exception("TTRL rollout failed (prompt head=%r)", prompt[:60])
                continue
            rollouts.append(result.get("raw") or result.get("response") or "")
        return rollouts
