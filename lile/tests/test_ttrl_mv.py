"""TTRL majority-vote scheduler (PR L) — unit tests without GPU.

Pins the five load-bearing behaviors:

1. ``majority_vote`` picks plurality and breaks ties by first-occurrence.
2. ``majority_vote`` returns ``None`` when nothing is extractable.
3. ``_pick_prompt`` only yields inference records whose prompt a registered
   verifier claims, respects ``max_per_prompt`` cap, and gates on
   ``min_prompts``.
4. ``_maybe_run_one`` samples ``k_rollouts`` via ``controller.generate`` and
   enqueues an SFT spec tagged ``_ttrl_mv=True`` on the winner.
5. Idle-gate blocks action while the queue claims non-idle.

We stub Controller with a tiny async-friendly fake; no asyncio sleeps in
the core assertions — ``_maybe_run_one`` is called directly.

Run: pytest lile/tests/test_ttrl_mv.py -xvs
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from lile.teach.ttrl_mv import (
    TTRLPolicy,
    TTRLScheduler,
    _rollout_key,
    majority_vote,
)
from lile.trajectory import TrajectoryLog

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------- fakes
class _FakeQueue:
    def __init__(self, idle: bool = True) -> None:
        self.idle = idle

    def is_idle_for(self, seconds: float) -> bool:
        return self.idle


class _FakeController:
    """Minimal Controller stand-in for the TTRL scheduler."""

    def __init__(
        self,
        trajectory: TrajectoryLog,
        queue: _FakeQueue,
        rollouts: list[str] | None = None,
    ) -> None:
        self.trajectory = trajectory
        self.queue = queue
        self._rollouts = list(rollouts) if rollouts else []
        self.generate_calls: list[dict] = []
        self.submitted: list[dict] = []

    async def generate(self, messages, **kwargs):
        self.generate_calls.append({"messages": messages, **kwargs})
        # Cycle through the scripted rollouts so a single list can feed
        # k_rollouts calls.
        idx = (len(self.generate_calls) - 1) % max(1, len(self._rollouts))
        raw = self._rollouts[idx] if self._rollouts else ""
        return {"raw": raw, "response": raw}

    async def submit_train(self, spec):
        self.submitted.append(spec)
        return {"commit_token": len(self.submitted) - 1}


def _write_inference(
    log: TrajectoryLog, *, prompt: str, response: str = "x",
    ts: float | None = None,
) -> int:
    """Append an inference event via ``append_raw`` so tests control ``ts``."""
    payload = {
        "kind": "inference",
        "ts": ts if ts is not None else time.time(),
        "response_id": f"r_test_{id(prompt)}",
        "prompt": prompt,
        "response": response,
        "model_fingerprint": "deadbeef",
    }
    return log.append_raw(payload)


# ---------------------------------------------------------------- majority_vote
def test_majority_vote_clear_winner_math():
    rollouts = ["answer: 42", "the result is 42", "I get 17"]
    result = majority_vote(rollouts, domain="math")
    assert result is not None
    idx, key = result
    assert key == "42"
    # First occurrence of the winning key:
    assert idx == 0


def test_majority_vote_ties_break_on_first_occurrence():
    # Two 1-vote keys: "5" vs "7". Tie → first occurrence wins.
    rollouts = ["the answer is 5", "actually 7"]
    result = majority_vote(rollouts, domain="math")
    assert result is not None
    idx, key = result
    assert idx == 0 and key == "5"


def test_majority_vote_none_when_no_extraction():
    # Prose with no digits at all — math extractor returns None for each.
    rollouts = ["no numbers here", "still no numbers"]
    assert majority_vote(rollouts, domain="math") is None


def test_majority_vote_empty_list_returns_none():
    assert majority_vote([], domain="math") is None


def test_majority_vote_skips_unextractable_rollouts():
    # Mix: two rollouts with extractable "9", one with nothing.
    rollouts = ["just words", "answer: 9", "my guess is 9"]
    result = majority_vote(rollouts, domain="math")
    assert result is not None
    idx, key = result
    assert key == "9"
    # First rollout whose key == "9" is index 1.
    assert idx == 1


# ---------------------------------------------------------------- _rollout_key
def test_rollout_key_math_boxed_and_hash():
    assert _rollout_key("math", "\\boxed{12}") == "12"
    assert _rollout_key("math", "#### 7") == "7"
    assert _rollout_key("math", "prose without any numbers") is None


def test_rollout_key_unknown_domain_returns_none():
    assert _rollout_key("unknown", "anything") is None


def test_rollout_key_code_runs_sandbox():
    # Smoke: fenced python block whose stdout becomes the key.
    rollout = "```python\nprint('hi')\n```"
    assert _rollout_key("code", rollout) == "hi"


# ---------------------------------------------------------------- _pick_prompt
def test_pick_prompt_requires_min_prompts():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        _write_inference(log, prompt="How many apples if Jane has 3?")
        ctrl = _FakeController(log, _FakeQueue())
        sched = TTRLScheduler(ctrl, TTRLPolicy(min_prompts=3))
        assert sched._pick_prompt() is None


def test_pick_prompt_filters_to_verifier_claimed():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        # 2 math-claimed + 1 unclaimed free-text prompt.
        _write_inference(log, prompt="What is 2 plus 3?")
        _write_inference(log, prompt="How many coins if you have 4 and lose 1?")
        _write_inference(log, prompt="tell me a story about dragons")
        ctrl = _FakeController(log, _FakeQueue())
        sched = TTRLScheduler(ctrl, TTRLPolicy(min_prompts=2))
        choice = sched._pick_prompt()
        assert choice is not None
        offset, prompt, domain = choice
        assert domain == "math"
        # Most recent of the two math prompts → "How many coins..."
        assert "coins" in prompt


def test_pick_prompt_respects_per_offset_cap():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        # Two qualifying math prompts; cap the newer one.
        off1 = _write_inference(log, prompt="What is 2 plus 3?")
        off2 = _write_inference(log, prompt="How many coins if you have 4?")
        ctrl = _FakeController(log, _FakeQueue())
        sched = TTRLScheduler(ctrl, TTRLPolicy(min_prompts=1, max_per_prompt=1))
        # Simulate already-seen on the newer prompt.
        sched._seen[off2] = 1
        choice = sched._pick_prompt()
        assert choice is not None
        offset, _prompt, _domain = choice
        # With the newer one capped, we fall back to the older offset.
        assert offset == off1


# ---------------------------------------------------------------- _maybe_run_one
def test_maybe_run_one_samples_and_enqueues_majority():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        _write_inference(log, prompt="What is 2 plus 3? So how many?")
        _write_inference(log, prompt="How many coins if you have 4 and lose 1?")
        rollouts = [
            "the result is 3",   # key "3"
            "answer: 3",         # key "3"  ← winning majority
            "I think it is 5",   # key "5"
            "nope, 7",           # key "7"
        ]
        ctrl = _FakeController(log, _FakeQueue(), rollouts=rollouts)
        sched = TTRLScheduler(
            ctrl,
            TTRLPolicy(min_prompts=1, k_rollouts=4, max_per_prompt=5),
        )
        asyncio.run(sched._maybe_run_one())
        assert len(ctrl.submitted) == 1
        spec = ctrl.submitted[0]
        assert spec["objective"] == "sft"
        assert spec["_ttrl_mv"] is True
        assert "_ttrl_offset" in spec
        sample = spec["samples"][0]
        # Winner rollout was index 0 of the "3" keys (first occurrence).
        assert sample["response"] == "the result is 3"
        assert "coins" in sample["prompt"]
        assert sched.stats["labels_enqueued"] == 1
        assert sched.stats["rollouts_sampled"] == 4


def test_maybe_run_one_skips_when_no_majority_extractable():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        off = _write_inference(log, prompt="How many apples if Jane has 3?")
        # All rollouts have no extractable numeric answer.
        rollouts = ["no idea", "not sure", "pass", "hmm"]
        ctrl = _FakeController(log, _FakeQueue(), rollouts=rollouts)
        sched = TTRLScheduler(
            ctrl,
            TTRLPolicy(min_prompts=1, k_rollouts=4, max_per_prompt=2),
        )
        asyncio.run(sched._maybe_run_one())
        assert len(ctrl.submitted) == 0
        assert sched.stats["skipped_no_majority"] == 1
        # Burned a seen count so we don't immediately re-pick it forever.
        assert sched._seen[off] == 1


def test_maybe_run_one_skips_when_log_below_min_prompts():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        # One math prompt but min_prompts=3 → no pick.
        _write_inference(log, prompt="What is 2 plus 3?")
        ctrl = _FakeController(log, _FakeQueue(), rollouts=["answer: 5"])
        sched = TTRLScheduler(ctrl, TTRLPolicy(min_prompts=3, k_rollouts=1))
        asyncio.run(sched._maybe_run_one())
        assert len(ctrl.submitted) == 0
        assert sched.stats["skipped_empty"] == 1


# ---------------------------------------------------------------- lifecycle
def test_idle_gate_blocks_scheduler():
    with tempfile.TemporaryDirectory() as td:
        log = TrajectoryLog(Path(td) / "t.jsonl")
        # Enough prompts to clear min_prompts; queue says not-idle so nothing
        # should enqueue.
        for i in range(4):
            _write_inference(log, prompt=f"What is {i} plus 1?")
        ctrl = _FakeController(log, _FakeQueue(idle=False), rollouts=["answer: 1"])
        sched = TTRLScheduler(
            ctrl,
            TTRLPolicy(min_prompts=1, k_rollouts=1, poll_interval_s=0.01,
                       idle_threshold_s=10),
        )

        async def run():
            await sched.start()
            await asyncio.sleep(0.05)
            await sched.stop()

        asyncio.run(run())
        assert len(ctrl.submitted) == 0
        assert sched.stats["idle_checks"] > 0


def test_from_config_reads_ttrl_fields():
    class _Cfg:
        ttrl_k_rollouts = 7
        ttrl_idle_threshold_s = 45.0
        ttrl_poll_interval_s = 3.5
        ttrl_max_per_prompt = 11
        ttrl_min_prompts = 2
        ttrl_temperature = 0.6
        ttrl_top_p = 0.9

    p = TTRLPolicy.from_config(_Cfg())
    assert p.k_rollouts == 7
    assert p.idle_threshold_s == 45.0
    assert p.poll_interval_s == 3.5
    assert p.max_per_prompt == 11
    assert p.min_prompts == 2
    assert p.sampling_temperature == 0.6
    assert p.sampling_top_p == 0.9


def test_from_config_defaults_when_fields_missing():
    class _Bare:
        pass

    p = TTRLPolicy.from_config(_Bare())
    # Defaults match the dataclass.
    assert p.k_rollouts == 4
    assert p.idle_threshold_s == 30.0
    assert p.sampling_temperature == 0.8
