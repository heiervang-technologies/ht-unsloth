"""Smoke tests for ``lile.teach.eval_arc_smoke`` — no daemon, no GPU.

Pin the public shape of:

- :func:`run_arc_eval` — async; returns a structured summary with the
  per-task ``{task_id, correct, parse_error}`` fields and an aggregate
  ``correct/total/pass_rate``.
- The integration with :mod:`lile.teach.eval` — when ``arc_agi_3`` is
  passed, dispatch routes through ``_run_custom`` and resolves the
  ``custom_runner`` callable from the registry instead of lm-eval.

The HTTP plane is mocked end-to-end. Two layers of mocking:

1. A pure ``FakeClient`` with an async ``.generate`` is injected into
   :func:`run_arc_eval` via the ``client=`` kwarg — covers the runner
   contract without touching the network.
2. For the eval-dispatch test, ``LM_EVAL_TASKS["arc_agi_3"]`` is
   monkey-patched to point at a dummy runner; we then assert ``run()``
   invokes that dummy and produces a ``TaskResult`` with the expected
   metric and pass_rate value.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from lile.teach import eval as eval_mod
from lile.teach import eval_arc_smoke
from lile.teach.arc_agi_3.loader import load_tasks


pytestmark = [pytest.mark.cpu_only, pytest.mark.eval]


# ----------------------------------------------------------------- fakes
class _FakeClient:
    """Async generate-shaped stand-in. Replays one queued response per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, *, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self._responses:
            text = self._responses.pop(0)
        else:
            text = ""
        return {"response": text, "reasoning_content": None, "raw": text}


# ---------------------------------------------------- run_arc_eval shape
def test_run_arc_eval_returns_per_task_shape() -> None:
    tasks = load_tasks()
    # Replay the gold answer for every task → all correct.
    responses = [
        f"Reasoning... Final answer:\n```json\n{json.dumps(t['test_output'])}\n```"
        for t in tasks
    ]
    client = _FakeClient(responses)

    summary = asyncio.run(eval_arc_smoke.run_arc_eval(n=len(tasks), client=client))

    assert summary["n"] == len(tasks)
    assert summary["total"] == len(tasks)
    assert summary["correct"] == len(tasks)
    assert summary["pass_rate"] == 1.0
    # Per-task entries carry exactly the documented fields.
    expected_keys = {"task_id", "correct", "parse_error"}
    assert all(expected_keys <= set(t) for t in summary["tasks"])
    assert {t["task_id"] for t in summary["tasks"]} == {t["id"] for t in tasks}
    assert all(t["correct"] is True for t in summary["tasks"])
    assert all(t["parse_error"] is None for t in summary["tasks"])
    # And the client really was invoked once per task.
    assert len(client.calls) == len(tasks)


def test_run_arc_eval_counts_failures() -> None:
    tasks = load_tasks()
    # First task: correct. Rest: garbage that won't parse.
    responses = [
        f"```json\n{json.dumps(tasks[0]['test_output'])}\n```",
        *(["nope, no grid here"] * (len(tasks) - 1)),
    ]
    client = _FakeClient(responses)

    summary = asyncio.run(eval_arc_smoke.run_arc_eval(n=len(tasks), client=client))

    assert summary["correct"] == 1
    assert summary["total"] == len(tasks)
    assert summary["pass_rate"] == pytest.approx(1.0 / len(tasks))
    failing = [t for t in summary["tasks"] if not t["correct"]]
    assert len(failing) == len(tasks) - 1
    assert all(t["parse_error"] for t in failing)


def test_run_arc_eval_caps_n_at_loader_count() -> None:
    tasks = load_tasks()
    client = _FakeClient(["``` json\n[[0]]\n```"] * 100)
    # Asking for more tasks than exist must cap at the loader's count.
    summary = asyncio.run(eval_arc_smoke.run_arc_eval(n=999, client=client))
    assert summary["total"] == len(tasks)


# ------------------------------------------------- eval.py integration
def test_eval_dispatch_invokes_custom_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run()`` with ``arc_agi_3`` must route to the registry's custom_runner.

    Patches the registry entry to point at a dummy runner so the test is
    independent of the real ARC adapter. Asserts the dummy is invoked
    with the dispatched ``daemon_url`` / ``n`` and that the resulting
    ``TaskResult`` carries the registry's metric and the runner's
    ``pass_rate`` value.
    """
    invocations: list[dict[str, Any]] = []

    async def _dummy(*, daemon_url: str, n: int) -> dict[str, Any]:
        invocations.append({"daemon_url": daemon_url, "n": n})
        return {
            "daemon": daemon_url, "n": n,
            "correct": 1, "total": 2, "pass_rate": 0.5,
            "tasks": [
                {"task_id": "a", "correct": True, "parse_error": None},
                {"task_id": "b", "correct": False, "parse_error": "x"},
            ],
        }

    # Replace the registry pointer with a ``module:attr`` that resolves
    # to ``_dummy``. We monkey-patch the resolver instead of stuffing the
    # callable into ``sys.modules`` — same effect, less surface.
    monkeypatch.setattr(eval_mod, "_resolve_custom_runner", lambda spec: _dummy)
    monkeypatch.setattr(eval_mod, "_get_commit_cursor", lambda _endpoint: 7)

    result = eval_mod.run(
        endpoint="http://127.0.0.1:8768/v1",
        model="fake-model",
        tasks=["arc_agi_3"],
        code_tasks=[],
        limit=2,
        batch_size=1,
    )

    # Custom runner ran exactly once with the daemon URL stripped of /v1.
    assert invocations == [{"daemon_url": "http://127.0.0.1:8768", "n": 2}]
    assert len(result.tasks) == 1
    tr = result.tasks[0]
    assert tr.task == "arc_agi_3"
    assert tr.metric == "acc"
    assert tr.value == pytest.approx(0.5)
    assert tr.n == 2
    assert tr.stub is False
    # The raw payload from the runner is forwarded for downstream tooling.
    assert tr.raw["correct"] == 1 and tr.raw["total"] == 2


def test_eval_dispatch_does_not_call_lm_eval_for_custom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``custom_runner`` is set, ``_run_lm_eval`` must NOT be called.

    Guards the dispatch branch: a regression where lm-eval ran alongside
    the custom runner would double-bill / double-evaluate.
    """
    called = {"lm_eval": 0}

    def _boom_lm_eval(*args, **kwargs):  # noqa: ANN001, ARG001
        called["lm_eval"] += 1
        raise AssertionError("lm-eval path must not run for custom_runner tasks")

    async def _dummy(*, daemon_url: str, n: int) -> dict[str, Any]:
        return {"correct": 0, "total": n, "pass_rate": 0.0, "tasks": []}

    monkeypatch.setattr(eval_mod, "_run_lm_eval", _boom_lm_eval)
    monkeypatch.setattr(eval_mod, "_resolve_custom_runner", lambda spec: _dummy)
    monkeypatch.setattr(eval_mod, "_get_commit_cursor", lambda _endpoint: None)

    eval_mod.run(
        endpoint="http://127.0.0.1:8768/v1",
        model="fake-model",
        tasks=["arc_agi_3"],
        code_tasks=[],
        limit=3,
        batch_size=1,
    )
    assert called["lm_eval"] == 0


def test_arc_agi_3_registered_with_custom_runner_spec() -> None:
    """The registry entry itself must point at the eval_arc_smoke module."""
    meta = eval_mod.LM_EVAL_TASKS["arc_agi_3"]
    assert meta["metric"] == "acc"
    assert meta["custom_runner"] == "lile.teach.eval_arc_smoke:run_arc_eval"
    assert "lm_eval_name" not in meta
