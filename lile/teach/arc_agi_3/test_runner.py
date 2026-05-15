"""ARC-AGI-3 smoke tests — torchless / no daemon.

These tests pin the smoke deliverable's shape:

- loader returns the 3 frozen tasks with valid schema
- prompt builder embeds train pairs and the test input
- runner happy-path: a JSON-fenced answer round-trips to ``correct=True``
- runner sad-path: garbage response → ``correct=False`` with parse_error
- verifier claims ARC-shaped prompts and ignores math/code

The file lives next to the package (not under ``lile/tests/``) because
the agentic follow-up will grow more ARC-specific tests; it's invoked
explicitly via ``pytest lile/teach/arc_agi_3/test_runner.py``. The repo
``testpaths`` only auto-discovers ``lile/tests`` so this won't be picked
up by a bare ``pytest`` invocation — that's intentional, the smoke path
is run by hand and by Track E's eval CLI.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from lile.teach.arc_agi_3.loader import load_tasks
from lile.teach.arc_agi_3.prompts import build_prompt
from lile.teach.arc_agi_3.runner import run_task


pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------- loader / prompt


def test_load_tasks_returns_3_pinned() -> None:
    tasks = load_tasks()
    assert len(tasks) == 3
    seen_ids: set[str] = set()
    for task in tasks:
        assert {"id", "train_pairs", "test_input", "test_output"} <= task.keys()
        assert isinstance(task["id"], str) and task["id"]
        assert task["id"] not in seen_ids, "duplicate task id"
        seen_ids.add(task["id"])
        assert len(task["train_pairs"]) >= 1
        for pair in task["train_pairs"]:
            assert "input" in pair and "output" in pair
            assert all(0 <= v <= 9 for row in pair["input"] for v in row)
            assert all(0 <= v <= 9 for row in pair["output"] for v in row)
        assert all(0 <= v <= 9 for row in task["test_input"] for v in row)
        assert all(0 <= v <= 9 for row in task["test_output"] for v in row)


def test_prompt_includes_grids() -> None:
    tasks = load_tasks()
    task = tasks[0]
    prompt = build_prompt(task)
    # All training pairs must surface in the prompt.
    for pair in task["train_pairs"]:
        for row in pair["input"]:
            assert str(row) in prompt
        for row in pair["output"]:
            assert str(row) in prompt
    # And the test input.
    for row in task["test_input"]:
        assert str(row) in prompt
    # The header is what verifier.claims() pattern-matches on.
    assert "ARC-AGI-3 task" in prompt


# --------------------------------------------------------------- runner


class _StubController:
    """Minimal duck-type for ``controller.generate``."""

    def __init__(self, response_text: str) -> None:
        self._text = response_text
        self.calls: list[dict[str, object]] = []

    async def generate(self, *, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"response": self._text, "reasoning_content": None, "raw": self._text}


def test_run_task_correct_path() -> None:
    tasks = load_tasks()
    task = tasks[0]
    answer = json.dumps(task["test_output"])
    fenced = f"Here is my answer.\n```json\n{answer}\n```\n"
    ctrl = _StubController(fenced)
    result = asyncio.run(run_task(ctrl, task))
    assert result["task_id"] == task["id"]
    assert result["correct"] is True
    assert result["parse_error"] is None
    assert result["predicted"] == task["test_output"]
    # And we honored the deterministic-decoding contract from the runner.
    assert ctrl.calls and ctrl.calls[0]["kwargs"].get("temperature") == 0.0


def test_run_task_parse_error() -> None:
    tasks = load_tasks()
    task = tasks[0]
    ctrl = _StubController("I refuse to play. There is no grid.")
    result = asyncio.run(run_task(ctrl, task))
    assert result["correct"] is False
    assert result["parse_error"] is not None
    assert result["predicted"] is None


# ------------------------------------------------------------- verifier


def test_verifier_claims_arc_prompt() -> None:
    # Importing the verifier triggers @register("arc"); we also want to
    # exercise the registry's ``select`` to confirm routing works.
    from lile.objectives.verifiers import select
    from lile.teach.arc_agi_3 import verifier as arc_verifier

    arc_prompt = build_prompt(load_tasks()[0])
    assert arc_verifier.claims(arc_prompt) is True

    math_prompt = "What is 2 + 2? Answer with a single number."
    code_prompt = (
        "Write a python program. Expected: 6\n"
        "```python\nprint(1 + 2 + 3)\n```"
    )
    assert arc_verifier.claims(math_prompt) is False
    assert arc_verifier.claims(code_prompt) is False

    # And ``select`` should route the ARC prompt to "arc" (insertion order
    # in the registry is math, code, arc — but only arc claims this prompt).
    assert select(arc_prompt) == "arc"
