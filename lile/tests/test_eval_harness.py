"""Smoke tests for ``lile.teach.eval`` — stub path, no network, no GPU.

The harness itself is a thin CLI over ``lm-eval`` and ``evalplus``. The
value this test pins is the *stub* behavior: if those optional deps are
missing, ``_run_lm_eval`` / ``_run_evalplus`` must return a ``TaskResult``
with ``stub=True`` and a useful note, and ``_emit`` must serialize the
run to a stable JSON shape. This is what CI without ``--extra eval``
actually exercises; it also guards the public shape of the output file
that Studio/research tooling keys on.

Run:
    pytest -m eval lile/tests/test_eval_harness.py -xvs
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lile.teach import eval as eval_mod
from lile.teach.eval import (
    CODE_TASKS,
    LM_EVAL_TASKS,
    RunResult,
    TaskResult,
    _emit,
    _run_evalplus,
    _run_lm_eval,
    run,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.eval]


# ----------------------------------------------------------------- registry
def test_lm_eval_task_registry_is_stable() -> None:
    assert set(LM_EVAL_TASKS) == {"hellaswag", "arc_easy", "arc_challenge", "gsm8k"}
    for name, meta in LM_EVAL_TASKS.items():
        assert {"metric", "lm_eval_name"} <= meta.keys()


def test_code_task_registry_is_stable() -> None:
    assert set(CODE_TASKS) == {"humaneval_plus"}
    assert CODE_TASKS["humaneval_plus"]["metric"] == "pass@1"


# ----------------------------------------------------------------- stub path
def _force_missing(monkeypatch: pytest.MonkeyPatch, *module_names: str) -> None:
    """Make ``import <module_name>`` inside eval.py raise ImportError.

    Patches ``builtins.__import__`` so the two optional deps (lm-eval,
    evalplus) appear absent even when they are installed in the env.
    """
    import builtins
    real_import = builtins.__import__
    blocked = set(module_names)

    def _raise(name, *args, **kwargs):
        head = name.split(".", 1)[0]
        if head in blocked:
            raise ImportError(f"stubbed-out {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise)


def test_run_lm_eval_returns_stub_when_dep_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_missing(monkeypatch, "lm_eval")
    r = _run_lm_eval("hellaswag", "http://127.0.0.1:8768/v1", "fake-model",
                     limit=10, batch_size=4)
    assert isinstance(r, TaskResult)
    assert r.stub is True
    assert r.task == "hellaswag"
    assert r.metric == "acc_norm"
    assert r.n == 0
    assert "lm-eval" in r.raw.get("note", "")


def test_run_evalplus_returns_stub_when_dep_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_missing(monkeypatch, "evalplus")
    r = _run_evalplus("humaneval_plus", "http://127.0.0.1:8768/v1",
                      "fake-model", limit=10)
    assert r.stub is True
    assert r.task == "humaneval_plus"
    assert r.metric == "pass@1"
    assert "evalplus" in r.raw.get("note", "")


# ----------------------------------------------------------------- unknown task
def test_run_rejects_unknown_lm_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod, "_get_commit_cursor", lambda _endpoint: None)
    with pytest.raises(SystemExit):
        run("http://127.0.0.1:8768/v1", "fake-model",
            tasks=["not_a_task"], code_tasks=[], limit=1, batch_size=1)


def test_run_rejects_unknown_code_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod, "_get_commit_cursor", lambda _endpoint: None)
    with pytest.raises(SystemExit):
        run("http://127.0.0.1:8768/v1", "fake-model",
            tasks=[], code_tasks=["not_a_task"], limit=1, batch_size=1)


# ----------------------------------------------------------------- full stub run
def test_run_emits_stable_json_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_missing(monkeypatch, "lm_eval", "evalplus")
    monkeypatch.setattr(eval_mod, "_get_commit_cursor", lambda _endpoint: 42)

    result = run(
        endpoint="http://127.0.0.1:8768/v1",
        model="fake-model",
        tasks=["hellaswag", "gsm8k"],
        code_tasks=["humaneval_plus"],
        limit=10,
        batch_size=4,
    )
    assert isinstance(result, RunResult)
    assert result.commit_cursor_before == 42
    assert result.commit_cursor_after == 42
    assert len(result.tasks) == 3
    assert all(t.stub for t in result.tasks)

    out = tmp_path / "baseline.json"
    _emit(result, out)
    # Strict-JSON parse — `json.loads` rejects `NaN` unless parse_constant
    # is overridden. We emit null for stub values so downstream tooling
    # (Studio charts, research diffs) doesn't blow up on non-standard JSON.
    raw = out.read_text()
    assert "NaN" not in raw
    payload = json.loads(raw)
    assert {"run_id", "timestamp", "endpoint", "model",
            "commit_cursor_before", "commit_cursor_after", "tasks"} <= payload.keys()
    task_keys = {"task", "metric", "value", "n", "wall_clock_s", "stub", "raw"}
    for t in payload["tasks"]:
        assert task_keys <= t.keys()
        # Stub tasks emit value=None, not NaN.
        assert t["value"] is None


# ----------------------------------------------------------------- commit cursor probe
def test_get_commit_cursor_handles_unreachable_endpoint() -> None:
    # Unroutable port on loopback; probe must return None, not raise.
    cursor = eval_mod._get_commit_cursor("http://127.0.0.1:1/v1")
    assert cursor is None
