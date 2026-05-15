"""Unit tests for ``lile.teach.teacher_oss120b.judge``.

Stdlib + pytest only. The HTTP layer is mocked by monkeypatching
``urllib.request.urlopen`` so no network is touched. Style mirrors
``lile/tests/test_eval_harness.py`` — torchless and clean.

Run:
    /home/me/ht/forks/ht-unsloth/.venv/bin/python -m pytest \\
        lile/teach/teacher_oss120b_test.py -q
"""
from __future__ import annotations

import io
import json
from typing import Any

import pytest

from lile.teach import teacher_oss120b
from lile.teach.teacher_oss120b import JudgeResult, judge


# ---------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Clear the LRU cache before every test so cases don't bleed."""
    teacher_oss120b._clear_cache_for_tests()
    yield
    teacher_oss120b._clear_cache_for_tests()


def _frozen_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a judge JSON object in an OpenRouter chat-completions envelope."""
    return {
        "id": "gen-test-0001",
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload),
                },
                "finish_reason": "stop",
            }
        ],
    }


class _FakeResponse:
    """Minimal context-manager around bytes that quacks like ``urlopen``'s
    return value (only ``read()`` and the context-manager protocol are used
    by ``_post_openrouter``)."""

    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _install_fake_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    counter: list[int] | None = None,
) -> None:
    """Replace ``urllib.request.urlopen`` (as imported into the teacher
    module) so each call returns a fresh ``_FakeResponse`` of the envelope.

    If ``counter`` is provided, append 1 per call so tests can assert the
    cache short-circuits subsequent invocations.
    """
    envelope = _frozen_envelope(payload)
    body = json.dumps(envelope).encode("utf-8")

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        if counter is not None:
            counter.append(1)
        return _FakeResponse(body)

    monkeypatch.setattr(
        teacher_oss120b.urllib.request,
        "urlopen",
        _fake_urlopen,
    )


# ---------------------------------------------------------------- happy path
def test_judge_returns_expected_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    payload = {
        "grades": ["correct", "wrong", "ambiguous"],
        "critiques": [None, "Off by one; should add 1.", None],
        "counterfactuals": [None, "390", None],
        "demonstration": "17 * 23 = 391",
    }
    _install_fake_urlopen(monkeypatch, payload)

    result = judge(
        prompt="What is 17 * 23?",
        rollouts=["391", "390", "I'm not sure"],
        domain="math",
    )

    assert isinstance(result, dict)
    # TypedDict at runtime is a dict; check the four keys explicitly.
    assert set(result.keys()) == {
        "grades",
        "critiques",
        "counterfactuals",
        "demonstration",
    }
    assert result["grades"] == ["correct", "wrong", "ambiguous"]
    assert result["critiques"] == [None, "Off by one; should add 1.", None]
    assert result["counterfactuals"] == [None, "390", None]
    assert result["demonstration"] == "17 * 23 = 391"


def test_judge_accepts_single_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    payload = {
        "grades": ["correct"],
        "critiques": [None],
        "counterfactuals": [None],
        "demonstration": "42",
    }
    _install_fake_urlopen(monkeypatch, payload)
    result = judge(prompt="6 * 7?", rollouts=["42"], domain="math")
    assert result["grades"] == ["correct"]
    assert result["demonstration"] == "42"


def test_judge_normalizes_empty_strings_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some upstreams emit ``""`` instead of ``null`` for the no-critique
    case. The parser should normalize to ``None`` so callers can branch
    on truthiness without surprises."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    payload = {
        "grades": ["correct", "wrong"],
        "critiques": ["", "off by one"],
        "counterfactuals": ["   ", "390"],
        "demonstration": "391",
    }
    _install_fake_urlopen(monkeypatch, payload)
    result = judge(prompt="17*23?", rollouts=["391", "390"], domain="math")
    assert result["critiques"] == [None, "off by one"]
    assert result["counterfactuals"] == [None, "390"]


# ---------------------------------------------------------------- missing key
def test_missing_api_key_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # urlopen must never get called — but install a sentinel just so the
    # error message is helpful if it does.
    def _no_call(*a, **k):  # noqa: ARG001
        raise AssertionError("urlopen should not be called when API key is missing")

    monkeypatch.setattr(teacher_oss120b.urllib.request, "urlopen", _no_call)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        judge(prompt="x", rollouts=["y"], domain="math")


# ---------------------------------------------------------------- cache
def test_cache_short_circuits_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    payload = {
        "grades": ["correct"],
        "critiques": [None],
        "counterfactuals": [None],
        "demonstration": "42",
    }
    counter: list[int] = []
    _install_fake_urlopen(monkeypatch, payload, counter=counter)

    r1 = judge(prompt="6 * 7?", rollouts=["42"], domain="math")
    r2 = judge(prompt="6 * 7?", rollouts=["42"], domain="math")

    assert r1 == r2
    assert len(counter) == 1, (
        f"expected exactly 1 HTTP call across two identical judge() invocations, "
        f"got {len(counter)} — LRU cache is not engaging"
    )

    # A different rollout list should bypass the cache.
    r3 = judge(prompt="6 * 7?", rollouts=["41"], domain="math")
    assert r3["grades"] == ["correct"]
    assert len(counter) == 2


# ---------------------------------------------------------------- bad payload
def test_judge_rejects_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    payload = {
        # 2 grades but 3 rollouts → rejected.
        "grades": ["correct", "wrong"],
        "critiques": [None, "x"],
        "counterfactuals": [None, "y"],
        "demonstration": "42",
    }
    _install_fake_urlopen(monkeypatch, payload)
    with pytest.raises(RuntimeError, match="lengths"):
        judge(prompt="?", rollouts=["a", "b", "c"], domain="math")


def test_judge_rejects_non_json_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    # Envelope is valid JSON; the inner message content is not.
    envelope = {
        "choices": [
            {"message": {"content": "this is not json {{ broken"}}
        ]
    }
    body = json.dumps(envelope).encode("utf-8")

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResponse(body)

    monkeypatch.setattr(
        teacher_oss120b.urllib.request, "urlopen", _fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="strict JSON"):
        judge(prompt="?", rollouts=["a"], domain="math")


def test_judge_rejects_invalid_grade_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    payload = {
        "grades": ["correct", "perfectish"],   # second value is illegal
        "critiques": [None, None],
        "counterfactuals": [None, None],
        "demonstration": "42",
    }
    _install_fake_urlopen(monkeypatch, payload)
    with pytest.raises(RuntimeError, match="invalid grade"):
        judge(prompt="?", rollouts=["a", "b"], domain="math")


def test_judge_typeddict_export() -> None:
    """``JudgeResult`` is exported from the public surface so callers can
    type their own helpers without re-deriving the schema."""
    assert JudgeResult.__name__ == "JudgeResult"
    # TypedDict's runtime annotations expose the four declared fields.
    fields = set(JudgeResult.__annotations__)
    assert fields == {"grades", "critiques", "counterfactuals", "demonstration"}
