"""Seed verifier registry — pins the contract for W2 (unblocks PR L / TTRL).

Covers:

- Registry dispatch (known domain, unknown domain, exception swallowing).
- ``select(prompt)`` routing between math and code domains.
- Math verifier: answer extraction, non-math prompt returns None,
  fallback-last-number path, boxed + ``####`` + ``answer: N`` forms.
- Code verifier: expected-match pass, missing code block, runtime error,
  sandbox escape (``__import__`` blocked), wall-clock timeout.
"""
from __future__ import annotations

import threading

import pytest

pytestmark = pytest.mark.cpu_only

from lile.objectives.verifiers import VERIFIERS, register, select, verify
from lile.objectives.verifiers._math import extract_answer
from lile.objectives.verifiers._code import extract_code, extract_expected


# ---------------------------------------------------------------- registry


def test_registry_seeded_with_math_and_code():
    assert "math" in VERIFIERS
    assert "code" in VERIFIERS


def test_verify_unknown_domain_returns_none():
    assert verify("does_not_exist", "p", "c") is None


def test_verify_swallows_verifier_exceptions():
    @register("boom_test")
    def _boom(prompt: str, candidate: str):
        raise RuntimeError("verifier blew up")

    try:
        assert verify("boom_test", "p", "c") is None
    finally:
        VERIFIERS.pop("boom_test", None)


def test_register_decorator_returns_fn_unchanged():
    def _fn(p, c):
        return True
    try:
        wrapped = register("noop_test")(_fn)
        assert wrapped is _fn
        assert VERIFIERS["noop_test"] is _fn
    finally:
        VERIFIERS.pop("noop_test", None)


# ---------------------------------------------------------------- select


def test_select_math_for_arithmetic_prompt():
    assert select("How many apples are left if Jane has 3 and gives 1 away?") == "math"


def test_select_code_for_expected_output_prompt():
    prompt = "Write a program that prints hello. Expected: hello"
    assert select(prompt) == "code"


def test_select_none_for_unclaimed_prompt():
    assert select("tell me a story about dragons") is None


# ---------------------------------------------------------------- math


@pytest.mark.parametrize("text,expected", [
    ("The answer is 42.", "42"),
    ("#### 17", "17"),
    ("#### -3.50", "-3.5"),
    ("so we get \\boxed{128}.", "128"),
    ("Answer: 1,234", "1234"),
    ("Step 1: 2+2=4. Final: 12", "12"),  # fallback to last number
    ("no numbers here", None),
])
def test_math_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_math_verify_pass_on_math_prompt():
    assert verify("math", "How many apples? 3 + 4.", "So 3 + 4 = 7. #### 7") is True


def test_math_verify_none_on_non_math_prompt():
    # Non-math prompt: verifier declines rather than failing.
    assert verify("math", "tell me a joke", "7") is None


def test_math_verify_false_on_no_extractable_answer():
    assert verify("math", "What is 2 + 2?", "I don't know!") is False


# ---------------------------------------------------------------- code


def test_code_extract_expected_and_code_roundtrip():
    prompt = "Print the first 3 squares. Expected: 1 4 9"
    candidate = "```python\nprint(' '.join(str(i*i) for i in range(1,4)))\n```"
    assert extract_expected(prompt) == "1 4 9"
    assert extract_code(candidate) is not None


def test_code_verify_pass():
    prompt = "Print hello world. Expected: hello world"
    candidate = "```python\nprint('hello world')\n```"
    assert verify("code", prompt, candidate) is True


def test_code_verify_mismatch_is_false():
    prompt = "Expected: 1"
    candidate = "```python\nprint(2)\n```"
    assert verify("code", prompt, candidate) is False


def test_code_verify_none_when_prompt_has_no_expected():
    candidate = "```python\nprint(1)\n```"
    assert verify("code", "tell me a joke", candidate) is None


def test_code_verify_false_on_missing_fence():
    assert verify("code", "Expected: 1", "I would write print(1)") is False


def test_code_verify_false_on_runtime_error():
    prompt = "Expected: 1"
    candidate = "```python\nraise ValueError('nope')\n```"
    assert verify("code", prompt, candidate) is False


def test_code_verify_blocks_import():
    # Sandbox strips ``__import__`` from builtins, so ``import os`` fails
    # before it can exfiltrate anything — verify returns False, not True.
    prompt = "Expected: ok"
    candidate = "```python\nimport os\nprint('ok')\n```"
    assert verify("code", prompt, candidate) is False


def test_code_verify_false_on_infinite_loop():
    # SIGALRM trips after 1s — loop raises _ExecTimeout inside the sandbox,
    # adapter catches it, returns False.
    prompt = "Expected: never"
    candidate = "```python\nwhile True: pass\n```"
    assert verify("code", prompt, candidate) is False


def test_code_verify_works_from_worker_thread():
    """Regression: ``signal.signal`` is main-thread-only; off-thread callers
    used to silently drop to ``None`` because ``verify`` swallowed the
    ``ValueError``. The guard in ``_run_sandboxed`` skips alarm install off
    the main thread so the verification itself still runs."""
    prompt = "Expected: ok"
    candidate = "```python\nprint('ok')\n```"

    results: list[object] = []

    def _run():
        results.append(verify("code", prompt, candidate))

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert results == [True]
