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
from lile.objectives.verifiers._math import answers_match, extract_answer
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


# ---------------------------------------------------------------- fraction canonicalization
#
# Mini-GSM8K regression (2026-04-17): on tutor_run_01_40steps the verifier
# scored 18/20; both misses were format-level. One was "5/8 vs 5" — the
# fraction answer "#### 5/8" in the candidate got extracted as "5" because
# the decimal regex stops at the slash. Fraction-aware extraction kicks in
# only when the prompt asks for a fraction, keeping GSM8K-exact behaviour
# untouched for non-fraction prompts.


def test_extract_default_mode_splits_fraction_as_today():
    # With no prompt hint, legacy behaviour wins: "5/8" → "5". This is the
    # byte-stable contract for plain GSM8K short-answer extraction.
    assert extract_answer("#### 5/8") == "5"


def test_extract_fraction_mode_keeps_denominator():
    """The mini-GSM8K miss: 'what fraction ...?' with candidate '#### 5/8'
    must now extract '5/8', not '5'."""
    prompt = "What fraction of the pizza did she eat?"
    assert extract_answer("So she ate 5/8. #### 5/8", prompt=prompt) == "5/8"


def test_extract_fraction_mode_handles_boxed_and_answer_label():
    prompt = "Express your answer as a fraction."
    assert extract_answer("so we get \\boxed{3/4}.", prompt=prompt) == "3/4"
    assert extract_answer("Answer: 7/12", prompt=prompt) == "7/12"


def test_extract_fraction_mode_falls_back_to_last_fraction():
    """No anchored fraction, but the response contains one — grab it."""
    prompt = "Give the answer as a fraction in lowest terms."
    assert extract_answer("It's simply 2/3.", prompt=prompt) == "2/3"


def test_extract_fraction_mode_falls_back_to_decimal_if_no_fraction():
    """Fraction mode but the candidate didn't produce a fraction: fall
    through to the decimal pattern rather than returning None. Keeps the
    verifier useful when the model ignores the formatting instruction."""
    prompt = "What fraction of the pie is left?"
    assert extract_answer("#### 1", prompt=prompt) == "1"


def test_extract_fraction_mode_ignores_date_like_slashes():
    """Word-boundary anchor on ``_FRACTION`` shouldn't match dates."""
    prompt = "What fraction?"
    # 3/4/2020 is a date, not a fraction answer; candidate has nothing else.
    assert extract_answer("The date was 3/4/2020.", prompt=prompt) is None


def test_extract_fraction_prompt_hint_is_narrow():
    """'fraction' appearing incidentally in the prompt doesn't flip mode."""
    # No answer-shape directive — stay in decimal mode and split "5/8" → "5".
    prompt = "Give the decimal equivalent of the fraction 5/8."
    assert extract_answer("#### 5/8", prompt=prompt) == "5"


# ---------------------------------------------------------------- answers_match
#
# The other mini-GSM8K miss: "43.98 vs 43.98226" — gold carried extra
# decimals, candidate was rounded. Exact match failed; with rtol=1e-3 it
# passes.


@pytest.mark.parametrize("ref,cand,expected", [
    # Precision miss from mini-GSM8K — default rtol=1e-3 reclaims it.
    ("43.98226", "43.98", True),
    # Fraction == decimal.
    ("5/8", "0.625", True),
    ("0.625", "5/8", True),
    # Whitespace / comma grouping.
    ("1,234", "1234", True),
    ("5 / 8", "5/8", True),
    # Exact integers.
    ("7", "7", True),
    # Real mismatches stay false under default rtol.
    ("42", "43", False),
    ("5/8", "3/8", False),
    # Unparseable strings compare False, not raise.
    ("foo", "bar", False),
    ("7", "seven", False),
    # Division by zero in a fraction → False, not raise.
    ("1/0", "1", False),
])
def test_answers_match_default_rtol(ref, cand, expected):
    assert answers_match(ref, cand) is expected


def test_answers_match_tighter_rtol_rejects_loose_matches():
    """Harness can tighten rtol for promotion-grade evals."""
    # With rtol=1e-3 this matches; at rtol=1e-6 it doesn't.
    assert answers_match("43.98226", "43.98", rtol=1e-3) is True
    assert answers_match("43.98226", "43.98", rtol=1e-6) is False


def test_answers_match_abs_tol_handles_near_zero():
    """rtol alone is useless around zero; abs_tol is the escape hatch."""
    assert answers_match("0", "0.0001") is False
    assert answers_match("0", "0.0001", abs_tol=1e-3) is True


def test_answers_match_none_inputs_return_false():
    assert answers_match(None, "5") is False
    assert answers_match("5", None) is False


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
