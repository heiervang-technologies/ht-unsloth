"""PR G — KL anchor scope flag (full-sequence vs prompt-only).

Unit tests the pure-function helper _sample_text. The full end-to-end
kl_anchor_loss path needs a model+tokenizer and is covered by the existing
smoke_objectives tests once the flag is plumbed through the controller.

Run: python -m lile.tests.test_kl_scope
"""
from __future__ import annotations

import sys

import pytest

from lile.objectives.kl import _sample_text

pytestmark = pytest.mark.cpu_only


def test_scope_prompt_ignores_response():
    s = {"prompt": "Q:", "response": "A.", "chosen": "x", "good": "y"}
    assert _sample_text(s, "prompt") == "Q:"


def test_scope_full_sequence_prefers_response():
    s = {"prompt": "Q:", "response": "A.", "chosen": "x", "good": "y"}
    assert _sample_text(s, "full_sequence") == "Q:A."


def test_scope_full_sequence_falls_back_to_good():
    """No response → uses next field in the priority order (good before chosen)."""
    s = {"prompt": "Q:", "good": "y", "chosen": "x"}
    assert _sample_text(s, "full_sequence") == "Q:y"


def test_scope_full_sequence_falls_back_to_chosen():
    s = {"prompt": "Q:", "chosen": "x"}
    assert _sample_text(s, "full_sequence") == "Q:x"


def test_scope_full_sequence_no_response_noop():
    """No response-like field at all → returns prompt unchanged (no crash)."""
    s = {"prompt": "Q:"}
    assert _sample_text(s, "full_sequence") == "Q:"


def test_scope_full_sequence_skips_empty_response():
    """Empty-string response should be skipped, not concatenated."""
    s = {"prompt": "Q:", "response": "", "chosen": "x"}
    assert _sample_text(s, "full_sequence") == "Q:x"


def test_unknown_scope_raises():
    s = {"prompt": "Q:"}
    with pytest.raises(ValueError):
        _sample_text(s, "bogus")


def test_prefix_falls_back_when_no_prompt():
    """unlike objective samples carry ``prefix`` not ``prompt`` — kl_anchor
    composition must accept either."""
    s = {"prefix": "The antagonist is "}
    assert _sample_text(s, "prompt") == "The antagonist is "


def test_prompt_wins_over_prefix_when_both_present():
    s = {"prompt": "Q:", "prefix": "X:"}
    assert _sample_text(s, "prompt") == "Q:"


def test_empty_prefix_no_crash():
    """No prompt, no prefix → empty string (not KeyError)."""
    s: dict = {}
    assert _sample_text(s, "prompt") == ""


def main() -> int:
    test_scope_prompt_ignores_response()
    print("[kl-scope] scope='prompt' ignores response")
    test_scope_full_sequence_prefers_response()
    print("[kl-scope] scope='full_sequence' prefers 'response'")
    test_scope_full_sequence_falls_back_to_good()
    print("[kl-scope] fallback to 'good'")
    test_scope_full_sequence_falls_back_to_chosen()
    print("[kl-scope] fallback to 'chosen'")
    test_scope_full_sequence_no_response_noop()
    print("[kl-scope] no response field → prompt unchanged")
    test_scope_full_sequence_skips_empty_response()
    print("[kl-scope] empty response is skipped")
    try:
        test_unknown_scope_raises()
    except Exception:
        print("[kl-scope] unknown scope test failed")
        return 1
    print("[kl-scope] unknown scope raises")
    test_prefix_falls_back_when_no_prompt()
    print("[kl-scope] prefix fallback when no prompt")
    test_prompt_wins_over_prefix_when_both_present()
    print("[kl-scope] prompt wins when both present")
    test_empty_prefix_no_crash()
    print("[kl-scope] empty dict does not raise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
