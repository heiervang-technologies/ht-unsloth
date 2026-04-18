"""Unit tests for the unlike objective's trigger logic.

Covers the pure-function ``_should_trigger`` in lile/objectives/unlike.py.
The full end-to-end ``unlike_loss`` requires a model+tokenizer and is
covered by smoke_objectives.

Run: python -m lile.tests.test_unlike_trigger  (or pytest)
"""
from __future__ import annotations

import sys

import pytest

from lile.objectives.unlike import _should_trigger

pytestmark = pytest.mark.cpu_only


def test_argmax_hits_rank_below_5():
    assert _should_trigger(rank_bad=0, p_bad=0.4, rank_below=5, prob_above=None)


def test_top6_misses_rank_below_5():
    assert not _should_trigger(rank_bad=5, p_bad=0.001, rank_below=5, prob_above=None)


def test_rank_low_but_prob_high_still_fires():
    """Token ranks badly but carries high prob in a peakier distribution."""
    assert _should_trigger(rank_bad=10, p_bad=0.3, rank_below=5, prob_above=0.1)


def test_prob_below_threshold_and_rank_high_no_fire():
    assert not _should_trigger(rank_bad=20, p_bad=0.02,
                               rank_below=5, prob_above=0.1)


def test_either_criterion_alone_suffices():
    # rank-only spec, token is argmax → fire
    assert _should_trigger(rank_bad=0, p_bad=0.01, rank_below=1, prob_above=None)
    # prob-only spec, threshold exceeded → fire
    assert _should_trigger(rank_bad=100, p_bad=0.2, rank_below=None, prob_above=0.1)


def test_equality_on_prob_does_not_fire():
    """prob_above is strict >, not >= — boundary test."""
    assert not _should_trigger(rank_bad=3, p_bad=0.1, rank_below=None, prob_above=0.1)


def test_equality_on_rank_does_not_fire():
    """rank_below is strict <, not <=."""
    assert not _should_trigger(rank_bad=5, p_bad=0.0, rank_below=5, prob_above=None)


def test_both_none_raises():
    with pytest.raises(ValueError):
        _should_trigger(rank_bad=0, p_bad=1.0, rank_below=None, prob_above=None)


def main() -> int:
    tests = [
        test_argmax_hits_rank_below_5,
        test_top6_misses_rank_below_5,
        test_rank_low_but_prob_high_still_fires,
        test_prob_below_threshold_and_rank_high_no_fire,
        test_either_criterion_alone_suffices,
        test_equality_on_prob_does_not_fire,
        test_equality_on_rank_does_not_fire,
    ]
    for t in tests:
        t()
        print(f"[unlike-trigger] {t.__name__}")
    try:
        test_both_none_raises()
    except Exception:
        print("[unlike-trigger] test_both_none_raises FAILED")
        return 1
    print("[unlike-trigger] test_both_none_raises")
    return 0


if __name__ == "__main__":
    sys.exit(main())
