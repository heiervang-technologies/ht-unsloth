"""Task #19 — unlike tiered preconditions (4 tiers).

Covers the 10 test obligations from
``lile/docs/research/pr-specs/unlike-tiered-preconditions.md``:

1. Pure-unlike no anchor → ValueError (Tier 1).
2. Pure-unlike + allow_unanchored=True → runs.
3. Pure-unlike + scope='prompt' → RuntimeWarning (Tier 2).
4. Pure-unlike + scope='target_position' → no warning (clean path).
5. Positive-teacher + no anchor → no Tier-1 error.
6. Positive-teacher + no anchor → no Tier 1-3 warning either.
7. Existing unlike tests still pass (regression — see test_unlike_loss.py).
8. Tier 4 fires on small-η positive-teacher sample.
9. Tier 4 does not fire at the floor (exact floor + just above).
10. Tier 4 + Tier 1 together → ValueError preempts warn.

Stub model/tokenizer mirror the pattern in ``test_kl_target_position.py``:
a tiny embed+linear stands in for the real LM so the tests stay cpu_only
and don't download weights.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn

from lile.objectives.unlike import (
    _UNLIKE_LR_HEURISTIC_FLOOR,
    _check_preconditions,
    unlike_loss,
)

pytestmark = pytest.mark.cpu_only


# --- stubs (minimal shape: embed → linear; deterministic init) --------------

V = 32


@dataclass
class _StubOut:
    logits: torch.Tensor


class _StubModel(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(V, 8)
        self.head = nn.Linear(8, V)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(V, 8, generator=g))
            self.head.weight.copy_(torch.randn(V, 8, generator=g) * 0.1)
            self.head.bias.zero_()

    def forward(self, input_ids, attention_mask=None, use_cache=False, **_: Any):
        return _StubOut(logits=self.head(self.embed(input_ids)))


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str = "", add_special_tokens: bool = False, **_: Any):
        class _E:
            pass
        e = _E()
        ids = [((ord(c) % (V - 1)) + 1) for c in text] or [1]
        e.input_ids = torch.tensor(ids, dtype=torch.long)
        return e


def _mk() -> tuple[_StubModel, _StubTokenizer]:
    return _StubModel(seed=0), _StubTokenizer()


# --- obligation 1: Tier 1 error ---------------------------------------------

def test_pure_unlike_no_anchor_raises() -> None:
    m, tok = _mk()
    with pytest.raises(ValueError, match="pure-unlike requires kl_anchor"):
        unlike_loss(
            m, tok,
            samples=[{"prefix": "abc", "bad_token_id": 7, "rank_below": 5}],
            batch_objectives=[],
        )


# --- obligation 2: Tier 1 escape hatch --------------------------------------

def test_pure_unlike_allow_unanchored_runs() -> None:
    m, tok = _mk()
    out = unlike_loss(
        m, tok,
        samples=[{"prefix": "abc", "bad_token_id": 7, "rank_below": 5}],
        batch_objectives=[],
        allow_unanchored=True,
    )
    assert "loss" in out
    assert out["components"]["unlike_n"] == 1


# --- obligation 3: Tier 2 warn on wrong scope -------------------------------

def test_pure_unlike_prompt_scope_warns() -> None:
    m, tok = _mk()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        unlike_loss(
            m, tok,
            samples=[{"prefix": "abc", "bad_token_id": 7, "rank_below": 5}],
            batch_objectives=[
                {"name": "kl_anchor", "scope": "prompt", "weight": 0.1},
            ],
        )
    msgs = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
    assert any("scope='prompt'" in m for m in msgs), msgs


# --- obligation 4: Tier 2+3 silent on target_position -----------------------

def test_pure_unlike_target_position_scope_no_warn() -> None:
    m, tok = _mk()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        unlike_loss(
            m, tok,
            samples=[{"prefix": "abc", "bad_token_id": 7, "rank_below": 5}],
            batch_objectives=[
                {"name": "kl_anchor", "scope": "target_position",
                 "weight": 0.1},
            ],
            effective_lr=5e-4,
        )
    tier_warns = [
        w for w in rec
        if issubclass(w.category, RuntimeWarning)
        and ("scope=" in str(w.message) or "exclude_token_ids" in str(w.message))
    ]
    assert tier_warns == [], [str(w.message) for w in tier_warns]


# --- obligation 5: positive teacher + no anchor → no Tier-1 error -----------

def test_positive_teacher_no_anchor_no_error() -> None:
    m, tok = _mk()
    out = unlike_loss(
        m, tok,
        samples=[{"prefix": "abc", "bad_token_id": 7, "good_token_id": 11,
                  "rank_below": 5}],
        batch_objectives=[],
        effective_lr=5e-4,
    )
    assert out["components"]["unlike_n"] == 1


# --- obligation 6: positive teacher → no Tier 1-3 warning -------------------

def test_positive_teacher_no_anchor_no_tier123_warn() -> None:
    m, tok = _mk()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        unlike_loss(
            m, tok,
            samples=[{"prefix": "abc", "bad_token_id": 7, "good_token_id": 11,
                      "rank_below": 5}],
            batch_objectives=[],
            effective_lr=5e-4,
        )
    # Tiers 1-3 all mention scope= / exclude_token_ids / kl_anchor.
    tier123 = [
        w for w in rec
        if issubclass(w.category, RuntimeWarning)
        and ("scope=" in str(w.message)
             or "exclude_token_ids" in str(w.message)
             or "kl_anchor" in str(w.message))
    ]
    assert tier123 == [], [str(w.message) for w in tier123]


# --- obligation 7: bare-primitive call skips tiers (regression guard) -------

def test_bare_primitive_call_runs_without_preconditions() -> None:
    """Existing test_unlike_loss.py calls unlike_loss without batch_objectives
    or effective_lr. Those dispatches must still run — the preconditions
    gate is keyed on the kwargs being explicitly passed in, so bare
    primitive use (research REPL, unit tests) opts out cleanly.
    """
    m, tok = _mk()
    # Pure-unlike without any batch_objectives kwarg at all — no tier fires.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = unlike_loss(
            m, tok,
            samples=[{"prefix": "abc", "bad_token_id": 7, "rank_below": 5}],
        )
    assert "loss" in out
    tier_warns = [
        w for w in rec if issubclass(w.category, RuntimeWarning)
    ]
    assert tier_warns == [], [str(w.message) for w in tier_warns]


# --- obligation 8: Tier 4 fires on small-η positive-teacher sample ----------

def test_tier4_fires_on_small_eta_positive_teacher() -> None:
    m, tok = _mk()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        unlike_loss(
            m, tok,
            samples=[{"prefix": "abc", "bad_token_id": 7,
                      "good_token_id": 11, "rank_below": 5}],
            batch_objectives=[],
            effective_lr=1e-5,
        )
    tier4 = [
        w for w in rec
        if issubclass(w.category, RuntimeWarning)
        and "known-unsafe regime" in str(w.message)
    ]
    assert len(tier4) == 1, [str(w.message) for w in rec]
    assert "effective_lr=1e-05" in str(tier4[0].message)


# --- obligation 9: Tier 4 silent at / above the floor -----------------------

def test_tier4_silent_at_floor() -> None:
    m, tok = _mk()
    for lr in (_UNLIKE_LR_HEURISTIC_FLOOR, 5.1e-5):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            unlike_loss(
                m, tok,
                samples=[{"prefix": "abc", "bad_token_id": 7,
                          "good_token_id": 11, "rank_below": 5}],
                batch_objectives=[],
                effective_lr=lr,
            )
        tier4 = [
            w for w in rec
            if issubclass(w.category, RuntimeWarning)
            and "known-unsafe regime" in str(w.message)
        ]
        assert tier4 == [], f"tier4 fired at lr={lr:g}"


# --- obligation 10: Tier 1 preempts Tier 4 ---------------------------------

def test_tier1_preempts_tier4() -> None:
    m, tok = _mk()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="pure-unlike requires kl_anchor"):
            unlike_loss(
                m, tok,
                samples=[{"prefix": "abc", "bad_token_id": 7,
                          "rank_below": 5}],
                batch_objectives=[],
                effective_lr=1e-6,
            )
    # Tier 4 must NOT have emitted before the raise.
    tier4 = [
        w for w in rec
        if issubclass(w.category, RuntimeWarning)
        and "known-unsafe regime" in str(w.message)
    ]
    assert tier4 == [], "tier4 fired before tier1 raised"


# --- pure unit tests on the gate itself -------------------------------------

def test_check_preconditions_none_batch_objectives_skips_tiers_123() -> None:
    # Pure-unlike, no anchor — would be Tier 1 — but batch_objectives=None
    # means "called bare". Must not raise.
    _check_preconditions(
        samples=[{"bad_token_id": 1}],
        batch_objectives=None,
        allow_unanchored=False,
        effective_lr=None,
    )


def test_check_preconditions_none_effective_lr_skips_tier4() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _check_preconditions(
            samples=[{"bad_token_id": 1, "good_token_id": 2}],
            batch_objectives=[],
            allow_unanchored=False,
            effective_lr=None,
        )
    assert rec == []
