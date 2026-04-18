"""End-to-end unit test for unlike_loss using a stub model+tokenizer.

Exercises the full forward path without loading a real LM: a torch.nn.Linear
stands in for the model, returning synthetic logits whose argmax and
distribution shape we control.

Run: python -m lile.tests.test_unlike_loss  (or pytest)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from lile.objectives.unlike import unlike_loss

pytestmark = pytest.mark.cpu_only


VOCAB_SIZE = 32
HIDDEN = 8


class StubTokenizer:
    """No chat_template, no special tokens — just word-level IDs."""
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0}

    def _tok_id(self, w: str) -> int:
        if w not in self._vocab:
            self._vocab[w] = len(self._vocab) % VOCAB_SIZE
        return self._vocab[w]

    def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
        ids = [self._tok_id(w) for w in text.split()]
        return SimpleNamespace(input_ids=ids)


class StubModel(nn.Module):
    """Linear over a tiny embedding — produces deterministic logits from ids."""
    def __init__(self, target_token: int | None = None) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB_SIZE)
        self._target = target_token
        # Initialize head so the chosen target_token dominates logits.
        if target_token is not None:
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.zero_()
                self.head.bias[target_token] = 10.0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                use_cache: bool = False) -> Any:
        h = self.embed(input_ids)
        logits = self.head(h)
        return SimpleNamespace(logits=logits)


def test_triggered_sample_contributes_positive_loss():
    """When bad_token == argmax, unlike term is -log(1 - p_bad) >> 0."""
    torch.manual_seed(0)
    tok = StubTokenizer()
    bad = 7
    model = StubModel(target_token=bad)
    out = unlike_loss(
        model=model, tokenizer=tok,
        samples=[{"prefix": "hello world", "bad_token_id": bad,
                  "rank_below": 5}],
    )
    assert out["components"]["unlike_triggered"] == 1
    assert out["loss"].item() > 0.1
    assert out["components"]["unlike_p_bad_mean"] > 0.5  # bias 10 dominates


def test_non_triggered_sample_zero_ul():
    """Bad token ranks low → not triggered → ul loss = 0 (no good teacher, so total=0)."""
    torch.manual_seed(0)
    tok = StubTokenizer()
    # Target a DIFFERENT token than "bad" so p_bad is tiny.
    model = StubModel(target_token=3)
    out = unlike_loss(
        model=model, tokenizer=tok,
        samples=[{"prefix": "hello world", "bad_token_id": 7,
                  "rank_below": 1, "prob_above": 0.9}],
    )
    assert out["components"]["unlike_triggered"] == 0
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_positive_teacher_fires_even_when_not_triggered():
    """good_token_id adds -log p(good) independent of trigger state."""
    torch.manual_seed(0)
    tok = StubTokenizer()
    model = StubModel(target_token=3)
    out = unlike_loss(
        model=model, tokenizer=tok,
        samples=[{"prefix": "hello world", "bad_token_id": 7,
                  "good_token_id": 11,
                  "rank_below": 1, "prob_above": 0.9}],
    )
    # Not triggered but positive teacher runs.
    assert out["components"]["unlike_triggered"] == 0
    assert out["loss"].item() > 0.1  # -log p(good=11), with target_token=3 → low p(11)


def test_mixed_batch_only_triggered_samples_contribute_ul():
    """Two samples, only one has bad_token == argmax. Triggered count = 1."""
    torch.manual_seed(0)
    tok = StubTokenizer()
    model = StubModel(target_token=5)
    # Stub head has weight=0 + bias[5]=10, so only token 5 is argmax; every
    # other token is tied at logit 0. Use rank_below=1 (strict argmax-only)
    # so the non-matching sample does not trigger by rank either.
    out = unlike_loss(
        model=model, tokenizer=tok,
        samples=[
            {"prefix": "hello world", "bad_token_id": 5,
             "rank_below": 1, "prob_above": None},                        # argmax-hit
            {"prefix": "foo bar baz", "bad_token_id": 9,
             "rank_below": 1, "prob_above": None},                        # rank>=1, no trigger
        ],
    )
    assert out["components"]["unlike_triggered"] == 1
    assert out["components"]["unlike_n"] == 2


def test_gradient_flows_into_model():
    """loss.backward() should populate grads on the head bias for bad_token."""
    torch.manual_seed(0)
    tok = StubTokenizer()
    bad = 7
    model = StubModel(target_token=bad)
    model.zero_grad()
    out = unlike_loss(
        model=model, tokenizer=tok,
        samples=[{"prefix": "hello world", "bad_token_id": bad,
                  "rank_below": 5}],
    )
    out["loss"].backward()
    # The bad-token bias should have received a positive gradient (since we
    # minimize -log(1 - p_bad), d/d(bias_bad) is positive = push bias down).
    grad = model.head.bias.grad
    assert grad is not None
    assert grad[bad].item() > 0.01


def main() -> int:
    tests = [
        test_triggered_sample_contributes_positive_loss,
        test_non_triggered_sample_zero_ul,
        test_positive_teacher_fires_even_when_not_triggered,
        test_mixed_batch_only_triggered_samples_contribute_ul,
        test_gradient_flows_into_model,
    ]
    for t in tests:
        t()
        print(f"[unlike-loss] {t.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
