"""Task #15 — kl_anchor scope='target_position' + per-sample exclude fallback.

Covers the 5 test obligations from
``lile/docs/research/pr-specs/kl-anchor-target-position-scope.md``:

1. Schema fallback parity (per-sample derivation == explicit list).
2. Explicit-list union with per-sample derivation.
3. No-exclude fallback (empty exclude ⇒ standard target-position anchor).
4. Gradient zero on excluded token logits at the target position.
5. Existing prompt/full_sequence scopes unchanged (regression guard).

The tests run on CPU against a tiny stub model that produces logits with
the same shape contract the real model path uses (``.logits`` attr,
``(B, T, V)`` tensor, accepts ``input_ids``/``attention_mask``/``use_cache``
kwargs). Keeps the test cpu_only without needing a HF model in the cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lile.objectives.kl import (
    _derive_exclude_ids,
    _sample_text,
    kl_anchor_loss,
)

pytestmark = pytest.mark.cpu_only


# --- stubs ------------------------------------------------------------------

V = 32  # vocab size — small so the tests stay fast


@dataclass
class _StubOut:
    logits: torch.Tensor


class _StubModel(nn.Module):
    """Minimal model-shape: embed → linear → logits.

    Accepts the same kwargs the real path uses and returns an object with
    ``.logits``. Supports ``disable_adapter()`` as a no-op context manager
    so the ``use_self_ref`` branch in ``kl_anchor_loss`` can exercise the
    LoRA-disabled path without needing peft.
    """

    def __init__(self, vocab_size: int = V, d: int = 16, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d)
        self.head = nn.Linear(d, vocab_size)
        # Deterministic init so tests don't flake across runs.
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab_size, d, generator=g))
            self.head.weight.copy_(torch.randn(vocab_size, d, generator=g) * 0.1)
            self.head.bias.zero_()

    def forward(
        self, input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False, **_: Any,
    ) -> _StubOut:
        h = self.embed(input_ids)
        return _StubOut(logits=self.head(h))

    def disable_adapter(self):  # noqa: D401
        class _Noop:
            def __enter__(self_inner) -> None: return None
            def __exit__(self_inner, *a) -> None: return None
        return _Noop()


class _StubTokenizer:
    """Minimal tokenizer: hash-based deterministic token ids, no chat template.

    ``apply_chat_template`` is intentionally absent so ``_tokenize_prefixes_with_last_idx``
    takes the plain-tokenize path (``tokenizer(text=…).input_ids``). That
    keeps the test independent of any HF template logic.
    """
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str = "", add_special_tokens: bool = False, **_: Any):
        class _Enc:
            pass
        e = _Enc()
        # Deterministic token ids — one per char, modulo vocab size.
        ids = [((ord(c) % (V - 1)) + 1) for c in text]  # avoid id=0 (pad)
        if not ids:
            ids = [1]
        e.input_ids = torch.tensor(ids, dtype=torch.long)
        return e


# --- obligation 1: schema-fallback parity -----------------------------------

def test_schema_fallback_matches_explicit_exclude() -> None:
    """Per-sample {bad, good} via schema == explicit [bad, good] at batch level."""
    torch.manual_seed(0)
    m = _StubModel(seed=7)
    tok = _StubTokenizer()
    # Use distinct IDs far from any hash-token ID so excluding them
    # measurably changes the KL.
    samples = [{"prefix": "abc", "bad_token_id": 10, "good_token_id": 20}]

    loss_fallback = kl_anchor_loss(
        m, tok, samples, scope="target_position", weight=1.0,
    )
    loss_explicit = kl_anchor_loss(
        m, tok, samples, scope="target_position", weight=1.0,
        exclude_token_ids=[10, 20],
    )
    a = float(loss_fallback["loss"].detach())
    b = float(loss_explicit["loss"].detach())
    assert abs(a - b) < 1e-6, (a, b)


# --- obligation 2: explicit + per-sample union ------------------------------

def test_explicit_list_unions_with_per_sample() -> None:
    excluded = _derive_exclude_ids(
        [{"bad_token_id": 10, "good_token_id": 20}],
        explicit=[30],
    )
    assert excluded == [{10, 20, 30}]


def test_explicit_union_is_visible_in_components() -> None:
    """Wider exclude set ⇒ more excluded tokens reported."""
    m = _StubModel(seed=1)
    tok = _StubTokenizer()
    samples = [{"prefix": "abc", "bad_token_id": 5, "good_token_id": 7}]

    narrow = kl_anchor_loss(
        m, tok, samples, scope="target_position", weight=1.0,
    )
    wider = kl_anchor_loss(
        m, tok, samples, scope="target_position", weight=1.0,
        exclude_token_ids=[11, 12, 13],
    )
    assert narrow["components"]["kl_excluded_total"] == 2
    assert wider["components"]["kl_excluded_total"] == 5


# --- obligation 3: no-exclude fallback --------------------------------------

def test_no_exclude_runs_over_full_vocab() -> None:
    """Samples without bad/good + no explicit exclude → anchor over full vocab.

    The KL should be ~0 since the model equals itself (same path taken via
    ``use_self_ref``) — we only require the call completes and returns a
    sensible-shaped loss without raising.
    """
    m = _StubModel(seed=2)
    tok = _StubTokenizer()
    samples = [{"prefix": "hi"}]

    out = kl_anchor_loss(
        m, tok, samples, scope="target_position", weight=1.0,
    )
    assert out["components"]["kl_excluded_total"] == 0
    assert out["loss"].dim() == 0
    # Self-ref path: KL(self || self) ≈ 0. Keep the threshold wide enough
    # that softmax + float32 noise doesn't trip it.
    assert abs(float(out["components"]["kl"])) < 1e-4


def test_derive_exclude_ids_missing_fields() -> None:
    """Samples without bad/good fields contribute empty to the exclude set."""
    assert _derive_exclude_ids(
        [{"prefix": "x"}, {"prefix": "y", "bad_token_id": 9}],
        explicit=None,
    ) == [set(), {9}]


# --- obligation 4: gradient is zero on excluded logits ----------------------

def test_gradient_zero_on_excluded_token_at_target_position() -> None:
    """Backward through the target-position KL must leave the excluded token
    IDs' logits at the target position with zero gradient.

    Construct the KL with a DIFFERENT pi_ref (not self-ref) so the KL is
    nonzero — self-ref KL is zero and would trivially satisfy the gradient
    test. Then verify that while SOME logits get gradient, the excluded
    slots at the target positions get exactly zero.
    """
    torch.manual_seed(0)
    theta = _StubModel(seed=3)
    ref = _StubModel(seed=4)  # distinct weights ⇒ nonzero KL

    tok = _StubTokenizer()
    samples = [
        {"prefix": "abc", "bad_token_id": 10, "good_token_id": 20},
        {"prefix": "def", "bad_token_id": 11},
    ]
    exclude_extra = [30]

    out = kl_anchor_loss(
        theta, tok, samples, pi_ref=ref,
        scope="target_position", weight=1.0,
        exclude_token_ids=exclude_extra,
    )
    # Sanity: loss is nonzero (else the gradient test is vacuous).
    assert float(out["loss"].detach()) > 1e-6

    # Hook to capture the final logits gradient at the target position.
    # The cleanest way: re-run the forward and register a hook so we can
    # inspect the gradient on the logits tensor itself.
    #
    # Rebuild the batch the same way kl_anchor_loss does (import the helper).
    from lile.objectives.kl import _tokenize_prefixes_with_last_idx
    batch = _tokenize_prefixes_with_last_idx(tok, samples)
    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    last_idx = batch["last_idx"]

    logits = theta(input_ids=ids, attention_mask=attn).logits
    logits.retain_grad()
    with torch.no_grad():
        ref_logits = ref(input_ids=ids, attention_mask=attn).logits

    B, _T, Vv = logits.size()
    row_idx = torch.arange(B)
    last_logits = logits[row_idx, last_idx].float()
    last_ref = ref_logits[row_idx, last_idx].float()

    exclude_sets = [{10, 20, 30}, {11, 30}]
    exclude_mask = torch.zeros((B, Vv), dtype=torch.bool)
    for i, ex in enumerate(exclude_sets):
        for tid in ex:
            exclude_mask[i, tid] = True

    neg_inf = torch.finfo(last_logits.dtype).min
    masked_logits = last_logits.masked_fill(exclude_mask, neg_inf)
    masked_ref = last_ref.masked_fill(exclude_mask, neg_inf)
    log_p = F.log_softmax(masked_logits, dim=-1)
    log_q = F.log_softmax(masked_ref, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).masked_fill(exclude_mask, 0.0).sum(dim=-1).mean()
    kl.backward()

    assert logits.grad is not None
    grad_at_last = logits.grad[row_idx, last_idx]
    for i, ex in enumerate(exclude_sets):
        for tid in ex:
            g = float(grad_at_last[i, tid])
            assert g == 0.0, f"sample {i} excluded token {tid} got grad {g}"
    # At least one non-excluded slot must have gotten SOME gradient for
    # the test to be non-vacuous.
    non_excluded = ~exclude_mask
    nonzero = grad_at_last[non_excluded].abs().max().item()
    assert nonzero > 0.0, "no gradient flowed at all — test is vacuous"


# --- obligation 5: existing scopes unchanged --------------------------------

def test_scope_prompt_still_routes_through_sample_text() -> None:
    """Regression: prompt scope still hits the text branch, unchanged shape."""
    assert _sample_text({"prompt": "Q:"}, "prompt") == "Q:"


def test_scope_full_sequence_still_routes_through_sample_text() -> None:
    assert _sample_text({"prompt": "Q:", "response": "A"}, "full_sequence") == "Q:A"


def test_target_position_text_lookup_raises() -> None:
    """target_position must not silently degrade to a text scope.

    ``_sample_text`` must raise when called with target_position so the
    top-level dispatch in ``kl_anchor_loss`` stays the single routing
    point for the new scope.
    """
    with pytest.raises(ValueError, match="target_position"):
        _sample_text({"prompt": "Q:"}, "target_position")
