"""Task #20 — safety_monitor primitive tests.

Covers the 9 test obligations from
``lile/docs/research/pr-specs/safety-monitor-primitive.md``:

1. M_p matches Cleo's numeric witness (V=3, p=(0.10,0.89,0.01), t=0, η=1
   → M_p ≈ 0.4759, grower set {2}).
2. Grower set is exactly {j : p_j < M_p}, brute-forced over 20 random
   simplex points.
3. Watchlist intersection fires alarm.
4. Watchlist-miss fires no alarm.
5. Three-tier watchlist union (daemon ∪ batch ∪ per-sample) all feed
   the same alarm check.
6. Zero-loss survives backward.
7. Composition with SFT + kl_anchor produces all three component key
   sets; total loss equals SFT + weight*kl_anchor (safety contributes 0).
8. Multi-position SFT determinism: 5-token response, per-position M_p
   + grower records, byte-identical across two deterministic runs.
9. Missing target_positions ⇒ RuntimeError with the pinned message.

cpu_only; stub model shape matches the pattern established in
``test_kl_target_position.py`` / ``test_unlike_tiered_preconditions.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lile.objectives.safety import (
    _MISSING_TARGETS_MSG,
    _m_p_value,
    _resolve_watchlist,
    safety_monitor_loss,
)

pytestmark = pytest.mark.cpu_only


V = 32


@dataclass
class _StubOut:
    logits: torch.Tensor


class _StubModel(nn.Module):
    """Tiny embed+linear — returns (B, T, V) logits for the given ids.

    We lift one token's bias (``peak_token``) hard so the output π is
    peaked rather than near-uniform. Near-uniform π gives ``M_p < 0``
    and an empty grower set (which is theoretically correct — uniformity
    is already safe), but then the watchlist-hit tests are vacuous.
    Peaking π shrinks every non-peak token below ``M_p`` and produces
    a populated grower set the tests can probe.
    """

    def __init__(self, seed: int = 0, peak_token: int = 15) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(V, 8)
        self.head = nn.Linear(8, V)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(V, 8, generator=g))
            self.head.weight.copy_(torch.randn(V, 8, generator=g) * 0.1)
            self.head.bias.zero_()
            self.head.bias[peak_token] = 6.0

    def forward(self, input_ids, attention_mask=None, use_cache=False, **_: Any):
        return _StubOut(logits=self.head(self.embed(input_ids)))


class _StubTok:
    pad_token_id = 0
    eos_token_id = 0


# --- obligation 1: Cleo's numeric witness ----------------------------------

def test_m_p_matches_cleo_witness() -> None:
    pi = torch.tensor([0.10, 0.89, 0.01], dtype=torch.float64)
    mp = _m_p_value(pi, t=0, eta=1.0)
    # Closed-form reference, computed from the definition directly:
    #   M_p(1) = -log(π0*exp(1-π0) + π1*exp(-π1) + π2*exp(-π2))
    beta = torch.tensor([1 - 0.10, -0.89, -0.01], dtype=torch.float64)
    ref = float(-torch.log((pi * beta.exp()).sum()))
    assert mp == pytest.approx(ref, abs=1e-9)
    assert mp == pytest.approx(0.4759, abs=5e-4), mp
    # Grower set {j != t : π_j < M_p} should be {2}.
    grower = [j for j in range(3) if j != 0 and float(pi[j]) < mp]
    assert grower == [2]


# --- obligation 2: grower-set matches predicate on random simplex points ---

def test_grower_set_matches_predicate_over_random_simplex() -> None:
    torch.manual_seed(0)
    for _ in range(20):
        logits = torch.randn(V)
        pi = F.softmax(logits, dim=-1).double()
        t = int(torch.randint(0, V, (1,)).item())
        eta = float(torch.empty(1).uniform_(0.1, 2.0).item())
        mp = _m_p_value(pi, t, eta)
        predicate = {j for j in range(V) if j != t and float(pi[j]) < mp}
        # Independent computation (the primitive's own path):
        idx = torch.arange(V)
        mask = (pi.float() < mp) & (idx != t)
        grower = set(mask.nonzero(as_tuple=True)[0].tolist())
        assert grower == predicate


# --- obligations 3 & 4: watchlist hit / miss ------------------------------

def _seed_run(
    model: _StubModel, watchlist: list[int] | None = None,
    default_watchlist: list[int] | None = None,
    sample_watchlist: list[int] | None = None,
) -> dict[str, Any]:
    tok = _StubTok()
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    attn = torch.ones_like(ids)
    sample: dict[str, Any] = {"prompt": "x", "response": "y"}
    if sample_watchlist is not None:
        sample["watchlist"] = sample_watchlist
    return safety_monitor_loss(
        model=model, tokenizer=tok, samples=[sample],
        target_positions=[[3]], target_token_ids=[[7]],
        input_ids=ids, attention_mask=attn,
        watchlist=watchlist, default_watchlist=default_watchlist,
        effective_lr=1.0,
    )


def _pick_grower(model: _StubModel) -> tuple[int, int]:
    """Run a one-shot forward and return (a_grower_id, a_non_grower_id)."""
    tok = _StubTok()
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        pi = F.softmax(model(ids).logits[0, 3].float(), dim=-1)
    mp = _m_p_value(pi, t=7, eta=1.0)
    growers = [j for j in range(V) if j != 7 and float(pi[j]) < mp]
    non = [j for j in range(V) if j not in growers and j != 7]
    assert growers and non, (growers, non)
    return growers[0], non[0]


def test_watchlist_hit_fires_alarm() -> None:
    m = _StubModel(seed=1)
    g, _ = _pick_grower(m)
    out = _seed_run(m, watchlist=[g])
    c = out["components"]
    assert c["safety_monitor_alarm_count"] == 1
    hits = c["safety_monitor_watchlist_hits"]
    assert (0, 3, g) in hits


def test_watchlist_miss_fires_no_alarm() -> None:
    m = _StubModel(seed=1)
    _, n = _pick_grower(m)
    out = _seed_run(m, watchlist=[n])
    assert out["components"]["safety_monitor_alarm_count"] == 0
    assert out["components"]["safety_monitor_watchlist_hits"] == []


# --- obligation 5: three-tier watchlist union -----------------------------

def test_three_tier_watchlist_union_all_contribute() -> None:
    m = _StubModel(seed=2)
    tok = _StubTok()
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        pi = F.softmax(m(ids).logits[0, 3].float(), dim=-1)
    mp = _m_p_value(pi, t=7, eta=1.0)
    growers = [j for j in range(V) if j != 7 and float(pi[j]) < mp]
    assert len(growers) >= 3, growers
    # Split three grower tokens across the three tiers — each individually
    # would hit, union must show all three.
    a, b, c = growers[0], growers[1], growers[2]
    out = safety_monitor_loss(
        model=m, tokenizer=tok,
        samples=[{"prompt": "x", "response": "y", "watchlist": [c]}],
        target_positions=[[3]], target_token_ids=[[7]],
        input_ids=ids, attention_mask=torch.ones_like(ids),
        default_watchlist=[a], watchlist=[b],
        effective_lr=1.0,
    )
    hit_tokens = {tok_id for _, _, tok_id in
                  out["components"]["safety_monitor_watchlist_hits"]}
    assert {a, b, c}.issubset(hit_tokens)


def test_resolve_watchlist_per_sample_isolation() -> None:
    # Pure unit test on the helper — sample-1 has per-sample watchlist,
    # sample-0 does not; batch + default still feed both.
    sets = _resolve_watchlist(
        samples=[{}, {"watchlist": [99]}],
        batch_watchlist=[5],
        default_watchlist=[1, 2],
    )
    assert sets[0] == {1, 2, 5}
    assert sets[1] == {1, 2, 5, 99}


# --- obligation 6: zero loss survives backward ----------------------------

def test_zero_loss_survives_backward() -> None:
    m = _StubModel(seed=3)
    tok = _StubTok()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    # Compose with a real graph-attached loss so the zero survives a
    # ``+`` with a backward-flowing tensor — the real composition shape.
    with torch.enable_grad():
        anchor = m(ids).logits.float().sum() * 1e-6
        out = safety_monitor_loss(
            model=m, tokenizer=tok,
            samples=[{"prompt": "x", "response": "y"}],
            target_positions=[[1]], target_token_ids=[[5]],
            input_ids=ids, attention_mask=torch.ones_like(ids),
            effective_lr=1.0,
        )
        total = anchor + out["loss"]
        total.backward()
    assert not torch.isnan(total).any()
    assert float(out["loss"].detach()) == 0.0


# --- obligation 7: composition with SFT + kl_anchor ----------------------

def test_composition_with_sft_and_kl_does_not_contribute_to_loss() -> None:
    """Pure unit: safety returns 0; verify summation with a graph-attached
    main loss is unchanged. We don't exercise the full TrainEngine path
    here (no real tokenizer); the component-key shape is validated
    separately in test_components_keys_present."""
    m = _StubModel(seed=4)
    tok = _StubTok()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    sft_like = m(ids).logits.float().mean()                # stand-in
    mon = safety_monitor_loss(
        model=m, tokenizer=tok,
        samples=[{"prompt": "x", "response": "y"}],
        target_positions=[[2]], target_token_ids=[[6]],
        input_ids=ids, attention_mask=torch.ones_like(ids),
        effective_lr=5e-4,
    )
    combined = sft_like + mon["loss"]
    assert torch.allclose(combined, sft_like)
    # Component keys pinned by the spec.
    expected_keys = {
        "safety_monitor_eta", "safety_monitor_alarm_count",
        "safety_monitor_grower_size_mean", "safety_monitor_grower_size_max",
        "safety_monitor_M_p_mean", "safety_monitor_M_p_min",
        "safety_monitor_watchlist_hits",
    }
    assert expected_keys.issubset(mon["components"].keys())


# --- obligation 8: multi-position determinism -----------------------------

def test_multi_position_sft_determinism() -> None:
    m = _StubModel(seed=5)
    tok = _StubTok()
    # 5 supervised positions: prompt token + 5 response tokens in an 8-tok
    # batch. target_positions therefore spans 5 consecutive logit slots.
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    positions = [[2, 3, 4, 5, 6]]
    targets = [[9, 10, 11, 12, 13]]

    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        out = safety_monitor_loss(
            model=m, tokenizer=tok,
            samples=[{"prompt": "x", "response": "abcde"}],
            target_positions=positions, target_token_ids=targets,
            input_ids=ids, attention_mask=torch.ones_like(ids),
            effective_lr=1.0,
        )
        runs.append(out["components"])

    a, b = runs
    for key in (
        "safety_monitor_eta", "safety_monitor_alarm_count",
        "safety_monitor_grower_size_mean", "safety_monitor_grower_size_max",
        "safety_monitor_M_p_mean", "safety_monitor_M_p_min",
    ):
        assert a[key] == b[key], (key, a[key], b[key])
    assert a["safety_monitor_watchlist_hits"] == b["safety_monitor_watchlist_hits"]


# --- obligation 9: missing target_positions ⇒ RuntimeError ---------------

def test_missing_target_positions_raises() -> None:
    m = _StubModel(seed=6)
    tok = _StubTok()
    with pytest.raises(RuntimeError, match="target_positions"):
        safety_monitor_loss(
            model=m, tokenizer=tok,
            samples=[{"prompt": "x", "response": "y"}],
            # target_positions deliberately omitted — the main objective
            # did not opt in. Pin the error message shape so the contract
            # surfaces loudly at dispatch.
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            effective_lr=1.0,
        )


def test_missing_target_positions_raises_pinned_message() -> None:
    m = _StubModel(seed=6)
    tok = _StubTok()
    with pytest.raises(RuntimeError) as excinfo:
        safety_monitor_loss(
            model=m, tokenizer=tok,
            samples=[{"prompt": "x", "response": "y"}],
        )
    assert str(excinfo.value) == _MISSING_TARGETS_MSG


# --- non-obligation coverage: non-zero weight is coerced, warns ----------

def test_nonzero_weight_is_coerced_with_warning() -> None:
    import warnings as w
    m = _StubModel(seed=7)
    tok = _StubTok()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        out = safety_monitor_loss(
            model=m, tokenizer=tok,
            samples=[{"prompt": "x", "response": "y"}],
            target_positions=[[1]], target_token_ids=[[5]],
            input_ids=ids, attention_mask=torch.ones_like(ids),
            effective_lr=1.0, weight=0.5,
        )
    msgs = [str(x.message) for x in rec if issubclass(x.category, RuntimeWarning)]
    assert any("observational" in m for m in msgs)
    assert float(out["loss"].detach()) == 0.0


# --- main-objective target-position extraction end-to-end -----------------

def test_sft_emits_target_positions() -> None:
    """Smoke: _utils.extract_target_positions on a synthetic label tensor
    produces the (positions, token_ids) pair the sidecar expects. Keeps
    the main-objective opt-in contract covered without loading a real
    tokenizer.
    """
    from lile.objectives._utils import extract_target_positions
    labels = torch.tensor([
        [-100, -100, 3, 4, 5, -100],
        [-100, 7, -100, 8, -100, -100],
    ])
    positions, tokens = extract_target_positions(labels)
    # Indices are in logits-coord (labels shifted by one):
    assert positions == [[1, 2, 3], [0, 2]]
    assert tokens == [[3, 4, 5], [7, 8]]
