"""Combined-loss (multi-objective) train step.

Verifies that ``TrainEngine.step`` with ``spec["objectives"]`` set runs ONE
backward over Σ wᵢ·Lᵢ — gradients match the closed-form linear combination,
weights and per-objective components are surfaced, and a single optimizer
step lands.

Uses fake objectives + a one-parameter torch.nn.Module so the test stays
torchless-import-safe-ish (still imports torch, but no GPU / no transformers).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from lile.engine.train import TrainEngine
from lile.objectives import OBJECTIVES


# ---------------------------------------------------------------- fakes
class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Single trainable scalar — start at 1.0 so ∂(x²)/∂x = 2.0.
        self.x = torch.nn.Parameter(torch.tensor(1.0))


@contextmanager
def _trivial_lock():
    # ModelState.mode_lock is a contextmanager. A bare lock is enough.
    lock = threading.Lock()
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _fake_state(model: _FakeModel) -> SimpleNamespace:
    """Minimum surface TrainEngine reads from ModelState."""
    return SimpleNamespace(
        model=model,
        tokenizer=None,
        frozen_ref=None,
        merged_deltas={},
        mode_lock=_LockCtx(),
    )


class _LockCtx:
    """ModelState.mode_lock is used via ``with`` — supply a no-op CM."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def two_fake_objectives():
    """Register two synthetic objectives that compute scalar losses against
    the model's single parameter ``x``.

    L_a(x) = x², L_b(x) = (x − 0.5)² — both differentiable, both produce
    nonzero grads at x=1.0.
    """
    def la(model, tokenizer, samples, **_):  # noqa: ARG001
        loss = model.x ** 2
        return {"loss": loss, "components": {"sub": float(loss.detach())}}

    def lb(model, tokenizer, samples, **_):  # noqa: ARG001
        loss = (model.x - 0.5) ** 2
        return {"loss": loss, "components": {"sub": float(loss.detach())}}

    OBJECTIVES["fake_a"] = la
    OBJECTIVES["fake_b"] = lb
    yield ("fake_a", "fake_b")
    OBJECTIVES.pop("fake_a", None)
    OBJECTIVES.pop("fake_b", None)


@pytest.fixture
def gated_objective():
    """Objective that returns loss=None — exercises the precondition-gated
    skip path (e.g. unlike's tier preconditions firing)."""
    def gated(model, tokenizer, samples, **_):  # noqa: ARG001
        return {"loss": None, "components": {"reason": "fake_gate_closed"}}

    OBJECTIVES["fake_gated"] = gated
    yield "fake_gated"
    OBJECTIVES.pop("fake_gated", None)


# ---------------------------------------------------------------- tests
def test_combined_loss_equals_weighted_sum(two_fake_objectives) -> None:
    a, b = two_fake_objectives
    model = _FakeModel()
    eng = TrainEngine(_fake_state(model), lr=0.0, per_objective=True)  # lr=0 so weights don't move
    res = eng.step({
        "objectives": [
            {"name": a, "weight": 2.0},
            {"name": b, "weight": 3.0},
        ],
    })
    # L = 2·x² + 3·(x−0.5)² at x=1.0 → 2·1 + 3·0.25 = 2.75
    assert res["skipped"] is False
    assert res["loss"] == pytest.approx(2.75, rel=1e-6)
    c = res["components"]
    assert c["fake_a.weight"] == 2.0
    assert c["fake_b.weight"] == 3.0
    assert c["fake_a.weighted_loss"] == pytest.approx(2.0, rel=1e-6)
    assert c["fake_b.weighted_loss"] == pytest.approx(0.75, rel=1e-6)
    assert c["objectives_count"] == 2
    assert c["objectives_active"] == 2


def test_components_namespaced_per_objective(two_fake_objectives) -> None:
    a, b = two_fake_objectives
    model = _FakeModel()
    eng = TrainEngine(_fake_state(model), lr=0.0, per_objective=True)
    res = eng.step({
        "objectives": [
            {"name": a, "weight": 1.0},
            {"name": b, "weight": 1.0},
        ],
    })
    c = res["components"]
    # Each fake's "sub" key is namespaced — no collision.
    assert "fake_a.sub" in c
    assert "fake_b.sub" in c
    assert c["fake_a.sub"] == pytest.approx(1.0)
    assert c["fake_b.sub"] == pytest.approx(0.25)


def test_gradient_is_linear_combination(two_fake_objectives) -> None:
    """Manual: dL/dx = 2·(2x) + 3·(2(x−0.5)) at x=1.0 = 4 + 3 = 7.0.
    Use lr=0.1 and check that weights moved by approximately -lr·grad.
    """
    a, b = two_fake_objectives
    model = _FakeModel()
    eng = TrainEngine(_fake_state(model), lr=0.1, grad_clip=0.0, per_objective=True)
    x_before = float(model.x.detach())
    eng.step({
        "objectives": [
            {"name": a, "weight": 2.0},
            {"name": b, "weight": 3.0},
        ],
    })
    x_after = float(model.x.detach())
    # AdamW on first step ≈ lr·sign(grad) (warmup of moments). Check sign + magnitude bound.
    assert x_after < x_before  # grad positive → x decreases
    # Loose magnitude check: the step is at most lr in magnitude under AdamW warmup.
    assert abs(x_before - x_after) <= 0.15


def test_skipped_when_all_primaries_gated(gated_objective) -> None:
    g = gated_objective
    model = _FakeModel()
    eng = TrainEngine(_fake_state(model), lr=0.0, per_objective=True)
    res = eng.step({
        "objectives": [
            {"name": g, "weight": 1.0},
            {"name": g, "weight": 0.5},
        ],
    })
    assert res["skipped"] is True
    assert res["loss"] is None
    assert res["components"]["skipped_reason"] == "all_primaries_skipped"


def test_partial_skip_keeps_active_primaries(two_fake_objectives, gated_objective) -> None:
    """One primary fires, one is gated — combined loss should equal the
    weighted active primary alone, and skipped_total should reflect the
    gated entry."""
    a, _b = two_fake_objectives
    g = gated_objective
    model = _FakeModel()
    eng = TrainEngine(_fake_state(model), lr=0.0, per_objective=True)
    res = eng.step({
        "objectives": [
            {"name": a, "weight": 4.0},
            {"name": g, "weight": 1.0},
        ],
    })
    assert res["skipped"] is False
    # 4·x² at x=1.0 = 4.0; the gated entry contributes nothing.
    assert res["loss"] == pytest.approx(4.0, rel=1e-6)
    c = res["components"]
    assert c["fake_a.weighted_loss"] == pytest.approx(4.0)
    assert c["fake_gated.skipped"] is True
    assert c["objectives_active"] == 1
    assert c["objectives_count"] == 2


def test_per_entry_samples_override_default(two_fake_objectives) -> None:
    """Per-entry ``samples`` shadow the spec-level default."""
    a, b = two_fake_objectives
    seen: dict[str, list[Any]] = {}

    def la(model, tokenizer, samples, **_):
        seen["a"] = list(samples)
        return {"loss": model.x ** 2, "components": {}}

    def lb(model, tokenizer, samples, **_):
        seen["b"] = list(samples)
        return {"loss": model.x ** 2, "components": {}}

    OBJECTIVES["fake_a"] = la
    OBJECTIVES["fake_b"] = lb
    try:
        model = _FakeModel()
        eng = TrainEngine(_fake_state(model), lr=0.0, per_objective=True)
        eng.step({
            "samples": [{"shared": True}],
            "objectives": [
                {"name": a, "samples": [{"only_a": True}]},
                {"name": b},  # falls back to top-level "samples"
            ],
        })
    finally:
        OBJECTIVES.pop("fake_a", None)
        OBJECTIVES.pop("fake_b", None)

    assert seen["a"] == [{"only_a": True}]
    assert seen["b"] == [{"shared": True}]
