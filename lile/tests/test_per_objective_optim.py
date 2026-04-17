"""PR B — per-objective optimizer instances isolate Adam m/v per family.

PyTorch keys ``optimizer.state[param]`` by tensor id, so N param_groups
over the same LoRA params share one running ``m``/``v``. Only per-objective
optimizer *instances* isolate the second-moment running variance — which
is what we want when mixing objectives with substantially different grad
magnitudes (SFT-on-tokens vs. hinge on pairwise margins vs. CCPD on
response gradients).

See ``lile/docs/research/optimizer-sample-efficiency.md`` §3 + the
anti-patterns correction note. This file pins:

  1. Off-by-default: flag must be False on ``ServeConfig`` and one shared
     optimizer is reused across objectives.
  2. Instance identity: in per-objective mode, distinct names produce
     distinct ``torch.optim.Optimizer`` instances.
  3. State isolation: ``opt_sft.state is not opt_kto.state`` — an object-
     identity guard separate from the ||Δθ|| ratio dynamic check.
  4. Reset clears the whole dict, not one key (snapshot rewinds the
     shared weights every instance's state is keyed to).
  5. ``||Δθ||`` ratio sanity: after the same gradient sequence, the
     per-objective path produces a weight delta of the same order of
     magnitude as the shared path. Not bit-exact (8bit vs 32bit), but
     within a loose band — the regression we guard against is "per-
     objective mode silently produces zero or runaway updates".
"""
from __future__ import annotations

import pytest
import torch

from lile.config import ServeConfig
from lile.engine.train import TrainEngine

pytestmark = pytest.mark.cpu_only


# ---------- fixtures ---------------------------------------------------------


class _ToyState:
    """Mimics ``ModelState`` with the surface ``TrainEngine`` touches for
    optimizer construction. Real training steps are not exercised here —
    ``test_residual_live_path.py`` covers that on GPU.
    """

    def __init__(self, n_params: int = 2) -> None:
        self.model = torch.nn.Sequential(
            *[torch.nn.Linear(4, 4, bias=False) for _ in range(n_params)]
        )
        # Simulate LoRA-style: every param trainable.
        for p in self.model.parameters():
            p.requires_grad_(True)


def _fresh_engine(per_objective: bool, **kwargs) -> TrainEngine:
    return TrainEngine(_ToyState(), lr=1e-3, per_objective=per_objective, **kwargs)


# ---------- 1. default-off ---------------------------------------------------


def test_flag_default_off_on_serve_config():
    cfg = ServeConfig()
    assert cfg.per_objective_optim is False, (
        "per_objective_optim must be False by default — VRAM cost is real"
    )
    assert cfg.per_objective_lr == {}, (
        "per_objective_lr default must be an empty dict"
    )


def test_shared_mode_reuses_single_optimizer_across_objectives():
    engine = _fresh_engine(per_objective=False)
    opt_sft = engine._optimizer("sft")
    opt_kto = engine._optimizer("kto")
    opt_coh = engine._optimizer("coh")
    assert opt_sft is opt_kto is opt_coh, (
        "shared mode must reuse one optimizer regardless of objective name"
    )
    assert len(engine._opts) == 1


# ---------- 2. instance identity --------------------------------------------


def test_per_objective_mode_produces_distinct_instances():
    engine = _fresh_engine(per_objective=True)
    opt_sft = engine._optimizer("sft")
    opt_kto = engine._optimizer("kto")
    assert opt_sft is not opt_kto, (
        "per-objective mode must create a distinct optimizer per name"
    )
    # Same name → same instance (lazy cache).
    assert engine._optimizer("sft") is opt_sft


# ---------- 3. state isolation (Mei's identity guard) -----------------------


def test_per_objective_mode_isolates_optimizer_state():
    """Static object-identity guard. If someone regresses this to
    ``torch.optim.AdamW(params, lr=lr, ...)`` via shared param_groups on
    one instance, the ``state`` dicts would be the same object even
    though they look like separate dicts in logs.
    """
    engine = _fresh_engine(per_objective=True)
    opt_sft = engine._optimizer("sft")
    opt_kto = engine._optimizer("kto")
    assert opt_sft.state is not opt_kto.state, (
        "opt_sft.state and opt_kto.state must be distinct dict objects"
    )
    # And they must be instances, not the same class-level default.
    assert isinstance(opt_sft.state, dict)
    assert isinstance(opt_kto.state, dict)


# ---------- 4. reset clears the whole dict ----------------------------------


def test_reset_optimizer_clears_all_instances_in_per_objective_mode():
    engine = _fresh_engine(per_objective=True)
    engine._optimizer("sft")
    engine._optimizer("kto")
    engine._optimizer("coh")
    assert len(engine._opts) == 3

    engine.reset_optimizer()

    assert engine._opts == {}, (
        "reset_optimizer must clear every per-objective instance — snapshot "
        "rewinds the shared weights that every instance's state is keyed to"
    )


def test_per_objective_lr_override_applied():
    engine = _fresh_engine(
        per_objective=True,
        per_objective_lr={"sft": 5e-4, "kto": 2e-5},
    )
    opt_sft = engine._optimizer("sft")
    opt_kto = engine._optimizer("kto")
    opt_default = engine._optimizer("coh")  # falls back to engine.lr

    assert opt_sft.param_groups[0]["lr"] == 5e-4
    assert opt_kto.param_groups[0]["lr"] == 2e-5
    assert opt_default.param_groups[0]["lr"] == 1e-3  # engine.lr


# ---------- 5. ||Δθ|| ratio sanity ------------------------------------------


def _run_one_step(engine: TrainEngine, objective: str, seed: int) -> torch.Tensor:
    """Deterministic single-step update: seeds grads, calls `.step()`, returns
    the flat ``Δθ`` across every trainable param on the toy model.
    """
    params = [p for p in engine.state.model.parameters() if p.requires_grad]
    before = torch.cat([p.detach().flatten().clone() for p in params])

    opt = engine._optimizer(objective)
    opt.zero_grad()
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g) * 1e-2
    opt.step()

    after = torch.cat([p.detach().flatten().clone() for p in params])
    return after - before


def test_exp_avg_sq_buffers_isolated_after_two_steps(monkeypatch):
    """End-to-end state isolation: step sft with grad_A, step kto with
    grad_B, then confirm each optimizer's ``exp_avg_sq`` buffer reflects
    only its own family's gradients. This is the definitive gate —
    static identity plus ||Δθ|| ratio cover regressions indirectly, but
    a malformed dispatch that accidentally routed both steps to one
    instance would still pass those checks yet fail this one.

    Adam's ``exp_avg_sq`` is an EMA of g², so with the default β₂=0.999
    after a single step: ``exp_avg_sq = 0.001 * g²`` (plus tiny fp noise).
    """
    import builtins
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (
        real_import(n, *a, **k) if n != "bitsandbytes"
        else (_ for _ in ()).throw(ImportError("forced"))
    ))

    engine = _fresh_engine(per_objective=True)
    params = [p for p in engine.state.model.parameters() if p.requires_grad]

    # Step sft with one seed.
    opt_sft = engine._optimizer("sft")
    opt_sft.zero_grad()
    g_sft = torch.Generator().manual_seed(100)
    grads_sft = [torch.randn(p.shape, generator=g_sft) * 1e-2 for p in params]
    for p, g in zip(params, grads_sft):
        p.grad = g.clone()
    opt_sft.step()

    # Step kto with a different seed.
    opt_kto = engine._optimizer("kto")
    opt_kto.zero_grad()
    g_kto = torch.Generator().manual_seed(200)
    grads_kto = [torch.randn(p.shape, generator=g_kto) * 1e-2 for p in params]
    for p, g in zip(params, grads_kto):
        p.grad = g.clone()
    opt_kto.step()

    # Each optimizer's state entry for a given param must reflect ONLY
    # that optimizer's gradient. With β₂=0.999 and one step:
    #   exp_avg_sq ≈ (1 - β₂) * g² = 1e-3 * g²
    for p, g_s, g_k in zip(params, grads_sft, grads_kto):
        sft_exp_avg_sq = opt_sft.state[p]["exp_avg_sq"]
        kto_exp_avg_sq = opt_kto.state[p]["exp_avg_sq"]

        # sft's buffer should be proportional to sft's grad², not kto's.
        assert torch.allclose(sft_exp_avg_sq, 1e-3 * g_s.pow(2), atol=1e-9), (
            "opt_sft.exp_avg_sq diverged from sft's own grad² — state bleed"
        )
        assert torch.allclose(kto_exp_avg_sq, 1e-3 * g_k.pow(2), atol=1e-9), (
            "opt_kto.exp_avg_sq diverged from kto's own grad² — state bleed"
        )
        # And cross-check: sft's state is not kto's gradient signature.
        assert not torch.allclose(sft_exp_avg_sq, 1e-3 * g_k.pow(2), atol=1e-9)


def test_delta_norm_ratio_shared_vs_per_objective_in_band(monkeypatch):
    """After the same seeded gradient, the per-objective path and the
    shared path produce weight deltas of the same order. The regression
    this catches: a misplumbed flag that produces zero updates (opt not
    stepping) or runaway updates (opt reconstructed every call, losing
    Adam bias correction).

    bitsandbytes is forcibly unavailable here — AdamW8bit only steps on
    GPU tensors, so exercising it on CPU would fail in bnb internals
    rather than in this test. The shared path falls back to
    ``torch.optim.AdamW``, matching the per-objective path; ratio should
    be ≈ 1.0.
    """
    # Force the shared path into the torch.AdamW fallback.
    import builtins
    real_import = builtins.__import__

    def no_bnb(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("forced in test — keep shared path on torch.AdamW")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_bnb)

    shared = _fresh_engine(per_objective=False)
    per_obj = _fresh_engine(per_objective=True)

    d_shared = _run_one_step(shared, "sft", seed=0)
    d_per = _run_one_step(per_obj, "sft", seed=0)

    n_shared = float(d_shared.norm())
    n_per = float(d_per.norm())

    assert n_shared > 0, "shared path produced zero update — opt did not step"
    assert n_per > 0, "per-objective path produced zero update — opt did not step"
    ratio = n_per / n_shared
    # Both on torch.AdamW with identical hyperparams → ratio should be ~1.0.
    assert 0.5 < ratio < 2.0, (
        f"||Δθ|| ratio {ratio:.3f} outside [0.5, 2.0] — "
        "per-objective path diverges from shared path by more than 2x"
    )
