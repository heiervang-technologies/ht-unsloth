"""Tests for :mod:`lile.adapters` — the §6 progressive merge.

We use a *synthetic* PEFT-shaped layer (minimum viable surface) to keep these
tests CPU-only and fast. The same code path is exercised in the smoke test
against the real Qwen3 4-bit model.

The load-bearing claim under test: after ``merge_active_lora`` returns, the
forward output of ``base_layer + active_LoRA + residual_hook`` (with active
LoRA reset to zero contribution) is numerically equal to the pre-merge output
of ``base_layer + active_LoRA`` (within bf16 quantisation noise of the delta).

If this fails, ``snapshot save→load→generate`` will silently corrupt outputs.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from lile.adapters import (
    AdapterManager,
    iter_lora_layers,
    lora_state_dict,
    load_lora_state_dict,
)


# --- Synthetic PEFT-shaped Linear ----------------------------------------


class _SyntheticLoRALinear(nn.Module):
    """Mirrors the surface of a PEFT-wrapped LoRA Linear that the merge code
    inspects: ``base_layer.weight``, ``lora_A``, ``lora_B``, ``scaling``,
    ``in_features``, ``out_features``."""

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        # Freeze the base — this matches PEFT semantics.
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.ModuleDict({
            "default": nn.Linear(in_features, r, bias=False),
        })
        self.lora_B = nn.ModuleDict({
            "default": nn.Linear(r, out_features, bias=False),
        })
        self.scaling = {"default": float(alpha) / float(r)}
        # PEFT init: A ~ Kaiming, B = zeros. We deliberately set B nonzero so
        # the merge has something to fold.
        nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))
        nn.init.normal_(self.lora_B["default"].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.base_layer(x)
        a = self.lora_A["default"](x)
        b = self.lora_B["default"](a)
        return out + self.scaling["default"] * b


class _Wrapper(nn.Module):
    """A ``model``-shaped container so ``iter_lora_layers`` can find the layer."""

    def __init__(self):
        super().__init__()
        self.layer1 = _SyntheticLoRALinear(8, 16)
        self.layer2 = _SyntheticLoRALinear(16, 32)


def _seed_inputs(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 8, generator=g)


def test_iter_lora_layers_finds_synthetic_layers():
    model = _Wrapper()
    found = dict(iter_lora_layers(model))
    assert set(found) == {"layer1", "layer2"}


def test_merge_then_reset_preserves_output_via_residual():
    """The headline correctness check: after merging, the layer's output must
    match the pre-merge output (down to bf16 round-trip), even though the
    LoRA itself has been reset to zero contribution."""
    torch.manual_seed(0)
    model = _Wrapper()
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(model, deltas)

    x = _seed_inputs()
    h1 = model.layer1(x)  # pre-merge output (base + lora)

    # Merge folds the active LoRA into deltas and resets LoRA to fresh init.
    summary = mgr.merge_active_lora()
    assert "layer1" in summary and "layer2" in summary

    # After merge: base_layer is unchanged; LoRA was reset (so its contribution
    # is now ~0 because B was zero'd by _reset_lora). The residual hook should
    # add back exactly what we just folded in.
    assert deltas["layer1"].dtype == torch.bfloat16
    assert deltas["layer1"].shape == (16, 8)

    h2 = model.layer1(x)  # post-merge output (base + reset_lora + residual_hook)

    # Tolerance: bf16 has ~7 bits of mantissa, so the merged delta loses
    # precision relative to the fp32 BA product. The hook re-applies it in bf16
    # so the round-trip error is bounded by bf16(BA) − fp32(BA) ≈ 2^-7 ‖BA‖.
    rel_err = (h1 - h2).abs().max() / h1.abs().max().clamp_min(1e-6)
    assert rel_err < 5e-2, f"merge round-trip too lossy: rel_err={rel_err.item()}"


def test_merge_accumulates_across_two_merges():
    """Two consecutive merges must accumulate, not overwrite."""
    torch.manual_seed(0)
    model = _Wrapper()
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(model, deltas)

    # Merge #1
    mgr.merge_active_lora()
    delta1 = deltas["layer1"].detach().clone()

    # Re-randomise lora_B so the next merge has new contribution.
    nn.init.normal_(model.layer1.lora_B["default"].weight, std=0.02)
    nn.init.normal_(model.layer2.lora_B["default"].weight, std=0.02)

    # Merge #2
    mgr.merge_active_lora()
    delta2 = deltas["layer1"]

    # delta2 must be != delta1 (accumulation happened).
    assert not torch.allclose(delta1, delta2), "second merge must accumulate"


def test_lora_state_dict_round_trip():
    torch.manual_seed(0)
    src = _Wrapper()
    dst = _Wrapper()  # different random init for both base + LoRA
    # Match the (frozen) base so the only variable is LoRA — what we're testing.
    with torch.no_grad():
        dst.layer1.base_layer.weight.copy_(src.layer1.base_layer.weight)
        dst.layer2.base_layer.weight.copy_(src.layer2.base_layer.weight)

    sd = lora_state_dict(src)
    n = load_lora_state_dict(dst, sd)
    assert n == len(sd)

    x = _seed_inputs()
    assert torch.allclose(src.layer1(x), dst.layer1(x), atol=1e-6)


def test_uninstall_hooks_removes_residual():
    """After uninstall_hooks, the model's output drops the residual term."""
    torch.manual_seed(0)
    model = _Wrapper()
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(model, deltas)
    mgr.merge_active_lora()

    x = _seed_inputs()
    out_with_hook = model.layer1(x)
    mgr.uninstall_hooks()
    out_without_hook = model.layer1(x)

    # With the hook gone and LoRA reset (B=0), the layer's output is just the
    # base — which is *not* equal to the pre-merge output.
    assert not torch.allclose(out_with_hook, out_without_hook)


def test_load_deltas_reinstalls_hooks():
    """A fresh AdapterManager loading saved deltas must reinstall the residual
    hooks so the layer's output matches the source's post-merge output."""
    torch.manual_seed(0)
    src = _Wrapper()
    src_deltas: dict[str, torch.Tensor] = {}
    src_mgr = AdapterManager(src, src_deltas)
    src_mgr.merge_active_lora()
    # Zero the LoRA for a clean comparison (B is already zero post-reset).
    with torch.no_grad():
        for layer in (src.layer1, src.layer2):
            layer.lora_B["default"].weight.zero_()

    x = _seed_inputs()
    src_after = src.layer1(x)

    # Build a fresh model with the *same* base weights but no deltas yet.
    dst = _Wrapper()
    with torch.no_grad():
        dst.layer1.base_layer.weight.copy_(src.layer1.base_layer.weight)
        dst.layer2.base_layer.weight.copy_(src.layer2.base_layer.weight)
        for layer in (dst.layer1, dst.layer2):
            layer.lora_B["default"].weight.zero_()
    dst_deltas: dict[str, torch.Tensor] = {}
    dst_mgr = AdapterManager(dst, dst_deltas)

    # Restore the deltas — load_deltas_state_dict must reinstall the hooks.
    dst_mgr.load_deltas_state_dict({k: v.cpu() for k, v in src_deltas.items()})
    dst_after = dst.layer1(x)

    assert torch.allclose(src_after, dst_after, atol=1e-3), (
        "deltas restore must reproduce the residual contribution"
    )
