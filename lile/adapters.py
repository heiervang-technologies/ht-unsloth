"""Progressive merge & LoRA-pool management.

The single load-bearing piece here is :func:`merge_active_lora_into_deltas`,
which implements the §6 dequant→fp32-merge→bf16-store sequence on a
4-bit-base + LoRA model without ever requantising back to NF4 (which would
silently corrupt quality).

The merge sequence per LoRA-targeted layer:

1. ``W = fast_dequantize(layer.base_layer.weight, quant_state)``  →  fp32 base.
2. ``W += scale * (B @ A)`` in fp32 via in-place ``addmm_``.
3. ``delta = (W − dequant(layer.base_layer.weight)).to(bfloat16)`` — the *new*
   contribution from this LoRA. (We subtract the dequantised base so the delta
   we store is purely the LoRA contribution; merged_deltas accumulates LoRA
   contributions across many merges.)
4. ``self._merged_deltas[name] += delta`` (allocate to zeros if absent).
5. Reset LoRA A to N(0, σ_A) and B to zero — i.e. re-initialise the active
   adapter so subsequent training continues from the merged state.

Then a forward post-hook on each merged layer adds ``x @ delta.T`` so the
residual is reflected in inference. The hook is the §6 "residual at forward
time" path. For untargeted layers (embedding, lm_head), no delta is ever
allocated — they're inherently 16-bit.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any, Iterable

import torch
from torch import nn


# --- Layer discovery ---------------------------------------------------

def _is_lora_layer(layer: Any) -> bool:
    """A PEFT LoRA-wrapped 4-bit (or 16-bit) Linear has ``base_layer`` and
    ``lora_A`` / ``lora_B`` attributes."""
    return (
        hasattr(layer, "base_layer")
        and hasattr(layer, "lora_A")
        and hasattr(layer, "lora_B")
    )


def iter_lora_layers(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(qualified_name, layer)`` for each PEFT LoRA-wrapped layer."""
    for name, mod in model.named_modules():
        if _is_lora_layer(mod):
            yield name, mod


# --- Merge primitive ---------------------------------------------------

def _dequant_base(layer: Any) -> torch.Tensor:
    """Dequantise the base weight of a PEFT-wrapped (4-bit or 16-bit) Linear.

    Returns a fp32 tensor of shape ``(out_features, in_features)``. Mirrors the
    contract of :func:`unsloth.kernels.fast_dequantize` for the 4-bit path; for
    16-bit base layers it just returns ``base_layer.weight.float()``.
    """
    base = layer.base_layer
    weight = base.weight
    quant_state = getattr(base, "weight", None)
    quant_state = getattr(quant_state, "quant_state", None) or getattr(
        base, "quant_state", None
    )
    if quant_state is not None:
        from unsloth.kernels import fast_dequantize  # noqa: PLC0415

        W = fast_dequantize(weight, quant_state)
        return W.to(torch.float32)
    return weight.detach().to(torch.float32)


def _lora_BA(layer: Any) -> torch.Tensor:
    """Compute ``scale * B @ A`` in fp32 for the active adapter of ``layer``.

    Returns a tensor of shape ``(out_features, in_features)``. If multiple
    adapter names are active, sums their contributions; in practice we only have
    one ("default").
    """
    out_features = layer.out_features
    in_features = layer.in_features
    device = layer.base_layer.weight.device
    delta = torch.zeros(
        out_features, in_features, dtype=torch.float32, device=device
    )
    # PEFT structures: lora_A[name] is a Linear; lora_B[name] is a Linear; scaling[name] is float.
    for name, A_mod in layer.lora_A.items():
        if name not in layer.lora_B:
            continue
        B_mod = layer.lora_B[name]
        scale = float(layer.scaling[name])
        A = A_mod.weight  # [r, in_features]
        B = B_mod.weight  # [out_features, r]
        # Result: out_features × in_features
        delta = delta.addmm_(B.to(torch.float32), A.to(torch.float32), alpha=scale)
    return delta


def _reset_lora(layer: Any, generator: torch.Generator | None = None) -> None:
    """Re-initialise the active LoRA parameters to fresh starting values."""
    for name, A_mod in layer.lora_A.items():
        nn.init.kaiming_uniform_(A_mod.weight, a=math.sqrt(5), generator=generator)
        # PEFT's standard init: A ~ Kaiming, B = zeros.
    for name, B_mod in layer.lora_B.items():
        nn.init.zeros_(B_mod.weight)


# --- Residual delta forward hook --------------------------------------

class ResidualDeltaHook:
    """A forward post-hook that adds ``x @ delta.T`` to the layer's output.

    The hook sees the post-LoRA output and adds the merged-deltas residual on
    top. ``delta`` is bf16; we cast input and output to match the layer's dtype.
    Held by :class:`AdapterManager` so it can detach the hook on save / shutdown.
    """

    __slots__ = ("delta", "_handle")

    def __init__(self, delta: torch.Tensor):
        self.delta = delta  # bf16 [out, in], on GPU
        self._handle: Any = None

    def __call__(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        # x: [..., in_features]; delta: [out_features, in_features]
        # Cast to bf16 for the residual mul and back.
        residual = torch.nn.functional.linear(x.to(self.delta.dtype), self.delta)
        return output + residual.to(output.dtype)


# --- AdapterManager ----------------------------------------------------

class AdapterManager:
    """Owns the merge state for a :class:`lile.state.LiveState`.

    Responsibilities:
        * Track merged_deltas dict; allocate lazily on first merge.
        * Run the §6 merge procedure.
        * Install / uninstall residual forward hooks.
        * Provide a state_dict for snapshot save/restore.
    """

    def __init__(self, model: nn.Module, merged_deltas: dict[str, torch.Tensor]):
        self.model = model
        self.merged_deltas = merged_deltas
        self._hooks: dict[str, ResidualDeltaHook] = {}
        self._lock = threading.Lock()

    # --- Hooks -------------------------------------------------------------

    def install_hooks(self) -> None:
        """(Re)install residual forward hooks for every layer in merged_deltas."""
        with self._lock:
            self._uninstall_unlocked()
            for name, layer in iter_lora_layers(self.model):
                if name not in self.merged_deltas:
                    continue
                delta = self.merged_deltas[name]
                hook = ResidualDeltaHook(delta)
                handle = layer.register_forward_hook(hook)
                hook._handle = handle
                self._hooks[name] = hook

    def uninstall_hooks(self) -> None:
        with self._lock:
            self._uninstall_unlocked()

    def _uninstall_unlocked(self) -> None:
        for hook in self._hooks.values():
            if hook._handle is not None:
                hook._handle.remove()
        self._hooks.clear()

    # --- Merge -------------------------------------------------------------

    @torch.no_grad()
    def merge_active_lora(self) -> dict[str, dict[str, float]]:
        """Fold the current active LoRA into ``merged_deltas`` and reset LoRA.

        Returns a per-layer summary dict useful for logging / tests.
        """
        summary: dict[str, dict[str, float]] = {}
        with self._lock:
            for name, layer in iter_lora_layers(self.model):
                # The contribution this merge adds, in fp32 then bf16.
                BA_fp32 = _lora_BA(layer)
                delta_bf16 = BA_fp32.to(torch.bfloat16)

                if name not in self.merged_deltas:
                    self.merged_deltas[name] = torch.zeros_like(delta_bf16)
                # Accumulate. We do the accumulation in fp32 then cast back to
                # bf16 to bound rounding error growth.
                acc = self.merged_deltas[name].to(torch.float32) + BA_fp32
                self.merged_deltas[name] = acc.to(torch.bfloat16)

                # Update / install hook for this layer if not yet installed.
                if name not in self._hooks:
                    hook = ResidualDeltaHook(self.merged_deltas[name])
                    handle = layer.register_forward_hook(hook)
                    hook._handle = handle
                    self._hooks[name] = hook
                else:
                    # Replace tensor reference in existing hook (in case we
                    # re-allocated above via .to(bf16) which returns a new tensor).
                    self._hooks[name].delta = self.merged_deltas[name]

                # Reset the LoRA so subsequent training is fresh.
                _reset_lora(layer)

                summary[name] = {
                    "abs_max": float(BA_fp32.abs().max().item()),
                    "frob_norm": float(BA_fp32.norm().item()),
                }
        return summary

    # --- Snapshot -------------------------------------------------------

    def deltas_state_dict(self) -> dict[str, torch.Tensor]:
        """Return CPU bf16 copies of merged_deltas for safetensors save."""
        with self._lock:
            return {name: t.detach().cpu().clone() for name, t in self.merged_deltas.items()}

    def load_deltas_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Replace merged_deltas with ``sd`` (moved onto the model's device)."""
        device = next(self.model.parameters()).device
        with self._lock:
            self.merged_deltas.clear()
            for name, t in sd.items():
                self.merged_deltas[name] = t.to(device=device, dtype=torch.bfloat16)
        # Reinstall hooks against the new tensors.
        self.install_hooks()


# --- Helpers for LoRA save/restore (separate from merge) ---------------

def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the LoRA parameters from a PEFT-wrapped model."""
    sd: dict[str, torch.Tensor] = {}
    rx = re.compile(r"\.lora_[AB]\.")
    for k, v in model.state_dict().items():
        if rx.search(k):
            sd[k] = v.detach().cpu().clone()
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]) -> int:
    """Load LoRA params into ``model``. Returns count of loaded tensors."""
    own_sd = model.state_dict()
    n = 0
    for k, v in sd.items():
        if k not in own_sd:
            continue
        own_sd[k].copy_(v.to(own_sd[k].device, dtype=own_sd[k].dtype))
        n += 1
    return n
