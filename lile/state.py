"""Live model state: base_weights ⊕ merged_deltas ⊕ active_lora_adapter.

The invariant from LIVELEARN §3.1/§6:
  * base is NF4, never modified.
  * merged_deltas is a bf16 residual held in CPU RAM between merges, applied at
    load time by adding into a dequantized copy of the base (never requantized
    back to NF4 — that's the 4-bit merge gotcha from §6).
  * active_lora_adapter is hot in GPU; gradients flow into it.

For this implementation we rely on Unsloth's FastLanguageModel + PEFT LoRA. The
merged_deltas residual is materialized by Unsloth's existing dequant→fp32→merge
path when `merge()` is called; we keep our own bf16 copy so we can repeat the
merge or restore it on load.

Live residual application
-------------------------
Unsloth's fast path (Qwen3 attention + MLP) bypasses both ``LoraLayer.forward``
and ``base_layer.forward``. A naive forward-hook on either never fires, so a
merged residual would sit inert on CPU. We close that gap by monkey-patching
``unsloth.kernels.utils.matmul_lora`` — the single low-level kernel every
QKV/O/gate/up/down call funnels into. The patch:

  * reads a sidecar attribute ``W._residual_delta`` (a bf16 GPU tensor),
  * adds ``F.linear(X, delta)`` to the kernel's output,
  * is a no-op when the attribute is absent.

Attaching the delta to the Parameter itself (rather than Gemini's
``id(W) -> delta`` global dict) keeps the binding local to the layer and
survives Unsloth's ``for_training``/``for_inference`` mode flips (``id(W)``
stays stable across those flips, empirically verified on Qwen3).

A fallback ``forward_pre_hook`` is also registered on ``base_layer`` so the
standard PEFT path (used under ``model.disable_adapter()`` for the KL anchor)
still applies the residual.
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- matmul_lora patch
# Module-level, installed once on import. Idempotent via the sentinel attribute.
def _install_matmul_lora_patch() -> None:
    import unsloth.kernels.utils as _uutils  # heavy but already paid for by Unsloth

    original = _uutils.matmul_lora
    if getattr(original, "_lile_patched", False):
        return

    def _patched(X, W, W_quant, A, B, s, out=None):
        out_res = original(X, W, W_quant, A, B, s, out=out)
        delta = getattr(W, "_residual_delta", None)
        if delta is None:
            return out_res
        # Residual is a (out_features, in_features) bf16 GPU tensor, no grad.
        # F.linear(X, delta) = X @ delta.T; shape matches out_res exactly since
        # matmul_lora's final view(...) mirrors F.linear's broadcasting.
        delta_cast = delta.to(dtype=out_res.dtype, device=out_res.device, non_blocking=True)
        return out_res + F.linear(X, delta_cast)

    _patched._lile_patched = True  # type: ignore[attr-defined]
    _patched._lile_original = original  # type: ignore[attr-defined]

    # Replace every binding across the unsloth package. Unsloth re-exports
    # matmul_lora in several submodules (kernels.utils, kernels.fast_lora, …)
    # and pre-bound references in those modules must also flip — otherwise the
    # fast path in fast_lora still sees the unpatched callable.
    replaced = 0
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("unsloth"):
            continue
        try:
            members = list(vars(mod).items())
        except Exception:
            continue
        for attr, val in members:
            if val is original:
                setattr(mod, attr, _patched)
                replaced += 1
    log.debug("installed matmul_lora residual patch (%d bindings rewritten)", replaced)


# Install at import — matches Gemini's ergonomics but with explicit idempotence.
_install_matmul_lora_patch()


@dataclass
class ModelState:
    """Owns the live model + tokenizer + LoRA config."""
    model: Any                      # FastLanguageModel-wrapped PEFT model
    tokenizer: Any
    base_model_name: str
    lora_rank: int
    lora_alpha: int
    # Bookkeeping of what has been merged so far in this process.
    merges_applied: int = 0
    # CPU bf16 residual: name -> delta tensor from all prior merges.
    merged_deltas: dict[str, torch.Tensor] = field(default_factory=dict)
    # Serializes train/infer mode flips. Unsloth's FastLanguageModel.for_inference
    # and .for_training toggle per-layer temp buffers (temp_QA/temp_O/…); if a
    # train step runs for_training() while a concurrent chat is mid-generate on
    # the inference fast path, the buffers vanish mid-call. Everything that
    # calls .generate() or .backward() must hold this lock across the mode flip
    # AND the work that depends on those buffers.
    mode_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    # Base-layer forward pre-hook handles (fallback path for disable_adapter).
    # Keyed by LoraLayer name; stored so we can remove/rebind without leaking.
    _residual_hook_handles: dict[str, Any] = field(default_factory=dict, repr=False)
    # Optional frozen reference model (base-only, eval, requires_grad=False).
    # When set, objectives consume it as ``pi_ref`` for KL anchoring instead
    # of falling back to ``model.disable_adapter()`` on the live model.
    frozen_ref: Any = field(default=None, repr=False)

    @classmethod
    def load(
        cls,
        model_name: str,
        max_seq_length: int = 2048,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        load_in_4bit: bool = True,
    ) -> "ModelState":
        from unsloth import FastLanguageModel  # heavy import; lazy

        log.info("loading base model %s (4bit=%s)", model_name, load_in_4bit)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        # Qwen3 series: target the standard LoRA attention+MLP set.
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            base_model_name=model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    # ------------------------------------------------------------------ adapter ops
    def reset_active_adapter(self) -> None:
        """Zero the active LoRA matrices. Keeps rank+target intact."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    p.zero_()
        log.info("active LoRA adapter zeroed")

    def extract_active_adapter(self) -> dict[str, torch.Tensor]:
        """Snapshot the LoRA A/B matrices only (the trainable part)."""
        out: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                out[name] = p.detach().to("cpu", dtype=torch.bfloat16).clone()
        return out

    def load_active_adapter(self, sd: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            named = dict(self.model.named_parameters())
            for name, t in sd.items():
                if name in named:
                    named[name].data.copy_(t.to(named[name].dtype).to(named[name].device))

    # ------------------------------------------------------------------ merge
    def merge_active_into_residual(self) -> None:
        """Bake the current LoRA into `merged_deltas` (CPU bf16 residual).

        We compute Δ = α/r · (B @ A) per LoRA site, accumulate into
        `merged_deltas`, then zero the active adapter. The *actual* model
        weights are not touched — this keeps us NF4-base-stable per §6.
        On next `ModelState.load` (with merged_deltas present) or before a
        forward pass, callers can apply the residual via `apply_residual()`.

        Ops keeps it simple: the residual is the canonical record of what has
        been trained; the in-GPU weights are base + residual + active. After
        a merge, active is zero, so base + residual == live model.
        """
        scale = self.lora_alpha / self.lora_rank
        new_deltas: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                # PEFT LoraLayer exposes lora_A / lora_B as ModuleDicts.
                if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                    continue
                if "default" not in module.lora_A:
                    continue
                A = module.lora_A["default"].weight  # (r, in)
                B = module.lora_B["default"].weight  # (out, r)
                if A.abs().sum() == 0 and B.abs().sum() == 0:
                    continue
                delta = (B @ A).to(torch.float32) * scale            # (out, in)
                key = f"{name}.weight"
                prev = self.merged_deltas.get(key)
                if prev is not None:
                    delta = delta + prev.to(delta.device, dtype=torch.float32)
                new_deltas[key] = delta.to("cpu", dtype=torch.bfloat16)
        self.merged_deltas.update(new_deltas)
        self.merges_applied += 1
        self.reset_active_adapter()
        self._apply_residual_to_model()
        log.info(
            "merged %d LoRA sites; total merges=%d",
            len(new_deltas), self.merges_applied,
        )

    # ------------------------------------------------------------------ residual → live
    def _apply_residual_to_model(self) -> None:
        """Bind the CPU bf16 residual onto each live LoRA site for forward-time use.

        Two channels:
          (a) ``base_layer.weight._residual_delta = delta_gpu`` — picked up by
              the patched matmul_lora kernel on every QKV/MLP forward under
              Unsloth's fast path.
          (b) ``forward_pre_hook`` on ``base_layer`` that adds
              ``F.linear(x, delta)`` to the layer's output. Fires under the
              standard PEFT path (e.g. during ``model.disable_adapter()`` used
              by the KL anchor in CCPD v2).

        Both channels reference the same delta tensor; if one fires, the other
        doesn't see the layer at all. Under Unsloth fast path, channel (a) fires
        and the hook stays idle. Under PEFT-standard, the hook fires and
        matmul_lora isn't called.
        """
        if not self.merged_deltas:
            return
        device = next(self.model.parameters()).device
        applied = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if "default" not in module.lora_A:
                continue
            key = f"{name}.weight"
            delta_cpu = self.merged_deltas.get(key)
            if delta_cpu is None:
                continue
            delta_gpu = delta_cpu.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            base_layer = getattr(module, "base_layer", None)
            target_w = base_layer.weight if base_layer is not None else None
            if target_w is None:
                continue
            # Channel (a): attach to the Parameter directly.
            target_w._residual_delta = delta_gpu  # type: ignore[attr-defined]
            # Channel (b): register forward hook (dedup via stored handle).
            old = self._residual_hook_handles.pop(name, None)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
            if base_layer is not None:
                handle = base_layer.register_forward_hook(
                    _make_residual_forward_hook(delta_gpu)
                )
                self._residual_hook_handles[name] = handle
            applied += 1
        log.info("applied residual to %d LoRA sites (matmul_lora + forward_hook)", applied)

    def residual_fingerprint(self) -> str:
        """Deterministic hash of the CPU residual — used by snapshot round-trip tests."""
        import hashlib
        h = hashlib.sha256()
        for k in sorted(self.merged_deltas.keys()):
            t = self.merged_deltas[k]
            h.update(k.encode("utf-8"))
            h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------ frozen ref
    def load_frozen_ref(self) -> None:
        """Load a second base-only model as the frozen reference for KL anchors.

        The live ``self.model`` carries an active LoRA adapter plus any applied
        residual. A frozen reference held in eval mode with grad disabled gives
        objectives a *stable* anchor point — the session-start policy — rather
        than the EMA-1 fallback (``model.disable_adapter()``) which anchors to
        the live merged-deltas state.

        Memory cost: the 4-bit NF4 base weighs roughly 0.4 GB for a 0.6B model
        and ~5 GB for a 7B model. Call from ``Controller.start`` only when
        ``cfg.frozen_ref=True`` so the VRAM cost is opt-in.
        """
        from unsloth import FastLanguageModel
        if self.frozen_ref is not None:
            log.info("frozen reference already loaded — skipping")
            return
        log.info("loading frozen reference (base-only) from %s", self.base_model_name)
        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        # Put into inference mode so the Unsloth fast path works.
        try:
            FastLanguageModel.for_inference(ref_model)
        except Exception:
            pass
        self.frozen_ref = ref_model
        log.info("frozen reference loaded (eval, requires_grad=False)")

    def save_residual(self, path: Path) -> None:
        from safetensors.torch import save_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.merged_deltas:
            # safetensors refuses empty; write a sentinel.
            path.with_suffix(".empty").touch()
            if path.exists():
                path.unlink()
            return
        save_file({k: v.contiguous() for k, v in self.merged_deltas.items()}, str(path))

    def load_residual(self, path: Path) -> None:
        from safetensors.torch import load_file
        if not path.exists():
            self.merged_deltas = {}
            return
        self.merged_deltas = load_file(str(path))
        self._apply_residual_to_model()


def _make_residual_forward_hook(delta_gpu: torch.Tensor):
    """Closure-scoped forward hook adding ``F.linear(x, delta)`` to base_layer output."""
    def _hook(module: Any, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        return output + F.linear(x, delta_gpu.to(dtype=output.dtype, device=output.device))
    return _hook
