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
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


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
        log.info(
            "merged %d LoRA sites; total merges=%d",
            len(new_deltas), self.merges_applied,
        )

    def residual_fingerprint(self) -> str:
        """Deterministic hash of the CPU residual — used by snapshot round-trip tests."""
        import hashlib
        h = hashlib.sha256()
        for k in sorted(self.merged_deltas.keys()):
            t = self.merged_deltas[k]
            h.update(k.encode("utf-8"))
            h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

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
