"""Live model state — the single mutable container the daemon owns.

Implements the §3.1 mental model:

    live_model = base_weights ⊕ merged_deltas ⊕ active_lora_adapter

* ``base_weights`` are loaded NF4 (or bf16 for full-FT) and *never* mutated.
* ``merged_deltas`` is a dict ``{layer_name → bf16 tensor}`` of size matching
  each LoRA-targeted base linear, holding the cumulative effect of all prior
  merged adapters. Allocated lazily on first merge; applied at forward time as
  an additive residual via a hook (see :mod:`lile.adapters`).
* ``active_lora_adapter`` is a standard PEFT LoRA wrapping the targeted layers.

The state object is the single owner of the model. All training/inference goes
through it, serialised by :class:`lile.controller.Controller`.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

# Lazy import of unsloth so importing lile doesn't trigger CUDA init in tests
# that don't need it.
_FastLanguageModel = None


def _get_fast_lm():
    global _FastLanguageModel
    if _FastLanguageModel is None:
        from unsloth import FastLanguageModel  # noqa: PLC0415

        _FastLanguageModel = FastLanguageModel
    return _FastLanguageModel


# Default LoRA targets — Unsloth's standard set for transformer attn + MLP.
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class StateConfig:
    model_name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    full_finetuning: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = 3407
    # Inference backend selection — see lile/engine/vllm_sidecar.py for the
    # contract of "vllm_sidecar". Default keeps single-process behaviour.
    inference_backend: str = "fast_generate"  # "fast_generate" | "vllm_sidecar"
    sidecar_mode: str = "colocate"  # "colocate" (1 GPU) | "separate" (≥2 GPUs)
    sidecar_device: str = "cuda:1"
    sidecar_gpu_memory_utilization: float = 0.4


class LiveState:
    """The single mutable model state container.

    Construction is two-phase: ``__init__`` records config; ``load()`` actually
    pulls weights and applies the LoRA. This separation lets tests instantiate a
    config without touching CUDA.
    """

    def __init__(self, config: StateConfig):
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        # The merged_deltas store. Lives on the same device as the model after
        # first merge. None means "no merges yet"; we don't allocate until needed.
        self._merged_deltas: dict[str, torch.Tensor] = {}
        # Hook handles for the residual-delta forward path (so we can detach on
        # destroy / save).
        self._delta_hooks: list[Any] = []
        # Bookkeeping for the active LoRA: how many merges have folded into
        # merged_deltas, plus the schema version for snapshot compatibility.
        self.merge_count: int = 0
        self.snapshot_schema = 1
        # Re-entrant write lock: training-step / merge / save are mutually
        # exclusive but inference-side reads of metadata are fine concurrently.
        self.write_lock = threading.RLock()

    # --- Lifecycle ---------------------------------------------------------

    def load(self) -> "LiveState":
        FLM = _get_fast_lm()
        kwargs = dict(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            full_finetuning=self.config.full_finetuning,
            dtype=None,
        )
        self.model, self.tokenizer = FLM.from_pretrained(**kwargs)
        if not self.config.full_finetuning:
            self.model = FLM.get_peft_model(
                self.model,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_targets),
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.random_state,
            )
        # Pad token: many Qwen models lack one; align with eos so batched train
        # collators don't choke.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self

    def load_frozen_ref(self) -> Any:
        """Load a second base-only model copy as a frozen reference.

        Used by KL anchor / KTO / CCPD for π_ref so the reference doesn't drift
        with training. Returns the model; caller (Controller) holds the handle.
        VRAM cost: one extra base-model copy (~0.4 GB on 0.6 B 4-bit, ~5 GB on
        7 B 4-bit). No PEFT wrap, eval mode, all params frozen.
        """
        FLM = _get_fast_lm()
        ref_model, _ = FLM.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            full_finetuning=False,
            dtype=None,
        )
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        return ref_model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # --- Mode switching ---------------------------------------------------

    def set_inference_mode(self) -> None:
        FLM = _get_fast_lm()
        FLM.for_inference(self.model)
        self.model.eval()

    def set_training_mode(self) -> None:
        FLM = _get_fast_lm()
        FLM.for_training(self.model)
        self.model.train()

    # --- VRAM ---------------------------------------------------------------

    def vram_summary(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "peak_gb": 0.0, "free_gb": 0.0}
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "free_gb": free / 1024**3,
            "total_gb": total / 1024**3,
        }

    # --- merged_deltas accessors ------------------------------------------

    @property
    def merged_deltas(self) -> dict[str, torch.Tensor]:
        return self._merged_deltas

    def has_merged_deltas(self) -> bool:
        return bool(self._merged_deltas)
