"""Training engine — composes per-sample + per-batch objectives, takes a step.

Composes objectives via the registry in :mod:`lile.objectives`. Each sample
declares its own objective list (per §3.3); the batch can additionally specify
batch-level objectives (e.g. ``kl_anchor``).

A single ``step(batch)`` call:

1. Sets the model to training mode.
2. For each ``Sample``:
   - Validates the requested objectives.
   - Computes each per-sample loss, weighted, summed.
3. Computes batch-level losses (KL anchor etc.) and adds them with their weight.
4. ``loss.backward()``, optimizer ``step()``, ``zero_grad()``.
5. Returns a ``StepResult`` describing what happened (loss components, time,
   grad-norm, vram).

Crucially the engine does *not* know about the queue or the snapshot manager.
That separation makes it easy to test in isolation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.optim import Optimizer

from lile import objectives as O
from lile.objectives import Batch, Sample


@dataclass
class StepResult:
    loss: float
    components: dict[str, float]
    grad_norm: float
    elapsed_s: float
    n_samples: int
    skipped_samples: int = 0
    notes: list[str] = field(default_factory=list)
    vram_peak_gb: float = 0.0


class TrainEngine:
    """Single-step trainer using the objective composer."""

    def __init__(self, state, *, lr: float = 1e-5, weight_decay: float = 0.0):
        self.state = state
        # Only LoRA params are trainable — confirm by filtering.
        trainable = [p for p in state.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "No trainable parameters found in model — did you call "
                "FastLanguageModel.get_peft_model?"
            )
        self.optimizer: Optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay
        )
        self.lr = lr
        self.global_step = 0

    def set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.lr = lr

    def step(self, batch: Batch, *, ref_model=None) -> StepResult:
        if not batch.samples:
            return StepResult(
                loss=0.0, components={}, grad_norm=0.0, elapsed_s=0.0, n_samples=0,
                notes=["empty batch"],
            )
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        self.state.set_training_mode()

        components: dict[str, float] = {}
        notes: list[str] = []
        skipped = 0

        # Defensive: zero grads up front.
        self.optimizer.zero_grad(set_to_none=True)

        # 1. Per-sample losses.
        per_sample_losses: list[torch.Tensor] = []
        for s in batch.samples:
            obj_specs = s.objectives or [{"sft": {}}]  # default
            sample_loss = None
            for obj_dict in obj_specs:
                if len(obj_dict) != 1:
                    raise ValueError(f"Each objective spec must have one key; got {obj_dict}")
                ((obj_name, obj_kwargs),) = obj_dict.items()
                O.validate_sample(s, obj_name)
                spec = O.get(obj_name)
                kwargs = dict(obj_kwargs or {})
                if "ref_model" in spec.fn.__code__.co_varnames:
                    kwargs.setdefault("ref_model", ref_model)
                try:
                    loss_tensor = spec.fn(self.state.model, self.state.tokenizer, s, **kwargs)
                except Exception as e:  # surface, but don't poison the whole batch
                    skipped += 1
                    notes.append(f"sample skipped ({obj_name}): {e!r}")
                    continue
                # Skip samples where the objective produced a 0-tensor as a
                # graceful "no signal here" signal (e.g. CCPD spread < tau).
                if loss_tensor is None:
                    skipped += 1
                    continue
                # If sample_loss already accumulated, sum.
                if sample_loss is None:
                    sample_loss = loss_tensor
                else:
                    sample_loss = sample_loss + loss_tensor
                components[obj_name] = components.get(obj_name, 0.0) + float(loss_tensor.detach().item())
            if sample_loss is not None:
                per_sample_losses.append(sample_loss)

        if not per_sample_losses:
            elapsed = time.time() - t0
            return StepResult(
                loss=0.0, components=components, grad_norm=0.0,
                elapsed_s=elapsed, n_samples=len(batch.samples),
                skipped_samples=skipped, notes=notes + ["no live losses"],
            )

        per_sample_total = torch.stack(per_sample_losses).mean()

        # 2. Batch-level losses.
        batch_total = per_sample_total
        for obj_dict in batch.batch_objectives:
            if len(obj_dict) != 1:
                raise ValueError(f"Each objective spec must have one key; got {obj_dict}")
            ((obj_name, obj_kwargs),) = obj_dict.items()
            O.validate_batch_objective(obj_name)
            spec = O.get(obj_name)
            kwargs = dict(obj_kwargs or {})
            if "ref_model" in spec.fn.__code__.co_varnames:
                kwargs.setdefault("ref_model", ref_model)
            loss_tensor = spec.fn(self.state.model, self.state.tokenizer, batch, **kwargs)
            if loss_tensor is None:
                continue
            components[obj_name] = components.get(obj_name, 0.0) + float(loss_tensor.detach().item())
            if loss_tensor.requires_grad:
                batch_total = batch_total + loss_tensor

        # 3. Backward + optimizer step.
        batch_total.backward()
        # Grad clip — prevents one bad batch from melting the model.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.state.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

        elapsed = time.time() - t0
        peak_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() else 0.0
        )
        return StepResult(
            loss=float(batch_total.detach().item()),
            components=components,
            grad_norm=float(grad_norm),
            elapsed_s=elapsed,
            n_samples=len(batch.samples),
            skipped_samples=skipped,
            notes=notes,
            vram_peak_gb=peak_gb,
        )
