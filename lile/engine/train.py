"""Training engine: composes objectives, runs forward+backward, manages LR.

This is the thing the compute queue handler calls on every train task. It
does NOT own the model — the ModelState owns the model, and this engine
takes it as a dependency. Same for the optimizer, which we create lazily
on first backward.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from ..objectives import get_objective
from ..state import ModelState

log = logging.getLogger(__name__)


class TrainEngine:
    def __init__(self, state: ModelState, lr: float = 1e-5,
                 grad_clip: float = 1.0) -> None:
        self.state = state
        self.lr = lr
        self.grad_clip = grad_clip
        self._opt: torch.optim.Optimizer | None = None

    def _optimizer(self) -> torch.optim.Optimizer:
        if self._opt is None:
            params = [p for p in self.state.model.parameters() if p.requires_grad]
            # 8-bit Adam if bitsandbytes is present, else plain AdamW.
            try:
                import bitsandbytes as bnb
                self._opt = bnb.optim.AdamW8bit(params, lr=self.lr)
                log.info("using bitsandbytes AdamW8bit (lr=%g)", self.lr)
            except Exception:
                self._opt = torch.optim.AdamW(params, lr=self.lr)
                log.info("using torch AdamW (lr=%g)", self.lr)
        return self._opt

    def step(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Execute one training step according to `spec`.

        `spec`:
          {
            "objective": "sft" | "kto" | "coh" | "hinge" | "ccpd_v2",
            "samples": [...],
            "batch_objectives": [{"name": "kl_anchor", "weight": 0.1}],
            "kwargs": {...}  # passed into the objective loss
          }

        Held under `state.mode_lock` so the Unsloth mode flip + forward +
        backward + optimizer step are mutually exclusive with any concurrent
        inference on the same model. See `state.ModelState.mode_lock`.
        """
        with self.state.mode_lock:
            # Training mode + LoRA grads on.
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(self.state.model)
            except Exception:
                pass
            self.state.model.train()

            name = spec["objective"]
            samples = spec.get("samples", [])
            kwargs = dict(spec.get("kwargs", {}))
            # Inject frozen reference if loaded; objectives that care (CCPD v2,
            # KL anchor) consume it, the rest absorb it via **_.
            if self.state.frozen_ref is not None and "pi_ref" not in kwargs:
                kwargs["pi_ref"] = self.state.frozen_ref
            fn = get_objective(name)
            result = fn(self.state.model, self.state.tokenizer, samples, **kwargs)
            loss = result["loss"]
            components = dict(result.get("components", {}))

            # Stack batch-level objectives (KL anchor, etc.).
            for bo in spec.get("batch_objectives", []):
                bo_name = bo["name"]
                bo_fn = get_objective(bo_name)
                bo_kwargs = {k: v for k, v in bo.items() if k != "name"}
                if self.state.frozen_ref is not None and "pi_ref" not in bo_kwargs:
                    bo_kwargs["pi_ref"] = self.state.frozen_ref
                bo_result = bo_fn(self.state.model, self.state.tokenizer,
                                  samples, **bo_kwargs)
                bo_loss = bo_result.get("loss")
                if bo_loss is not None:
                    loss = (loss if loss is not None else 0.0) + bo_loss
                for k, v in bo_result.get("components", {}).items():
                    components[f"batch.{bo_name}.{k}"] = v

            if loss is None:
                log.info("objective %s returned None (skipped: %s)", name, components)
                return {"loss": None, "components": components, "skipped": True}

            opt = self._optimizer()
            opt.zero_grad()
            loss.backward()
            grad_norm_total: float | None = None
            if self.grad_clip and self.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.state.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                grad_norm_total = float(gn)
            opt.step()

            components["loss"] = float(loss.detach().cpu())
            if grad_norm_total is not None:
                components["grad_norm_total"] = grad_norm_total
                components["grad_clipped"] = bool(grad_norm_total > self.grad_clip)
            return {"loss": components["loss"], "components": components, "skipped": False}
