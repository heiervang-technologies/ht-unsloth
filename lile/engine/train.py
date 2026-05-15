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


# Sentinel key used when per_objective=False — all objectives share one opt.
# Empty string is safe because `get_objective` rejects empty names, so no real
# objective collides with the shared slot.
_SHARED_KEY = ""


class TrainEngine:
    def __init__(self, state: ModelState, lr: float = 1e-5,
                 grad_clip: float = 1.0, per_objective: bool = False,
                 per_objective_lr: dict[str, float] | None = None,
                 default_watchlist: list[int] | None = None) -> None:
        self.state = state
        self.lr = lr
        self.grad_clip = grad_clip
        self.per_objective = per_objective
        self.per_objective_lr = dict(per_objective_lr or {})
        # Daemon-global safety_monitor watchlist floor. Three-tier union
        # (daemon ∪ batch ∪ per-sample) is resolved in safety_monitor_loss;
        # here we just forward the daemon-global slice. See
        # safety-monitor-primitive.md.
        self.default_watchlist: list[int] = list(default_watchlist or [])
        # Map objective_name -> optimizer. When per_objective=False, the only
        # key is _SHARED_KEY and every step reuses it. When True, each
        # objective gets its own torch.optim.AdamW so Adam m/v stay isolated
        # per family — PyTorch keys optimizer.state by tensor id, so
        # param_groups alone won't isolate moments.
        self._opts: dict[str, torch.optim.Optimizer] = {}

    def _optimizer(self, objective: str = _SHARED_KEY) -> torch.optim.Optimizer:
        key = objective if self.per_objective else _SHARED_KEY
        opt = self._opts.get(key)
        if opt is None:
            params = [p for p in self.state.model.parameters() if p.requires_grad]
            if self.per_objective:
                # Plain 32-bit AdamW per objective. bitsandbytes AdamW8bit is
                # deliberately avoided here: its GlobalOptimManager is a
                # process-wide singleton that does not cleanly support
                # multiple instances over the same params. See
                # optimizer-sample-efficiency.md §3 + anti-patterns.
                lr = self.per_objective_lr.get(objective, self.lr)
                opt = torch.optim.AdamW(params, lr=lr)
                log.info("per-objective AdamW for %r (lr=%g)", objective, lr)
            else:
                # 8-bit Adam if bitsandbytes is present, else plain AdamW.
                try:
                    import bitsandbytes as bnb
                    opt = bnb.optim.AdamW8bit(params, lr=self.lr)
                    log.info("using bitsandbytes AdamW8bit (lr=%g)", self.lr)
                except Exception:
                    opt = torch.optim.AdamW(params, lr=self.lr)
                    log.info("using torch AdamW (lr=%g)", self.lr)
            self._opts[key] = opt
        return opt

    def reset_optimizer(self) -> None:
        # Adam-family `m`/`v` moments are conditioned on the weight trajectory
        # that produced recent gradients. After a snapshot_load jumps weights
        # to an earlier point, those moments mis-scale the first few steps —
        # see `optimizer-sample-efficiency.md` §1 concern #3. In
        # per-objective mode we drop every instance, not just one — snapshot
        # rewinds the shared weights that every optimizer's state is keyed to.
        self._opts.clear()

    def step(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Execute one training step according to `spec`.

        `spec`:
          {
            "objective": "sft" | "kto" | "coh" | "hinge" | "ccpd_v2",
            "samples": [...],
            "batch_objectives": [{"name": "kl_anchor", "weight": 0.1}],
            "kwargs": {...}  # passed into the objective loss
          }

        Multi-objective form (combined-loss; one backward over Σ wᵢ·Lᵢ):
          {
            "objectives": [
              {"name": "weighted_sft", "weight": 0.1, "samples": [...], "kwargs": {...}},
              {"name": "unlike",       "weight": 1.0, "samples": [...]},
              ...
            ],
            "samples": [...],          # default for entries without their own samples
            "kwargs": {...},           # default kwargs for entries without their own
            "batch_objectives": [...], # sidecars run once after primaries (use first
                                       # primary's result for safety_monitor plumbing)
          }

        Exactly one of ``objective`` and ``objectives`` must be set.

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

            if spec.get("objectives"):
                return self._step_multi(spec)

            name = spec["objective"]
            samples = spec.get("samples", [])
            kwargs = dict(spec.get("kwargs", {}))
            # Inject frozen reference if loaded; objectives that care (CCPD v2,
            # KL anchor) consume it, the rest absorb it via **_.
            if self.state.frozen_ref is not None and "pi_ref" not in kwargs:
                kwargs["pi_ref"] = self.state.frozen_ref
            # Plumb the effective LR + batch_objectives into the primary
            # objective kwargs. Every objective absorbs unknown kwargs via
            # ``**_`` — ``unlike_loss`` consumes them to drive its tiered
            # precondition gate (see ``unlike-tiered-preconditions.md``).
            # Keeps the primitive pure — no reach-through into config.
            if "effective_lr" not in kwargs:
                kwargs["effective_lr"] = self.per_objective_lr.get(
                    name, self.lr,
                )
            if "batch_objectives" not in kwargs:
                kwargs["batch_objectives"] = list(
                    spec.get("batch_objectives", []) or [],
                )
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
                if bo_name == "safety_monitor":
                    # Plumb main-objective target positions + batch tensors so
                    # the sidecar piggybacks rather than re-tokenizing.
                    # Missing keys ⇒ safety_monitor raises RuntimeError on
                    # the caller — that's the contract (test 9).
                    for k in ("target_positions", "target_token_ids",
                              "input_ids", "attention_mask"):
                        if k in result and k not in bo_kwargs:
                            bo_kwargs[k] = result[k]
                    bo_kwargs.setdefault(
                        "default_watchlist", self.default_watchlist,
                    )
                    bo_kwargs.setdefault(
                        "effective_lr",
                        self.per_objective_lr.get(name, self.lr),
                    )
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

            opt = self._optimizer(name)
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

            # Post-step adapter + residual norm. Counterpart to grad_norm:
            # grad_norm is the *impulse* this step applied; these are the
            # *cumulative* size of the LoRA delta (live + merged residual).
            # Complement each other on the dashboard.
            adapter_sq = 0.0
            for p in self.state.model.parameters():
                if p.requires_grad:
                    adapter_sq += float(p.detach().pow(2).sum())
            components["adapter_norm_total"] = adapter_sq ** 0.5
            residual_sq = 0.0
            for d in self.state.merged_deltas.values():
                residual_sq += float(d.detach().pow(2).sum())
            components["residual_norm_total"] = residual_sq ** 0.5

            return {"loss": components["loss"], "components": components, "skipped": False}

    def _step_multi(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Combined-loss path: Σᵢ wᵢ·Lᵢ in one backward pass.

        Each entry of ``spec["objectives"]`` runs its existing objective
        function unchanged. Per-entry ``samples`` / ``kwargs`` override the
        spec-level defaults so the caller can route different rollouts to
        different primitives (RLVR pattern). ``batch_objectives`` execute once
        after primaries — for safety_monitor we plumb the FIRST primary's
        result tensors.

        Already inside ``state.mode_lock`` (the only caller is ``step``).
        """
        primaries = spec["objectives"]
        if not primaries:
            raise ValueError("objectives list is empty")
        default_samples = spec.get("samples", [])
        default_kwargs = dict(spec.get("kwargs", {}))

        total_loss: torch.Tensor | float | None = None
        components: dict[str, Any] = {}
        first_result: dict[str, Any] | None = None
        per_primary_losses: list[float | None] = []
        skipped_all = True

        for entry in primaries:
            name = entry["name"]
            weight = float(entry.get("weight", 1.0))
            # Pydantic serializes Optional[X] = None explicitly; fall back to
            # default_* when the key is missing OR present-as-None.
            samples = entry.get("samples")
            if samples is None:
                samples = default_samples
            entry_kwargs = entry.get("kwargs")
            if entry_kwargs is None:
                entry_kwargs = default_kwargs
            kwargs = dict(entry_kwargs)
            # Plumb the same shared kwargs the single-objective path adds.
            if self.state.frozen_ref is not None and "pi_ref" not in kwargs:
                kwargs["pi_ref"] = self.state.frozen_ref
            if "effective_lr" not in kwargs:
                kwargs["effective_lr"] = self.per_objective_lr.get(name, self.lr)
            # Each entry sees the same batch_objectives list — primitives that
            # care (e.g. ``unlike`` precondition gate) consume them the same
            # way as in the single-objective branch.
            if "batch_objectives" not in kwargs:
                kwargs["batch_objectives"] = list(
                    spec.get("batch_objectives", []) or [],
                )
            fn = get_objective(name)
            result = fn(self.state.model, self.state.tokenizer, samples, **kwargs)
            if first_result is None:
                first_result = result
            sub_loss = result.get("loss")
            if sub_loss is None:
                # Skipped (e.g. unlike precondition gate fired). Record but
                # don't accumulate — the primitive already logged via its own
                # components dict.
                per_primary_losses.append(None)
                for k, v in result.get("components", {}).items():
                    components[f"{name}.{k}"] = v
                components[f"{name}.weight"] = weight
                components[f"{name}.skipped"] = True
                continue
            skipped_all = False
            weighted = weight * sub_loss
            total_loss = weighted if total_loss is None else (total_loss + weighted)
            per_primary_losses.append(float(sub_loss.detach().cpu()))
            for k, v in result.get("components", {}).items():
                components[f"{name}.{k}"] = v
            components[f"{name}.weight"] = weight
            components[f"{name}.weighted_loss"] = float(weighted.detach().cpu())

        # Sidecar batch_objectives — run once on top of the combined primary
        # loss. Use first primary's result as the source of target_positions
        # / input_ids tensors so safety_monitor can piggyback. Samples for
        # batch primitives (kl_anchor needs at least one) come from the first
        # active primary — its samples are what just produced gradients, so
        # anchoring against them keeps the regularization aligned.
        bo_samples: list[Any] = list(default_samples)
        if not bo_samples:
            for entry in primaries:
                ent_samples = entry.get("samples")
                if ent_samples:
                    bo_samples = list(ent_samples)
                    break
        for bo in spec.get("batch_objectives", []) or []:
            bo_name = bo["name"]
            bo_fn = get_objective(bo_name)
            bo_kwargs = {k: v for k, v in bo.items() if k != "name"}
            if self.state.frozen_ref is not None and "pi_ref" not in bo_kwargs:
                bo_kwargs["pi_ref"] = self.state.frozen_ref
            if bo_name == "safety_monitor" and first_result is not None:
                for k in ("target_positions", "target_token_ids",
                          "input_ids", "attention_mask"):
                    if k in first_result and k not in bo_kwargs:
                        bo_kwargs[k] = first_result[k]
                bo_kwargs.setdefault(
                    "default_watchlist", self.default_watchlist,
                )
                bo_kwargs.setdefault(
                    "effective_lr",
                    self.per_objective_lr.get(primaries[0]["name"], self.lr),
                )
            bo_result = bo_fn(self.state.model, self.state.tokenizer,
                              bo_samples, **bo_kwargs)
            bo_loss = bo_result.get("loss")
            if bo_loss is not None:
                total_loss = (total_loss if total_loss is not None else 0.0) + bo_loss
                skipped_all = False
            for k, v in bo_result.get("components", {}).items():
                components[f"batch.{bo_name}.{k}"] = v

        if total_loss is None or skipped_all:
            log.info("multi-objective step skipped — all primaries gated out")
            components["skipped_reason"] = "all_primaries_skipped"
            return {"loss": None, "components": components, "skipped": True}

        # Single optimizer step over the combined loss. Uses shared optimizer
        # slot; per-objective Adam moments don't make sense when the gradient
        # is a linear combination — m/v would be incoherent.
        opt = self._optimizer(_SHARED_KEY)
        opt.zero_grad()
        total_loss.backward()
        grad_norm_total: float | None = None
        if self.grad_clip and self.grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in self.state.model.parameters() if p.requires_grad],
                self.grad_clip,
            )
            grad_norm_total = float(gn)
        opt.step()

        components["loss"] = float(total_loss.detach().cpu())
        components["objectives_count"] = len(primaries)
        components["objectives_active"] = sum(
            1 for ell in per_primary_losses if ell is not None
        )
        if grad_norm_total is not None:
            components["grad_norm_total"] = grad_norm_total
            components["grad_clipped"] = bool(grad_norm_total > self.grad_clip)

        adapter_sq = 0.0
        for p in self.state.model.parameters():
            if p.requires_grad:
                adapter_sq += float(p.detach().pow(2).sum())
        components["adapter_norm_total"] = adapter_sq ** 0.5
        residual_sq = 0.0
        for d in self.state.merged_deltas.values():
            residual_sq += float(d.detach().pow(2).sum())
        components["residual_norm_total"] = residual_sq ** 0.5

        return {"loss": components["loss"], "components": components, "skipped": False}
