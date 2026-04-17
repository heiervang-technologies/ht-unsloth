"""Pluggable objective registry.

Each objective is a callable mapping an :class:`Batch` (a flexible payload of
prompts + targets + auxiliary fields) to a *scalar loss tensor* with autograd
attached. Objectives compose by weighted sum at the trainer level.

The registry is a plain dict so users can add custom objectives at runtime via
:func:`register`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


# --- Batch & sample payloads -------------------------------------------

@dataclass
class Sample:
    """A single training sample. Fields are sparse — only what the objective
    needs is required."""

    prompt: str
    target: str | None = None  # for SFT, weighted SFT, distill
    rejected: str | None = None  # for hinge/contrastive, DPO-shape
    label: str | None = None  # for KTO: "desirable" | "undesirable"
    critique: str | None = None  # for CoH, CCPD
    response: str | None = None  # the original response that earned the critique
    weight: float = 1.0
    objectives: list[dict[str, Any]] | None = None  # per-sample obj overrides


@dataclass
class Batch:
    """A composer-level batch. ``samples`` are the per-sample payloads;
    ``batch_objectives`` adds objectives applied once per step (e.g. KL anchor)."""

    samples: list[Sample]
    batch_objectives: list[dict[str, Any]] = field(default_factory=list)


# --- Loss & trainer protocol -------------------------------------------

ObjectiveFn = Callable[..., torch.Tensor]


@dataclass
class ObjectiveSpec:
    """Registered objective metadata."""

    name: str
    fn: ObjectiveFn
    per_sample: bool  # True if applied per Sample; False if batch-level
    requires: tuple[str, ...] = ()  # required Sample fields (for validation)
    description: str = ""


_REGISTRY: dict[str, ObjectiveSpec] = {}


def register(spec: ObjectiveSpec) -> ObjectiveSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"Objective {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ObjectiveSpec:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown objective {name!r}. Registered: {sorted(_REGISTRY)}"
        ) from e


def list_objectives() -> list[str]:
    return sorted(_REGISTRY)


def validate_sample(sample: Sample, obj_name: str) -> None:
    """Raise if ``sample`` lacks any field required by objective ``obj_name``."""
    spec = get(obj_name)
    if not spec.per_sample:
        raise ValueError(
            f"Objective {obj_name!r} is batch-level; do not put it in sample.objectives"
        )
    missing = [f for f in spec.requires if getattr(sample, f) in (None, "")]
    if missing:
        raise ValueError(
            f"Objective {obj_name!r} requires sample fields {missing}; got "
            f"sample={sample!r}"
        )


def validate_batch_objective(obj_name: str) -> None:
    spec = get(obj_name)
    if spec.per_sample:
        raise ValueError(
            f"Objective {obj_name!r} is per-sample; put it in sample.objectives"
        )


# --- Re-export the loaded objectives ------------------------------------
# Importing the modules registers their specs as a side effect.

from lile.objectives import (  # noqa: E402, F401
    sft, kto, coh, hinge, ccpd, kl_anchor, rejection_sft,
)


__all__ = [
    "Sample",
    "Batch",
    "ObjectiveSpec",
    "register",
    "get",
    "list_objectives",
    "validate_sample",
    "validate_batch_objective",
]
