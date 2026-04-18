"""Objective registry.

Objectives are composable. A batch carries a list of per-sample objectives plus
a list of per-batch objectives. The trainer composes them into a weighted sum.
"""
from __future__ import annotations

from typing import Any, Callable

from .sft import sft_loss, weighted_sft_loss
from .ntp import ntp_loss
from .kto import kto_loss
from .coh import coh_loss
from .hinge import hinge_contrastive_loss
from .kl import kl_anchor_loss
from .safety import safety_monitor_loss
from .unlike import unlike_loss

# Registry: objective_name -> loss_fn(model, batch, **kwargs) -> dict
#
# Each loss_fn returns a dict with:
#   "loss": torch.Tensor (scalar, graph-attached)
#   "components": dict[str, float]   # for logging
OBJECTIVES: dict[str, Callable[..., dict[str, Any]]] = {
    "sft": sft_loss,
    "weighted_sft": weighted_sft_loss,
    "ntp": ntp_loss,
    "kto": kto_loss,
    "coh": coh_loss,
    "hinge": hinge_contrastive_loss,
    "kl_anchor": kl_anchor_loss,
    "safety_monitor": safety_monitor_loss,
    "unlike": unlike_loss,
}

# CCPD v2 is registered conditionally in objectives/ccpd.py if import succeeds.
try:
    from .ccpd import ccpd_v2_loss  # noqa: F401
    OBJECTIVES["ccpd_v2"] = ccpd_v2_loss
except Exception:
    pass


def get_objective(name: str):
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; registered: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name]
