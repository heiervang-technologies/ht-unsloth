# SPDX-License-Identifier: AGPL-3.0-only
"""W1 — LLM-family compatibility CPU smoke.

Exercises the three invariants that must hold across every family in the
`lile` compat matrix (see `lile/docs/research/production-implementation-roadmap.md`
§18):

  1. **PR A** — `TrainEngine.reset_optimizer()` drops `_opt` to None, so
     `_optimizer()` rebuilds fresh Adam m/v against restored weights after
     `snapshot_load`.
  2. **Residual-delta attach point** — a bf16 sidecar bound as an attribute
     on a weight Parameter is the mechanism the monkey-patched
     `unsloth.kernels.utils.matmul_lora` reads. The binding must survive
     in-place data mutation (Unsloth's mode-flip proxy) without shifting
     `id(W)`.
  3. **PR B** — per-objective optimizer instances (`dict[str, AdamW]`)
     with isolated `m`/`v` state (placeholder: xfail until PR B lands).

CPU-only: no model load, no unsloth import. Per-family integration
(does `matmul_lora` actually fire on each family's fast path?) lives in
`test_residual_live_<family>.py` and runs on GPU. This file pins the
shared invariants that must survive regardless of family.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from lile.engine.train import TrainEngine

pytestmark = pytest.mark.cpu_only

# Families from the compat matrix. Adding one here does not change behavior —
# the axes under test are family-independent today. Divergence (e.g. a family
# that bypasses `matmul_lora` on its fast path) lands as an xfail entry plus
# a per-family integration test.
FAMILIES = [
    "qwen3",
    "llama3",
    "deepseek-r1",
    "gpt-oss",
    "magistral",
    "mistral",
    "phi-4",
    "gemma-3",
]


@pytest.mark.parametrize("family", FAMILIES)
def test_reset_optimizer_per_family(family: str) -> None:
    engine = TrainEngine.__new__(TrainEngine)
    engine._opt = MagicMock()
    engine.reset_optimizer()
    assert engine._opt is None, f"{family}: reset_optimizer did not clear _opt"


@pytest.mark.parametrize("family", FAMILIES)
def test_residual_delta_binding_per_family(family: str) -> None:
    W = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    delta = torch.ones(4, 4, dtype=torch.bfloat16)
    W._residual_delta = delta  # type: ignore[attr-defined]

    assert hasattr(W, "_residual_delta"), f"{family}: attribute bind failed"
    assert torch.equal(W._residual_delta, delta), (
        f"{family}: delta identity lost after bind"
    )

    # id(W) stability under an in-place mutation stands in for an Unsloth
    # `for_training`/`for_inference` flip. If it shifts, the patch's
    # per-Parameter attribute read no longer finds the delta on the same
    # object, and the fast path silently drops the residual.
    wid_before = id(W)
    W.data.add_(torch.zeros_like(W.data))
    assert id(W) == wid_before, (
        f"{family}: id(W) shifted under in-place mutation — matmul_lora "
        "patch would fail to locate _residual_delta"
    )
    assert torch.equal(W._residual_delta, delta), (
        f"{family}: delta detached from Parameter after in-place op"
    )


@pytest.mark.xfail(reason="PR B (per-objective optimizer instances) not yet landed",
                   strict=False)
@pytest.mark.parametrize("family", FAMILIES)
def test_per_objective_optimizer_per_family(family: str) -> None:
    # PyTorch keys `optimizer.state[param]` by tensor id, so N param_groups
    # over the same LoRA params share `m`/`v` — only LR isolates. Isolating
    # per-objective second-moment `v` requires separate optimizer instances.
    # PR B wires `self._opts: dict[str, torch.optim.AdamW]` behind
    # `cfg.per_objective_optim=False` (default-off: VRAM hit is ~400MB-1.6GB
    # of fp32 Adam state per instance on 7B+). Flipping `strict=True` on the
    # xfail above once PR B lands is the intended tripwire.
    raise NotImplementedError("pending PR B")
