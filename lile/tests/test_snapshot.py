"""Tests for :mod:`lile.snapshot` — save/restore round-trip + manifest validation.

We use the same synthetic LoRA wrapper from :mod:`test_merge`, plus a synthetic
``LiveState`` shim so the test runs CPU-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from lile.adapters import AdapterManager, lora_state_dict
from lile.snapshot import (
    DELTAS_NAME,
    LORA_NAME,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    load_snapshot_manifest,
    restore_snapshot,
    save_snapshot,
)
from lile.tests.test_merge import _SyntheticLoRALinear, _Wrapper


# --- Shim for LiveState surface that snapshot.py touches -----------------


@dataclass
class _StateConfigShim:
    model_name: str
    max_seq_length: int = 512
    load_in_4bit: bool = False
    full_finetuning: bool = False
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_targets: tuple[str, ...] = ("layer1", "layer2")


class _StateShim:
    """Mirrors the surface of :class:`lile.state.LiveState` that
    :func:`save_snapshot` / :func:`restore_snapshot` consult."""

    def __init__(self, model: nn.Module, config: _StateConfigShim):
        self.config = config
        self.model = model
        self.merge_count = 0


def _make_state(model_name: str = "test/qwen-tiny") -> _StateShim:
    torch.manual_seed(0)
    return _StateShim(_Wrapper(), _StateConfigShim(model_name=model_name))


# --- Tests --------------------------------------------------------------


def test_save_writes_manifest_and_lora(tmp_path: Path):
    state = _make_state()
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(state.model, deltas)

    out = save_snapshot(state, mgr, tmp_path / "snap1")
    assert out.exists()
    assert (out / MANIFEST_NAME).exists()
    # No deltas yet → no deltas.safetensors.
    assert not (out / DELTAS_NAME).exists()

    manifest = load_snapshot_manifest(out)
    assert manifest["schema"] == SCHEMA_VERSION
    assert manifest["model_name"] == "test/qwen-tiny"
    assert manifest["merge_count"] == 0


def test_save_writes_deltas_after_merge(tmp_path: Path):
    state = _make_state()
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(state.model, deltas)
    mgr.merge_active_lora()
    state.merge_count = 1

    out = save_snapshot(state, mgr, tmp_path / "snap2")
    assert (out / DELTAS_NAME).exists()
    manifest = load_snapshot_manifest(out)
    assert manifest["merge_count"] == 1
    assert manifest["n_delta_tensors"] == 2


def test_save_is_atomic_overwrites_existing(tmp_path: Path):
    state = _make_state()
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(state.model, deltas)

    out = save_snapshot(state, mgr, tmp_path / "snap3")
    assert (out / MANIFEST_NAME).exists()
    # Save again into the same dir — must not fail and must replace.
    state.merge_count = 5
    save_snapshot(state, mgr, tmp_path / "snap3")
    manifest = load_snapshot_manifest(tmp_path / "snap3")
    assert manifest["merge_count"] == 5

    # No leftover .snapshot.* tmpdirs.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".snapshot.")]
    assert leftovers == []


def test_round_trip_lora_and_deltas_preserves_outputs(tmp_path: Path):
    """End-to-end: train (synth merge), save, restore into a fresh model with
    matching base weights, verify outputs equal."""
    src = _make_state()
    src_deltas: dict[str, torch.Tensor] = {}
    src_mgr = AdapterManager(src.model, src_deltas)
    src_mgr.merge_active_lora()
    # Zero LoRA so the comparison is purely about base + residual.
    with torch.no_grad():
        for name in ("layer1", "layer2"):
            getattr(src.model, name).lora_B["default"].weight.zero_()
    src.merge_count = 1

    out = save_snapshot(src, src_mgr, tmp_path / "snap4")

    # Build a fresh state with the same base weights and zero LoRA.
    dst = _make_state()
    with torch.no_grad():
        for name in ("layer1", "layer2"):
            getattr(dst.model, name).base_layer.weight.copy_(
                getattr(src.model, name).base_layer.weight
            )
            getattr(dst.model, name).lora_B["default"].weight.zero_()

    dst_deltas: dict[str, torch.Tensor] = {}
    dst_mgr = AdapterManager(dst.model, dst_deltas)
    manifest = restore_snapshot(dst, dst_mgr, out)

    assert manifest["merge_count"] == 1
    assert dst.merge_count == 1

    x = torch.randn(2, 8)
    src_out = src.model.layer1(x)
    dst_out = dst.model.layer1(x)
    assert torch.allclose(src_out, dst_out, atol=1e-3), (
        f"snapshot restore changed outputs: max_diff={((src_out - dst_out).abs().max()).item()}"
    )


def test_restore_rejects_schema_mismatch(tmp_path: Path):
    state = _make_state()
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(state.model, deltas)
    out = save_snapshot(state, mgr, tmp_path / "snap_bad_schema")
    # Corrupt the manifest's schema number.
    manifest_path = out / MANIFEST_NAME
    raw = json.loads(manifest_path.read_text())
    raw["schema"] = SCHEMA_VERSION + 99
    manifest_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="schema"):
        restore_snapshot(state, mgr, out)


def test_restore_rejects_model_name_mismatch(tmp_path: Path):
    src = _make_state(model_name="aaa")
    deltas: dict[str, torch.Tensor] = {}
    mgr = AdapterManager(src.model, deltas)
    out = save_snapshot(src, mgr, tmp_path / "snap_diff_model")

    # Try to restore into a state with a different model name.
    dst = _make_state(model_name="bbb")
    dst_deltas: dict[str, torch.Tensor] = {}
    dst_mgr = AdapterManager(dst.model, dst_deltas)

    with pytest.raises(ValueError, match="model"):
        restore_snapshot(dst, dst_mgr, out)
