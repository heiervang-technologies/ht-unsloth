"""Snapshot save / restore.

A snapshot is a directory containing:

* ``manifest.json`` — schema version, base model name, merge_count, trajectory
  log offset, queue committed_seq, timestamps.
* ``lora.safetensors`` — the active LoRA parameters (if not full-FT).
* ``deltas.safetensors`` — the merged_deltas residual (if any merges have
  happened).

Restore is the reverse: read manifest, instantiate :class:`lile.state.LiveState`
from the recorded base, load LoRA + deltas, reinstall residual hooks. The
post-restore output should be numerically identical to the pre-save output for
a fixed prompt + greedy decode (verified by ``tests/test_snapshot.py``).

We avoid `torch.save` / `pickle` to keep snapshots portable and inspectable.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
LORA_NAME = "lora.safetensors"
DELTAS_NAME = "deltas.safetensors"


def save_snapshot(
    state,
    adapter_mgr,
    out_dir: str | os.PathLike[str],
    *,
    trajectory_offset: int | None = None,
    committed_seq: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write the snapshot to ``out_dir``.

    Atomic: writes to a sibling tmpdir then ``os.replace``s into place. The
    ``out_dir`` is overwritten if it exists.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=str(out_dir.parent)))
    try:
        # 1. LoRA params (if any)
        lora_sd: dict[str, torch.Tensor] = {}
        for k, v in state.model.state_dict().items():
            if ".lora_A." in k or ".lora_B." in k:
                lora_sd[k] = v.detach().cpu().contiguous().to(torch.bfloat16)
        if lora_sd:
            save_file(lora_sd, str(tmp / LORA_NAME))

        # 2. Merged deltas (if any)
        deltas_sd = adapter_mgr.deltas_state_dict() if adapter_mgr is not None else {}
        if deltas_sd:
            # Ensure contiguous bf16.
            deltas_sd = {k: v.contiguous().to(torch.bfloat16) for k, v in deltas_sd.items()}
            save_file(deltas_sd, str(tmp / DELTAS_NAME))

        # 3. Manifest
        manifest = {
            "schema": SCHEMA_VERSION,
            "saved_at": time.time(),
            "model_name": state.config.model_name,
            "max_seq_length": state.config.max_seq_length,
            "load_in_4bit": state.config.load_in_4bit,
            "full_finetuning": state.config.full_finetuning,
            "lora_rank": state.config.lora_rank,
            "lora_alpha": state.config.lora_alpha,
            "lora_targets": list(state.config.lora_targets),
            "merge_count": state.merge_count,
            "trajectory_offset": trajectory_offset,
            "committed_seq": committed_seq,
            "n_lora_tensors": len(lora_sd),
            "n_delta_tensors": len(deltas_sd),
            "extra": extra or {},
        }
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        # 4. Atomic publish
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(str(tmp), str(out_dir))
        return out_dir
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def load_snapshot_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path) / MANIFEST_NAME
    return json.loads(p.read_text())


def restore_snapshot(
    state,
    adapter_mgr,
    in_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load LoRA + deltas from ``in_dir`` into ``state``/``adapter_mgr``.

    Caller is responsible for having already loaded the base model into
    ``state`` (via :meth:`LiveState.load`); we only restore the *deltas* on top.
    """
    in_dir = Path(in_dir)
    manifest = load_snapshot_manifest(in_dir)
    if manifest["schema"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported snapshot schema {manifest['schema']!r}")

    if manifest["model_name"] != state.config.model_name:
        raise ValueError(
            f"Snapshot is for model {manifest['model_name']!r} but state was "
            f"loaded as {state.config.model_name!r}"
        )

    # 1. LoRA
    lora_path = in_dir / LORA_NAME
    if lora_path.exists():
        sd = load_file(str(lora_path))
        own_sd = state.model.state_dict()
        n_loaded = 0
        for k, v in sd.items():
            if k not in own_sd:
                continue
            tgt = own_sd[k]
            tgt.copy_(v.to(tgt.device, dtype=tgt.dtype))
            n_loaded += 1
        if n_loaded == 0:
            raise RuntimeError(
                f"Snapshot LoRA tensors did not match any model state key. "
                f"Sample missing: {next(iter(sd))!r}"
            )
    # 2. Deltas
    deltas_path = in_dir / DELTAS_NAME
    if deltas_path.exists() and adapter_mgr is not None:
        adapter_mgr.load_deltas_state_dict(load_file(str(deltas_path)))

    state.merge_count = manifest.get("merge_count", 0)
    return manifest
