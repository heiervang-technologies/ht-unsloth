"""Snapshot manager.

A snapshot is `(base_ref, merged_deltas.safetensors, active_adapter.safetensors,
trajectory_log_offset)` — everything needed to reproduce the live model state.
See LIVELEARN §3.1.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from safetensors.torch import load_file, save_file

from .state import ModelState
from .trajectory import TrajectoryLog

log = logging.getLogger(__name__)


class SnapshotManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, name: str) -> Path:
        return self.root / name

    def save(self, name: str, state: ModelState, log_: TrajectoryLog | None = None) -> Path:
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": name,
            "created_at": time.time(),
            "base_model": state.base_model_name,
            "lora_rank": state.lora_rank,
            "lora_alpha": state.lora_alpha,
            "merges_applied": state.merges_applied,
            "residual_fingerprint": state.residual_fingerprint(),
            "trajectory_offset": log_.size() if log_ else 0,
            "trajectory_path": str(log_.path) if log_ else None,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # Residual (bf16 CPU) — always write even if empty so restore works.
        if state.merged_deltas:
            save_file(
                {k: v.contiguous() for k, v in state.merged_deltas.items()},
                str(d / "merged_deltas.safetensors"),
            )
        # Active adapter (LoRA A/B).
        sd = state.extract_active_adapter()
        if sd:
            save_file(sd, str(d / "active_adapter.safetensors"))
        log.info("snapshot saved to %s", d)
        return d

    def load(self, name: str, state: ModelState) -> dict:
        d = self._dir(name)
        manifest = json.loads((d / "manifest.json").read_text())
        residual_path = d / "merged_deltas.safetensors"
        if residual_path.exists():
            state.merged_deltas = load_file(str(residual_path))
        else:
            state.merged_deltas = {}
        adapter_path = d / "active_adapter.safetensors"
        if adapter_path.exists():
            state.load_active_adapter(load_file(str(adapter_path)))
        else:
            state.reset_active_adapter()
        state.merges_applied = manifest.get("merges_applied", 0)
        log.info("snapshot %s loaded (residual fp=%s)", name,
                 manifest.get("residual_fingerprint"))
        return manifest

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())
