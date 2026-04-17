"""Central configuration. Kept dataclass-simple; no YAML parsing unless asked."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServeConfig:
    model: str = "unsloth/qwen3-0.6b-unsloth-bnb-4bit"
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    load_in_4bit: bool = True

    host: str = "127.0.0.1"
    port: int = 8000

    data_dir: Path = field(default_factory=lambda: Path("./lile_data"))
    max_queue_depth: int = 64

    default_lr: float = 1e-5
    default_objective: str = "sft"

    # --- T4.1 idle replay ---------------------------------------------------
    # When true, a background task re-injects logged feedback records as
    # training batches whenever the compute queue has been idle for
    # ``idle_replay_threshold_s``. See ``lile/engine/replay.py``.
    idle_replay: bool = False
    idle_replay_threshold_s: float = 30.0
    replay_poll_interval_s: float = 2.0
    replay_max_per_record: int = 3
    replay_half_life_h: float = 24.0
    replay_min_records: int = 3

    # --- frozen reference model --------------------------------------------
    # When true, ``ModelState.load_frozen_ref()`` loads a second base-only
    # model (eval, requires_grad=False) that objectives consume as ``pi_ref``
    # for KL anchoring. When false (default), the KL anchor falls back to
    # ``model.disable_adapter()`` on the live model — cheaper, but anchored
    # to the live merged_deltas rather than session-start.
    frozen_ref: bool = False


@dataclass
class KLAnchorSpec:
    """Configuration for an optional KL anchor term added once per step."""
    target: str = "base"   # "base" | "ema" | "snapshot:<name>"
    weight: float = 0.0
