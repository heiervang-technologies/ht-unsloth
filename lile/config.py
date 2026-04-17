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


@dataclass
class KLAnchorSpec:
    """Configuration for an optional KL anchor term added once per step."""
    target: str = "base"   # "base" | "ema" | "snapshot:<name>"
    weight: float = 0.0
