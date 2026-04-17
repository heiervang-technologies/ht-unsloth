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

    # Budget passed to ``Controller.graceful_shutdown`` on FastAPI shutdown
    # (SIGINT/SIGTERM via uvicorn's default handler). The queue worker keeps
    # pulling tasks while the budget holds and cleanly resolves the rest
    # with ``ShutdownDroppedError`` — so no ``wait_for(token)`` ever hangs.
    shutdown_deadline_s: float = 30.0

    # Extra grace after the deadline expires with a still-running in-flight
    # task. We never cancel mid-GPU-step (would tear the LoRA), so the queue
    # worker needs a bounded post-deadline window to finish. Operators should
    # size ``shutdown_deadline_s + shutdown_hard_stop_grace_s`` to stay under
    # ``terminationGracePeriodSeconds`` on k8s — otherwise the pod is
    # SIGKILLed mid-flight regardless of the graceful path.
    shutdown_hard_stop_grace_s: float = 30.0

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

    # --- metrics logging backend -------------------------------------------
    # Optional fan-out of train-step metrics to an external visualization
    # tool (wandb, tensorboard, mlflow, trackio). The trajectory JSONL
    # remains canonical; this is a mirror for charting. Default "null"
    # means no external sink and zero extra deps.
    logger: str = "null"  # null | wandb | tensorboard | mlflow | trackio
    logger_project: str = "lile"
    logger_run_name: str | None = None
    logger_log_dir: str | None = None       # tensorboard
    logger_tracking_uri: str | None = None  # mlflow

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
    scope: str = "prompt"  # "prompt" | "full_sequence" — see objectives/kl.py
