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

    # Engine default LR. This value is a **known-unsafe regime** for the
    # ``unlike`` objective with a positive teacher — see ``objectives/unlike.py``
    # module docstring and ``DESIGN.md`` §Safety regime. Cleo's
    # razin-safety-sharpened.md (``docs/research/proofs/``) shows that at small
    # eta the positive-teacher side of unlike can push ``p_bad`` UP rather than
    # down. Scripts that call ``objective="unlike"`` should override via
    # ``per_objective_lr={"unlike": 5e-5}`` or higher (empirical safe floor
    # pending the ``unlike-defaults-calibration-sweep.md`` deliverable).
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

    # --- PR L: TTRL majority-vote pseudo-reward -----------------------------
    # When true, an idle-time scheduler samples ``ttrl_k_rollouts`` completions
    # for a verifier-claimed inference prompt, majority-votes over the
    # verifier-extracted answers, and enqueues an SFT step on the winning
    # rollout. Ships default-off; the roadmap's GSM8K eval gate is deferred
    # until ``lile[eval]`` is CI-promoted. See ``lile/teach/ttrl_mv.py``.
    ttrl_pseudo_reward: bool = False
    ttrl_k_rollouts: int = 4
    ttrl_idle_threshold_s: float = 30.0
    ttrl_poll_interval_s: float = 2.0
    ttrl_max_per_prompt: int = 3
    ttrl_min_prompts: int = 3
    ttrl_temperature: float = 0.8
    ttrl_top_p: float = 0.95

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

    # --- per-objective optimizer instances ---------------------------------
    # When true, ``TrainEngine`` keeps a separate ``torch.optim.AdamW``
    # instance per objective name (``sft``, ``kto``, ``coh``, ...) so the
    # Adam second-moment ``v`` tracks each family's gradient scale
    # independently. PyTorch keys ``optimizer.state[param]`` by tensor id,
    # so sharing one optimizer across objectives — even with separate
    # param_groups — shares ``m``/``v``; only LR would isolate. Multiple
    # instances are the only way to isolate the running variance.
    #
    # Default off because VRAM cost is real: plain 32-bit Adam state doubles
    # the LoRA param memory per instance (≈400MB-1.6GB for LoRA r=64 on 7B+
    # depending on target_modules), times N objectives. Turn on only when
    # mixing objectives with substantially different grad magnitudes.
    #
    # Deliberately plain ``torch.optim.AdamW`` (not ``bnb.AdamW8bit``):
    # bitsandbytes' ``GlobalOptimManager`` is a process-wide singleton that
    # does not cleanly support multiple AdamW8bit instances over the same
    # params. See ``docs/research/optimizer-sample-efficiency.md`` §3.
    per_objective_optim: bool = False
    per_objective_lr: dict[str, float] = field(default_factory=dict)

    # --- /v1/commits/stream SSE -------------------------------------------
    # Per-commit event stream, one event per successful train-task cursor
    # advance. See ``lile/docs/research/pr-specs/commits-sse-stream.md``.
    # When false the subscriber set short-circuits and the training path
    # pays zero cost. Clients filter on the consumer side — no server-side
    # filter expressions (would drift toward per-workflow state).
    commits_sse_enabled: bool = True

    # --- dev-mode hot reload + crash-safe model state ---------------------
    # Two-layer approach. (1) ``dev_autoreload`` enables ``jurigged``: function
    # bodies across ``lile/**/*.py`` are patched live when you save a file, so
    # most edits apply to the running daemon without restart — model,
    # optimizer, residual, trajectory all stay warm. Body-level only:
    # structural edits (new file, new import, class hierarchy, registry dict
    # mutation) still require a process bounce. (2) For that bounce — or any
    # crash / reboot — ``autosave_on_exit`` writes a byte-exact snapshot
    # during graceful shutdown, and ``autoload_on_boot`` restores it on next
    # startup. The "_autosave" slot is the reserved name; user snapshots live
    # beside it unaffected. See ``lile/dev/autoreload.py`` and
    # ``lile/snapshot.py``.
    dev_autoreload: bool = False
    autosave_on_exit: bool = True
    autoload_on_boot: bool = True
    autosave_snapshot_name: str = "_autosave"

    # --- safety_monitor daemon-global watchlist ---------------------------
    # Three-tier union at step time: this daemon-global floor
    # (absolute-never tokens — PII / safety-critical), ∪ batch-level
    # ``batch_objectives[].watchlist``, ∪ per-sample ``sample["watchlist"]``.
    # Consumed only when a ``safety_monitor`` batch objective is present
    # in the spec; zero cost otherwise. See
    # ``lile/docs/research/pr-specs/safety-monitor-primitive.md``.
    default_watchlist: list[int] = field(default_factory=list)

    # --- RLVR online teacher (Track B) -------------------------------------
    # The four-role teacher (grade / critique / counterfactual /
    # demonstration) that drives the online RLVR loop. One OpenRouter call
    # per RLVR step routes through ``lile/teach/teacher_oss120b.py``; the
    # API key is read from ``OPENROUTER_API_KEY`` in the daemon env (see
    # ``compose.lile-dev.yaml``). ``teacher_max_concurrent`` caps in-flight
    # judge() calls so a stuck OpenRouter region can't pile up requests.
    teacher_model: str = "openai/gpt-oss-120b"
    teacher_url: str = "https://openrouter.ai/api/v1"
    teacher_max_concurrent: int = 4

    # --- RLVR online scheduler (Track C) -----------------------------------
    # Drives the online RLVR loop in ``lile/teach/rlvr_loop.py``. Default-off
    # so the daemon ships dark; flip ``rlvr_online`` to wire the scheduler
    # into the lifespan once the prompt sources and the daemon-side teacher
    # call budget are sized. ``rlvr_weights`` is the linear-combination
    # weighting consumed by the combined-loss engine (Track A): wᵢ in
    # Σ wᵢ·Lᵢ for the per-rollout objectives that the four-role teacher
    # routes to. ``kl`` is the always-on anchor (scope="target_position")
    # that brakes mass movement on the rest of the vocab.
    rlvr_online: bool = False
    rlvr_k: int = 4
    rlvr_source: str = "mixed"  # "math" | "code" | "arc" | "mixed"
    rlvr_weights: dict[str, float] = field(default_factory=lambda: {
        "sft": 0.1,
        "coh": 1.0,
        "kto": 1.0,
        "unlike": 0.5,
        "kl": 0.05,
    })
    rlvr_log_path: str = "lile_data/rlvr_loop.jsonl"


@dataclass
class KLAnchorSpec:
    """Configuration for an optional KL anchor term added once per step."""
    target: str = "base"   # "base" | "ema" | "snapshot:<name>"
    weight: float = 0.0
    scope: str = "prompt"  # "prompt" | "full_sequence" — see objectives/kl.py
