"""Pluggable metrics-logging backends.

The JSONL trajectory at ``lile_data/trajectory.jsonl`` is the canonical
record for every train step / inference / feedback event. This module is
an *optional fan-out* that mirrors the same numeric components to an
external visualization tool — Weights & Biases, TensorBoard, MLflow,
trackio — using each tool's standard client API. Missing optional deps
do not break lile; the adapter raises at construction, and ``get_logger``
falls back to ``NullLogger`` only when the user explicitly selected
``"null"`` (default).

Design notes:
  * Adapters never raise from ``log_metrics`` or ``log_params`` — they
    swallow and log so a bad network or a revoked API token can never
    crash the hot train loop.
  * All scalar flattening is done by the caller (see
    ``flatten_scalars``). Adapters receive ``dict[str, float]`` and
    forward it verbatim.
  * Lazy imports: the module imports cleanly even when wandb / mlflow /
    tensorboard / trackio are not installed; import errors surface only
    when the corresponding backend is constructed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- protocol
@runtime_checkable
class MetricsLogger(Protocol):
    """Minimum contract for an external metrics sink.

    Implementations must be no-throw from the perspective of the caller —
    a bad backend should log and continue, never abort a training step.
    """

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float],
                    step: int | None = None) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------- helpers
def flatten_scalars(source: Mapping[str, Any], prefix: str = "",
                    out: dict[str, float] | None = None) -> dict[str, float]:
    """Flatten a nested dict to ``{"a.b.c": float}`` pairs.

    - ``bool`` is coerced to ``float(0|1)`` so wandb-style backends can plot
      clip indicators as step-indexed traces.
    - Non-scalar values (strings, objects, None) are dropped — scalar sinks
      can't render them.
    """
    if out is None:
        out = {}
    for k, v in source.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, Mapping):
            flatten_scalars(v, key, out)
        # strings / None / objects: drop silently.
    return out


# ---------------------------------------------------------------------- config
@dataclass
class LoggerConfig:
    """Runtime config for the metrics sink. Constructed from ServeConfig
    fields so callers don't need to know about every backend's own args."""
    backend: str = "null"   # "null" | "wandb" | "tensorboard" | "mlflow" | "trackio"
    project: str = "lile"
    run_name: str | None = None
    log_dir: str | None = None          # tensorboard
    tracking_uri: str | None = None     # mlflow
    extra: dict[str, Any] | None = None  # free-form, passed to adapter init


# ---------------------------------------------------------------------- null
class NullLogger:
    """No-op sink. Selected by default so lile has zero external deps
    unless the user opts in."""

    def log_params(self, params: Mapping[str, Any]) -> None:  # noqa: D401
        return

    def log_metrics(self, metrics: Mapping[str, float],
                    step: int | None = None) -> None:
        return

    def close(self) -> None:
        return


# ---------------------------------------------------------------------- wandb
class WandbLogger:
    """Weights & Biases adapter. Uses the project client as-is —
    ``wandb.init`` / ``wandb.log`` / ``wandb.finish``."""

    def __init__(self, cfg: LoggerConfig) -> None:
        import wandb  # noqa: F401  (raise if missing)
        self._wandb = wandb
        kwargs: dict[str, Any] = {"project": cfg.project}
        if cfg.run_name:
            kwargs["name"] = cfg.run_name
        if cfg.extra:
            kwargs.update(cfg.extra)
        self._run = wandb.init(**kwargs)

    def log_params(self, params: Mapping[str, Any]) -> None:
        try:
            self._wandb.config.update(dict(params), allow_val_change=True)
        except Exception as exc:  # pragma: no cover
            log.warning("wandb log_params failed: %s", exc)

    def log_metrics(self, metrics: Mapping[str, float],
                    step: int | None = None) -> None:
        try:
            self._wandb.log(dict(metrics), step=step)
        except Exception as exc:  # pragma: no cover
            log.warning("wandb log_metrics failed: %s", exc)

    def close(self) -> None:
        try:
            self._wandb.finish()
        except Exception as exc:  # pragma: no cover
            log.warning("wandb finish failed: %s", exc)


# ---------------------------------------------------------------------- tensorboard
class TensorBoardLogger:
    """TensorBoard adapter via ``torch.utils.tensorboard.SummaryWriter``."""

    def __init__(self, cfg: LoggerConfig) -> None:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = cfg.log_dir or f"./runs/{cfg.run_name or cfg.project}"
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_params(self, params: Mapping[str, Any]) -> None:
        # SummaryWriter stores hparams via add_hparams, but that requires
        # a metric dict too; easier to stash as text so the user sees them
        # in the TEXT tab.
        try:
            text = "\n".join(f"{k}: {v}" for k, v in sorted(params.items()))
            self._writer.add_text("lile/params", text, global_step=0)
        except Exception as exc:  # pragma: no cover
            log.warning("tensorboard log_params failed: %s", exc)

    def log_metrics(self, metrics: Mapping[str, float],
                    step: int | None = None) -> None:
        try:
            for tag, value in metrics.items():
                self._writer.add_scalar(tag, value, global_step=step)
        except Exception as exc:  # pragma: no cover
            log.warning("tensorboard log_metrics failed: %s", exc)

    def close(self) -> None:
        try:
            self._writer.flush()
            self._writer.close()
        except Exception as exc:  # pragma: no cover
            log.warning("tensorboard close failed: %s", exc)


# ---------------------------------------------------------------------- mlflow
class MLflowLogger:
    """MLflow adapter via the tracking API."""

    def __init__(self, cfg: LoggerConfig) -> None:
        import mlflow
        self._mlflow = mlflow
        if cfg.tracking_uri:
            mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.project)
        self._run = mlflow.start_run(run_name=cfg.run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        try:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception as exc:  # pragma: no cover
            log.warning("mlflow log_params failed: %s", exc)

    def log_metrics(self, metrics: Mapping[str, float],
                    step: int | None = None) -> None:
        try:
            self._mlflow.log_metrics(dict(metrics), step=step)
        except Exception as exc:  # pragma: no cover
            log.warning("mlflow log_metrics failed: %s", exc)

    def close(self) -> None:
        try:
            self._mlflow.end_run()
        except Exception as exc:  # pragma: no cover
            log.warning("mlflow end_run failed: %s", exc)


# ---------------------------------------------------------------------- trackio
class TrackioLogger:
    """trackio adapter — HF's lightweight wandb-alike. Same call shape
    as WandbLogger but via the trackio module."""

    def __init__(self, cfg: LoggerConfig) -> None:
        import trackio
        self._trackio = trackio
        kwargs: dict[str, Any] = {"project": cfg.project}
        if cfg.run_name:
            kwargs["name"] = cfg.run_name
        if cfg.extra:
            kwargs.update(cfg.extra)
        trackio.init(**kwargs)

    def log_params(self, params: Mapping[str, Any]) -> None:
        try:
            # trackio tracks config via init(config=...); after the run has
            # started the scalar log path is still the canonical way to
            # record static-ish values, so mirror them there with a prefix.
            stamped = {f"config/{k}": v for k, v in params.items()
                       if isinstance(v, (int, float, bool))}
            if stamped:
                self._trackio.log(stamped)
        except Exception as exc:  # pragma: no cover
            log.warning("trackio log_params failed: %s", exc)

    def log_metrics(self, metrics: Mapping[str, float],
                    step: int | None = None) -> None:
        try:
            self._trackio.log(dict(metrics), step=step)
        except Exception as exc:  # pragma: no cover
            log.warning("trackio log_metrics failed: %s", exc)

    def close(self) -> None:
        try:
            self._trackio.finish()
        except Exception as exc:  # pragma: no cover
            log.warning("trackio finish failed: %s", exc)


# ---------------------------------------------------------------------- factory
_BACKENDS: dict[str, type] = {
    "null": NullLogger,
    "wandb": WandbLogger,
    "tensorboard": TensorBoardLogger,
    "mlflow": MLflowLogger,
    "trackio": TrackioLogger,
}


def get_logger(cfg: LoggerConfig) -> MetricsLogger:
    """Build a logger for the configured backend.

    On construction failure (missing dep, bad credentials) the error is
    logged and a ``NullLogger`` is returned so the daemon never refuses
    to start because of an external tool.
    """
    name = (cfg.backend or "null").lower()
    if name == "null":
        return NullLogger()
    cls = _BACKENDS.get(name)
    if cls is None:
        log.warning("unknown metrics backend %r — falling back to NullLogger", name)
        return NullLogger()
    try:
        return cls(cfg)
    except Exception as exc:
        log.warning("metrics backend %r failed to init (%s: %s) — "
                    "falling back to NullLogger",
                    name, type(exc).__name__, exc)
        return NullLogger()
