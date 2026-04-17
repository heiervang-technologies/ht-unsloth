"""Prometheus metrics surface for the lile daemon.

Exposes `GET /metrics` in the standard Prometheus text exposition format.
Metrics come from two sources:

- **Counters / histograms** — updated in-line by callers (middleware, the
  Controller's task handler, the chat route). Cheap, hot-path safe.
- **Gauges** — sampled lazily at scrape time via a custom Collector bound to
  the live Controller. This keeps a single source of truth — the gauge value
  is whatever the Controller reports right now, not a shadow counter that
  can drift.

See issue #13 for the full spec.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from prometheus_client.openmetrics.exposition import (
    CONTENT_TYPE_LATEST as OPENMETRICS_CONTENT_TYPE,
    generate_latest as generate_openmetrics,
)
from prometheus_client.registry import Collector
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .middleware import current_request_id

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- registry

# Private registry — do not share with the default `prometheus_client.REGISTRY`
# global so test runs don't accumulate duplicate collectors across pytest
# sessions and so other libs can't clobber our metric families.
REGISTRY = CollectorRegistry(auto_describe=True)


# ---------------------------------------------------------------- counters

_REQUESTS = Counter(
    "lile_requests_total",
    "HTTP requests handled by /v1/* routes, labelled by route and status.",
    labelnames=("route", "status"),
    registry=REGISTRY,
)

_TRAIN_STEPS = Counter(
    "lile_train_steps_total",
    "Training steps committed by the compute-queue worker.",
    labelnames=("objective",),
    registry=REGISTRY,
)

_FEEDBACK_EVENTS = Counter(
    "lile_feedback_events_total",
    "Feedback payloads accepted on /v1/feedback.",
    labelnames=("kind",),
    registry=REGISTRY,
)

_QUEUE_DROPPED = Counter(
    "lile_queue_dropped_total",
    "Compute-queue tasks dropped before commit (reason label: full, shutdown).",
    labelnames=("reason",),
    registry=REGISTRY,
)

_REPLAY_ENQUEUED = Counter(
    "lile_replay_enqueued_total",
    "Batches enqueued by the idle replay scheduler.",
    registry=REGISTRY,
)


# ---------------------------------------------------------------- histograms

# Prometheus convention: base units are seconds (histogram suffix ``_seconds``).
# Buckets cover the ms-to-tens-of-seconds range both train-step and generate
# paths actually exercise.
_LATENCY_SECONDS_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)

_STEP_LATENCY = Histogram(
    "lile_step_latency_seconds",
    "Wall-clock time spent inside TrainEngine.step.",
    labelnames=("objective",),
    buckets=_LATENCY_SECONDS_BUCKETS,
    registry=REGISTRY,
)

_GENERATE_LATENCY = Histogram(
    "lile_generate_latency_seconds",
    "Wall-clock time for a chat completion. stream=true means time-to-first-token.",
    labelnames=("stream",),
    buckets=_LATENCY_SECONDS_BUCKETS,
    registry=REGISTRY,
)

# Loss histogram — bucket chosen to capture the common LLM NLL range.
_OBJECTIVE_LOSS = Histogram(
    "lile_objective_loss",
    "Per-step loss value, labelled by objective.",
    labelnames=("objective",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    registry=REGISTRY,
)


# ---------------------------------------------------------------- lazy gauges


class _ControllerGaugeCollector(Collector):
    """Reads live state off a bound Controller at scrape time.

    The Controller reference is optional — when unbound every gauge reports
    zero. This keeps the /metrics surface well-defined during daemon startup
    and in unit tests that don't instantiate a Controller.
    """

    def __init__(self) -> None:
        self._controller: Any = None

    def bind(self, controller: Any) -> None:
        self._controller = controller

    def collect(self) -> Iterable[GaugeMetricFamily]:
        c = self._controller

        # Queue depth.
        queue_depth = 0.0
        commit_cursor = 0.0
        merges = 0.0
        traj_bytes = 0.0
        snap_count = 0.0
        snap_bytes = 0.0
        shutting_down = 0.0
        if c is not None:
            try:
                queue_depth = float(c.queue._q.qsize())
            except Exception:  # pragma: no cover — scrape must never crash
                pass
            try:
                commit_cursor = float(getattr(c.queue, "committed", 0))
            except Exception:  # pragma: no cover
                pass
            try:
                if c.state is not None:
                    merges = float(getattr(c.state, "merges_applied", 0))
            except Exception:  # pragma: no cover
                pass
            try:
                traj_bytes = float(c.trajectory.size())
            except Exception:  # pragma: no cover
                pass
            try:
                root: Optional[Path] = getattr(c.snapshots, "root", None)
                if root is not None and Path(root).exists():
                    snap_count, snap_bytes = _snapshot_stats(Path(root))
            except Exception:  # pragma: no cover
                pass
            try:
                shutting_down = 1.0 if getattr(c, "_shutting_down", False) else 0.0
            except Exception:  # pragma: no cover
                pass

        yield GaugeMetricFamily(
            "lile_queue_depth",
            "Compute-queue depth (tasks awaiting the worker).",
            value=queue_depth,
        )
        yield GaugeMetricFamily(
            "lile_commit_cursor",
            "Monotone commit cursor (last committed task token).",
            value=commit_cursor,
        )
        yield GaugeMetricFamily(
            "lile_merges_applied",
            "Number of adapter→residual merges since daemon start.",
            value=merges,
        )
        yield GaugeMetricFamily(
            "lile_trajectory_bytes",
            "Size in bytes of the trajectory JSONL file.",
            value=traj_bytes,
        )
        yield GaugeMetricFamily(
            "lile_snapshots_count",
            "Number of on-disk snapshot directories.",
            value=snap_count,
        )
        yield GaugeMetricFamily(
            "lile_snapshots_bytes",
            "Total bytes used by the snapshots directory tree.",
            value=snap_bytes,
        )
        yield GaugeMetricFamily(
            "lile_shutting_down",
            "1 when the controller is draining for shutdown, 0 otherwise.",
            value=shutting_down,
        )


def _snapshot_stats(root: Path) -> tuple[float, float]:
    """Return (count, total_bytes) walked from the snapshots root."""
    count = 0
    total = 0
    for entry in root.iterdir():
        if entry.is_dir():
            count += 1
            for dirpath, _, filenames in os.walk(entry):
                for fn in filenames:
                    try:
                        total += (Path(dirpath) / fn).stat().st_size
                    except OSError:  # pragma: no cover
                        continue
    return float(count), float(total)


_GAUGES = _ControllerGaugeCollector()
REGISTRY.register(_GAUGES)


# ---------------------------------------------------------------- public API


def bind_controller(controller: Any) -> None:
    """Bind a Controller (or compatible stub) for lazy gauge scraping.

    Pass ``None`` to detach (gauges then report zero).
    """
    _GAUGES.bind(controller)


def render_prometheus() -> bytes:
    """Return the Prometheus text exposition for the lile registry."""
    return generate_latest(REGISTRY)


def render_openmetrics() -> bytes:
    """Return the OpenMetrics text exposition (includes exemplars)."""
    return generate_openmetrics(REGISTRY)


def render_negotiated(accept_header: str | None) -> tuple[bytes, str]:
    """Pick the response format based on the Accept header.

    Prometheus' scraper sends ``Accept: application/openmetrics-text;…`` when
    it wants exemplars; anything else (curl, browser probes) gets plain text.
    Returns ``(body, content_type)``.
    """
    if accept_header and "application/openmetrics-text" in accept_header:
        return render_openmetrics(), OPENMETRICS_CONTENT_TYPE
    return render_prometheus(), CONTENT_TYPE_LATEST


def _exemplar() -> dict[str, str] | None:
    """Bundle the active request id as an exemplar, if one is bound.

    Exemplars render only in the OpenMetrics exposition format; ``generate_latest``
    (plain-text Prometheus) ignores them. Keeping the attribute present on every
    observe is cheap and lets scrapers opt into exemplar-aware queries.
    """
    rid = current_request_id()
    return {"request_id": rid} if rid else None


# ---------------------------------------------------------------- hot-path hooks


def record_request(*, route: str, status: int) -> None:
    """Bump `lile_requests_total{route,status}`. Call once per HTTP response."""
    _REQUESTS.labels(route=route, status=str(status)).inc()


def record_train_step(*, objective: str, latency_s: float, loss: float | None = None) -> None:
    """Bump `lile_train_steps_total{objective}` + observe latency/loss histograms."""
    objective = objective or "unknown"
    exemplar = _exemplar()
    _TRAIN_STEPS.labels(objective=objective).inc()
    _STEP_LATENCY.labels(objective=objective).observe(latency_s, exemplar=exemplar)
    if loss is not None:
        try:
            _OBJECTIVE_LOSS.labels(objective=objective).observe(
                float(loss), exemplar=exemplar,
            )
        except (TypeError, ValueError):  # pragma: no cover — defensive
            pass


def record_feedback_event(*, kind: str) -> None:
    """Bump `lile_feedback_events_total{kind}`."""
    _FEEDBACK_EVENTS.labels(kind=kind or "unknown").inc()


def record_queue_drop(*, reason: str) -> None:
    """Bump `lile_queue_dropped_total{reason}` — reason in {full, shutdown}."""
    _QUEUE_DROPPED.labels(reason=reason).inc()


def record_replay_enqueued(n: int = 1) -> None:
    """Bump `lile_replay_enqueued_total` by n (called by IdleReplayScheduler)."""
    if n <= 0:
        return
    _REPLAY_ENQUEUED.inc(n)


def record_generate_latency(*, stream: bool, latency_s: float) -> None:
    """Observe `lile_generate_latency_seconds{stream}`. For streams this is TTFT."""
    _GENERATE_LATENCY.labels(stream="true" if stream else "false").observe(
        latency_s, exemplar=_exemplar(),
    )


# ---------------------------------------------------------------- middleware


class MetricsMiddleware(BaseHTTPMiddleware):
    """Bump `lile_requests_total{route, status}` once per response.

    Uses ``request.scope["route"].path`` when FastAPI has resolved a route,
    otherwise falls back to the raw path (so `/metrics` scrapes show up
    correctly even before routing).
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        try:
            response: Response = await call_next(request)
            status = response.status_code
        except Exception:
            # Let exception handlers format the envelope; count as 500.
            _REQUESTS.labels(
                route=_resolve_route(request), status="500",
            ).inc()
            raise
        _REQUESTS.labels(route=_resolve_route(request), status=str(status)).inc()
        return response


def _resolve_route(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    # Collapse unmatched routes (scanners, probes, 404s) into a single label
    # value — using ``request.url.path`` would let a hostile client mint
    # unbounded cardinality in ``lile_requests_total``.
    return "unmatched"
