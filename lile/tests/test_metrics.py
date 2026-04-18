"""Prometheus metrics — registry shape, gauge sampling, counter hooks.

Covers `lile/metrics.py`:

- Counters (requests_total, train_steps_total, feedback_events_total,
  queue_dropped_total, replay_enqueued_total).
- Histograms (step_latency_ms, generate_latency_ms, objective_loss).
- Lazy gauges (queue_depth, commit_cursor, merges_applied, trajectory_bytes,
  snapshots_bytes, snapshots_count, shutting_down) sampled from a bound
  Controller-shaped object at scrape time.
- `render_prometheus()` returns valid OpenMetrics/Prometheus text exposition
  format bytes.

Run with: pytest lile/tests/test_metrics.py
"""
from __future__ import annotations

import pytest

# NOTE: these tests lazy-import ``lile.metrics`` inside their bodies, which
# pulls prometheus_client — not available in the torchless cpu_only CI
# bucket. Leave unmarked so the conftest filter excludes the whole file
# rather than collecting and failing at test-run time.


# ---------------------------------------------------------------- fixtures


class _FakeQueue:
    def __init__(self, *, qsize: int = 0, committed: int = 0) -> None:
        self._q = type("_Q", (), {"qsize": lambda self: qsize})()
        self.committed = committed


class _FakeState:
    def __init__(self, *, merges_applied: int = 0) -> None:
        self.merges_applied = merges_applied


class _FakeTrajectory:
    def __init__(self, *, size: int = 0) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeController:
    """Shape-compatible stand-in for the gauge sampler.

    Exposes the attributes the gauge collector reads: ``queue``, ``state``,
    ``trajectory``, ``snapshots.root``, ``_shutting_down`` (optional).
    """

    def __init__(
        self,
        *,
        qsize: int = 0,
        committed: int = 0,
        merges: int = 0,
        traj_bytes: int = 0,
        snapshots_root=None,
        shutting_down: bool = False,
    ) -> None:
        self.queue = _FakeQueue(qsize=qsize, committed=committed)
        self.state = _FakeState(merges_applied=merges)
        self.trajectory = _FakeTrajectory(size=traj_bytes)
        self.snapshots = type("_S", (), {"root": snapshots_root})()
        self._shutting_down = shutting_down


# ---------------------------------------------------------------- module shape


def test_module_exposes_render_prometheus_and_registry():
    from lile import metrics

    assert hasattr(metrics, "render_prometheus")
    assert hasattr(metrics, "REGISTRY")
    assert hasattr(metrics, "bind_controller")
    assert hasattr(metrics, "record_request")
    assert hasattr(metrics, "record_train_step")
    assert hasattr(metrics, "record_feedback_event")
    assert hasattr(metrics, "record_generate_latency")


def test_render_prometheus_returns_bytes_with_text_format_header():
    from lile import metrics

    out = metrics.render_prometheus()
    assert isinstance(out, (bytes, bytearray))
    text = bytes(out).decode("utf-8")
    # Every metric family is emitted with a `# HELP` and `# TYPE` line.
    assert "# HELP lile_requests_total" in text
    assert "# TYPE lile_requests_total counter" in text
    assert "# HELP lile_queue_depth" in text
    assert "# TYPE lile_queue_depth gauge" in text


# ---------------------------------------------------------------- counters


def test_record_request_increments_counter():
    from lile import metrics

    before = _counter_value(metrics.render_prometheus(),
                            "lile_requests_total",
                            {"route": "/v1/train", "status": "200"})
    metrics.record_request(route="/v1/train", status=200)
    after = _counter_value(metrics.render_prometheus(),
                           "lile_requests_total",
                           {"route": "/v1/train", "status": "200"})
    assert after == before + 1.0


def test_record_train_step_increments_counter_and_histogram():
    from lile import metrics

    before = _counter_value(metrics.render_prometheus(),
                            "lile_train_steps_total", {"objective": "sft"})
    metrics.record_train_step(objective="sft", latency_s=0.123, loss=1.5)
    text = metrics.render_prometheus().decode("utf-8")
    after = _counter_value(text.encode(), "lile_train_steps_total", {"objective": "sft"})
    assert after == before + 1.0
    # Histogram count bumped.
    assert "lile_step_latency_ms_count" in text


def test_record_feedback_event_counter():
    from lile import metrics

    before = _counter_value(metrics.render_prometheus(),
                            "lile_feedback_events_total", {"kind": "binary"})
    metrics.record_feedback_event(kind="binary")
    after = _counter_value(metrics.render_prometheus(),
                           "lile_feedback_events_total", {"kind": "binary"})
    assert after == before + 1.0


def test_record_generate_latency_observes_histogram():
    from lile import metrics

    metrics.record_generate_latency(stream=False, latency_s=0.25)
    metrics.record_generate_latency(stream=True, latency_s=0.05)
    text = metrics.render_prometheus().decode("utf-8")
    assert 'lile_generate_latency_ms_count{stream="false"}' in text
    assert 'lile_generate_latency_ms_count{stream="true"}' in text


# ---------------------------------------------------------------- gauges (lazy)


def test_gauge_collector_samples_live_state(tmp_path):
    from lile import metrics

    # Put two fake snapshot dirs on disk so the snapshots gauges are non-zero.
    snap_root = tmp_path / "snapshots"
    (snap_root / "v1").mkdir(parents=True)
    (snap_root / "v2").mkdir()
    (snap_root / "v1" / "x.bin").write_bytes(b"0" * 100)

    ctrl = _FakeController(
        qsize=3,
        committed=42,
        merges=7,
        traj_bytes=1024,
        snapshots_root=snap_root,
        shutting_down=False,
    )
    metrics.bind_controller(ctrl)
    text = metrics.render_prometheus().decode("utf-8")

    assert _sample_value(text, "lile_queue_depth") == 3.0
    assert _sample_value(text, "lile_commit_cursor") == 42.0
    assert _sample_value(text, "lile_merges_applied") == 7.0
    assert _sample_value(text, "lile_trajectory_bytes") == 1024.0
    assert _sample_value(text, "lile_snapshots_count") == 2.0
    assert _sample_value(text, "lile_snapshots_bytes") == 100.0
    assert _sample_value(text, "lile_shutting_down") == 0.0


def test_shutting_down_gauge_flips_when_flag_set(tmp_path):
    from lile import metrics

    ctrl = _FakeController(snapshots_root=tmp_path, shutting_down=True)
    metrics.bind_controller(ctrl)
    text = metrics.render_prometheus().decode("utf-8")
    assert _sample_value(text, "lile_shutting_down") == 1.0


def test_unbound_gauges_emit_zero(tmp_path):
    from lile import metrics

    # Reset any bound controller.
    metrics.bind_controller(None)
    text = metrics.render_prometheus().decode("utf-8")
    # Gauges are still declared (text appears) but samples are zero.
    assert _sample_value(text, "lile_queue_depth") == 0.0
    assert _sample_value(text, "lile_shutting_down") == 0.0


# ---------------------------------------------------------------- /metrics route


def test_metrics_route_emits_prom_text_format():
    from fastapi.testclient import TestClient

    from lile.server_errors import register_error_handlers

    # We mount just the /metrics route on a bare app to avoid loading a
    # model. The real route lives in server.py and is identical in shape.
    from fastapi import FastAPI
    from fastapi.responses import Response

    from lile import metrics

    app = FastAPI()
    register_error_handlers(app)

    @app.get("/metrics")
    async def _metrics():  # pragma: no cover — handler body checked via client
        return Response(metrics.render_prometheus(),
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    with TestClient(app) as client:
        r = client.get("/metrics")

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text.startswith("#") or r.text.lstrip().startswith("#")
    assert "lile_requests_total" in r.text


def test_metrics_route_wired_in_server():
    """The real server.py must expose /metrics. Checked without triggering
    startup so we don't load a model in a cpu_only test."""
    import pathlib
    import tempfile

    from lile.config import ServeConfig
    from lile.server import create_app

    cfg = ServeConfig(data_dir=pathlib.Path(tempfile.mkdtemp(prefix="lile_m_")))
    app = create_app(cfg)
    # Enumerate registered paths without entering a TestClient context, which
    # would fire the startup event and call Controller.start → model load.
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/metrics" in paths


# ---------------------------------------------------------------- helpers


def _counter_value(rendered: bytes, name: str, labels: dict[str, str]) -> float:
    """Parse a single counter sample out of the rendered text format."""
    text = rendered.decode("utf-8") if isinstance(rendered, (bytes, bytearray)) else rendered
    # prometheus_client emits both `<name>` (current) and `<name>_total` for
    # counters, depending on version. Accept either.
    return max(
        _sample_value(text, name, labels),
        _sample_value(text, name + "_created" if name.endswith("_total") else name, labels),
        _sample_value(text, name[:-6] if name.endswith("_total") else name, labels),
    )


def _sample_value(text: str, name: str, labels: dict[str, str] | None = None) -> float:
    """Return the sample value for ``<name>{<labels>} <value>`` or 0.0 if absent."""
    label_str = ""
    if labels:
        # prom text format labels are alphabetized; match permissively by
        # checking every `name{...}` line and parsing.
        pass
    best = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if not line.startswith(name):
            continue
        # Split at first whitespace → "<name>{labels}" <value>
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        label_part = parts[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if labels:
            if not all(f'{k}="{v}"' in label_part for k, v in labels.items()):
                continue
            # Tighten: name must exactly match the prefix before `{`.
            prefix = label_part.split("{", 1)[0]
            if prefix != name:
                continue
        else:
            if label_part != name:
                continue
        best = value  # take last occurrence — prom_client emits one sample per line
    return best
