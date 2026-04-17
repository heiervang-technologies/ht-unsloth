"""Tests for lile/logging_backends.py.

Covers the pieces that can be verified without optional third-party deps:
  - ``flatten_scalars`` handles nesting, booleans, and drops non-scalars.
  - ``NullLogger`` satisfies the ``MetricsLogger`` protocol and is a
    total no-op (calling every method raises nothing).
  - ``get_logger`` with ``backend="null"`` returns a NullLogger.
  - ``get_logger`` with an unknown backend name logs a warning and
    returns a NullLogger rather than raising.
  - ``get_logger`` with a real backend name but missing dep falls back
    to NullLogger gracefully (import failure is caught).

Run with: python -m lile.tests.test_logging_backends
"""
from __future__ import annotations

import sys

import pytest

from lile.logging_backends import (
    LoggerConfig,
    MetricsLogger,
    NullLogger,
    flatten_scalars,
    get_logger,
)

pytestmark = pytest.mark.cpu_only


def test_flatten_scalars_basic():
    out = flatten_scalars({"loss": 0.42, "grad_clipped": True, "step": 7})
    assert out == {"loss": 0.42, "grad_clipped": 1.0, "step": 7.0}


def test_flatten_scalars_nested_uses_dot_path():
    out = flatten_scalars({
        "loss": 0.5,
        "batch": {"kl": {"loss": 0.1, "mean": 0.05}},
    })
    assert out == {"loss": 0.5, "batch.kl.loss": 0.1, "batch.kl.mean": 0.05}


def test_flatten_scalars_drops_non_scalar_values():
    out = flatten_scalars({
        "loss": 0.5,
        "source": "adapter_disabled",  # string → drop
        "tensor_like": object(),        # object → drop
        "none_val": None,               # None → drop
    })
    assert out == {"loss": 0.5}


def test_null_logger_is_total_noop_and_protocol_compatible():
    logger: MetricsLogger = NullLogger()
    # Every method must accept the documented args and return without error.
    logger.log_params({"a": 1, "b": "two"})
    logger.log_metrics({"loss": 0.5}, step=3)
    logger.log_metrics({"loss": 0.5})          # step=None
    logger.close()
    # isinstance runtime check against the Protocol.
    assert isinstance(logger, MetricsLogger)


def test_get_logger_null_returns_nulllogger():
    logger = get_logger(LoggerConfig(backend="null"))
    assert isinstance(logger, NullLogger)


def test_get_logger_unknown_backend_falls_back_to_null(caplog):
    with caplog.at_level("WARNING"):
        logger = get_logger(LoggerConfig(backend="definitely-not-a-real-backend"))
    assert isinstance(logger, NullLogger)
    assert any("unknown metrics backend" in rec.message for rec in caplog.records)


def test_get_logger_missing_dep_falls_back_to_null(monkeypatch, caplog):
    """Forcing the wandb import to fail proves the factory doesn't crash
    lile startup when the user selects a backend whose library isn't
    installed in this environment."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("wandb not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with caplog.at_level("WARNING"):
        logger = get_logger(LoggerConfig(backend="wandb", project="unit-test"))
    assert isinstance(logger, NullLogger)
    assert any("failed to init" in rec.message for rec in caplog.records)


def main() -> int:
    test_flatten_scalars_basic()
    test_flatten_scalars_nested_uses_dot_path()
    test_flatten_scalars_drops_non_scalar_values()
    test_null_logger_is_total_noop_and_protocol_compatible()
    test_get_logger_null_returns_nulllogger()
    # Tests that need caplog/monkeypatch are pytest-only.
    print("[test_logging_backends] standalone block OK — run via pytest for full coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
