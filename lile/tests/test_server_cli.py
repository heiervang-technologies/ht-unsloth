"""CLI arg parsing for `python -m lile.server ...`.

Pins the contract that Studio's /capsule/start subprocess.Popen relies on —
specifically that ``--port`` and ``--model`` override the ServeConfig defaults.
Before this fix, flags were silently ignored: the daemon always listened on
the ServeConfig default port, so Studio's proxy (configured via ``LILE_PORT``)
could never reach it.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_only


def test_defaults_match_serve_config():
    from lile.config import ServeConfig
    from lile.server import _parse_cli_args

    default = ServeConfig()
    cfg = _parse_cli_args([])
    assert cfg.host == default.host
    assert cfg.port == default.port
    assert cfg.model == default.model
    assert cfg.load_in_4bit is default.load_in_4bit


def test_port_flag_overrides_default():
    from lile.server import _parse_cli_args

    cfg = _parse_cli_args(["--port", "8766"])
    assert cfg.port == 8766


def test_host_and_model_flags_override_defaults():
    from lile.server import _parse_cli_args

    cfg = _parse_cli_args(["--host", "0.0.0.0", "--model", "foo/bar"])
    assert cfg.host == "0.0.0.0"
    assert cfg.model == "foo/bar"


def test_data_dir_becomes_path(tmp_path):
    from lile.server import _parse_cli_args

    cfg = _parse_cli_args(["--data-dir", str(tmp_path)])
    assert cfg.data_dir == tmp_path


def test_toggle_flags():
    from lile.server import _parse_cli_args

    cfg = _parse_cli_args(["--no-4bit", "--idle-replay", "--frozen-ref"])
    assert cfg.load_in_4bit is False
    assert cfg.idle_replay is True
    assert cfg.frozen_ref is True
