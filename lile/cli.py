"""CLI argument parsing for ``python -m lile.server``.

Lives outside ``lile.server`` so it can be imported (and tested) without
pulling in uvicorn / transformers / torch — keeping the cpu_only / torchless
test runner happy.
"""
from __future__ import annotations

import argparse
import pathlib

from .config import ServeConfig


def parse_cli_args(argv: list[str] | None = None) -> ServeConfig:
    """Parse argv into a ServeConfig, overriding only fields the user set."""
    cfg = ServeConfig()
    p = argparse.ArgumentParser(prog="lile.server")
    p.add_argument("--host", default=cfg.host)
    p.add_argument("--port", type=int, default=cfg.port)
    p.add_argument("--model", default=cfg.model)
    p.add_argument("--data-dir", type=pathlib.Path, default=cfg.data_dir)
    p.add_argument("--max-seq-length", type=int, default=cfg.max_seq_length)
    p.add_argument("--lora-rank", type=int, default=cfg.lora_rank)
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   default=cfg.load_in_4bit)
    p.add_argument("--idle-replay", dest="idle_replay", action="store_true",
                   default=cfg.idle_replay)
    p.add_argument("--frozen-ref", dest="frozen_ref", action="store_true",
                   default=cfg.frozen_ref)
    args = p.parse_args(argv)
    cfg.host = args.host
    cfg.port = args.port
    cfg.model = args.model
    cfg.data_dir = args.data_dir
    cfg.max_seq_length = args.max_seq_length
    cfg.lora_rank = args.lora_rank
    cfg.load_in_4bit = args.load_in_4bit
    cfg.idle_replay = args.idle_replay
    cfg.frozen_ref = args.frozen_ref
    return cfg


# Back-compat alias — older tests / external callers may still reach for this.
_parse_cli_args = parse_cli_args
