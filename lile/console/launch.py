"""Launcher: lile daemon on Qwen3.5-9B for human QA trial."""
from __future__ import annotations
import os, sys, logging
from pathlib import Path
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unsloth  # noqa: F401 — must come before transformers
from lile.config import ServeConfig
from lile.server import serve

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")

cfg = ServeConfig(
    model="unsloth/Qwen3.5-9B",
    max_seq_length=2048,
    host=os.environ.get("LILE_HOST", "127.0.0.1"),
    port=int(os.environ.get("LILE_PORT", "8768")),
    idle_replay=True,
    frozen_ref=False,
    # Dev defaults: hot reload + crash-safe state. Override via env.
    dev_autoreload=os.environ.get("LILE_DEV_AUTORELOAD", "1") == "1",
    autosave_on_exit=os.environ.get("LILE_AUTOSAVE_ON_EXIT", "1") == "1",
    autoload_on_boot=os.environ.get("LILE_AUTOLOAD_ON_BOOT", "1") == "1",
)
print(f"[launch] starting lile on http://{cfg.host}:{cfg.port} with {cfg.model}")
serve(cfg)
