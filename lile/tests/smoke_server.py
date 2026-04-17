"""HTTP smoke test for the lile FastAPI server.

Spawns the server in-process on an ephemeral port, POSTs /v1/train and
/v1/chat/completions with after_commit_token, and verifies the commit-cursor
invariant holds over HTTP (not just over the in-process Controller).

Run with: python -m lile.tests.smoke_server
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import socket
import sys
import tempfile
import threading
import time
from contextlib import closing

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import httpx
import uvicorn

import unsloth  # noqa: F401 — must be imported before transformers

from lile.config import ServeConfig
from lile.server import create_app


def _pick_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    port = _pick_port()
    data_dir = pathlib.Path(tempfile.mkdtemp(prefix="lile_smoke_"))
    cfg = ServeConfig(
        model="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024, lora_rank=8, lora_alpha=16,
        data_dir=data_dir, host="127.0.0.1", port=port,
    )
    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180.0
    with httpx.Client(base_url=base, timeout=180.0) as c:
        while time.time() < deadline:
            try:
                r = c.get("/health")
                if r.status_code == 200 and r.json().get("ok"):
                    break
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        else:
            print("[smoke_server] health never came up", file=sys.stderr)
            server.should_exit = True
            return 1

        print("[smoke_server] health OK")

        # Submit a train batch.
        train_r = c.post("/v1/train", json={
            "objective": "sft",
            "chunk_size": 1,
            "samples": [
                {"prompt": "The zebra's favorite color is",
                 "response": " fuchsia, because zebras love fuchsia."},
                {"prompt": "The zebra's favorite color is",
                 "response": " fuchsia, because zebras love fuchsia."},
            ],
        })
        assert train_r.status_code == 200, train_r.text
        token = train_r.json()["commit_token"]
        print(f"[smoke_server] train submitted, commit_token={token}")

        # Chat with after_commit_token — must block until training commits.
        chat_r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "The zebra's favorite color is"}],
            "max_tokens": 8,
            "temperature": 0.1,
            "after_commit_token": token,
        })
        assert chat_r.status_code == 200, chat_r.text
        body = chat_r.json()
        reply = body["choices"][0]["message"]["content"]
        cursor = body["lile"]["commit_cursor"]
        print(f"[smoke_server] chat reply={reply!r} (cursor={cursor})")
        assert cursor >= token, f"chat ran before commit: cursor={cursor}, token={token}"

        # Trajectory tail must include at least the inference and train_step events.
        tail_r = c.get("/v1/state/trajectory/tail", params={"n": 10})
        assert tail_r.status_code == 200
        kinds = [e["kind"] for e in tail_r.json()["events"]]
        assert "inference" in kinds, kinds
        assert "train_step" in kinds, kinds
        print(f"[smoke_server] trajectory tail kinds={kinds}")

        # Snapshot save + list.
        save_r = c.post("/v1/state/snapshot/save", json={"name": "smoke_snap"})
        assert save_r.status_code == 200
        print(f"[smoke_server] snapshot saved: {save_r.json()}")

        list_r = c.get("/v1/state/snapshots")
        assert "smoke_snap" in list_r.json()["snapshots"]
        print(f"[smoke_server] snapshot list: {list_r.json()['snapshots']}")

    server.should_exit = True
    t.join(timeout=10)
    print("[smoke_server] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
