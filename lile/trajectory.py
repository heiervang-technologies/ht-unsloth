"""Append-only JSONL trajectory log.

Every response handed to a user gets a `response_id`. Feedback routes through
a single endpoint; the daemon chooses the objective later. All training
material and all feedback lands here first, always — this is the canonical
record that lets us replay, re-weight, or apply new methods to old feedback.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


def new_response_id() -> str:
    return "r_" + uuid.uuid4().hex[:16]


class TrajectoryLog:
    """Thread-safe append-only JSONL writer with offset-based checkpointing."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.touch()

    # ------------------------------------------------------------------ writers
    def append_raw(self, payload: dict[str, Any]) -> int:
        """Append a pre-shaped JSON record with no kind/ts stamping.

        Intended for fixtures, migration tools, and tests that need to control
        ``ts`` or write pre-built records verbatim. Production code should use
        ``log_event`` (or the kind-specific wrappers), which enforce the
        canonical ``{kind, ts, ...}`` shape.
        """
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("ab") as f:
                offset = f.tell()
                f.write(line.encode("utf-8") + b"\n")
                f.flush()
        return offset

    def log_event(self, kind: str, data: dict[str, Any]) -> int:
        """Append a `{kind, ts, ...data}` line. Returns byte offset of the new line."""
        payload = {"kind": kind, "ts": time.time(), **data}
        return self.append_raw(payload)

    def log_inference(self, response_id: str, prompt: str, response: str,
                      model_fingerprint: str) -> int:
        return self.log_event("inference", {
            "response_id": response_id,
            "prompt": prompt,
            "response": response,
            "model_fingerprint": model_fingerprint,
        })

    def log_feedback(self, response_id: str, kind: str, **fields: Any) -> int:
        return self.log_event("feedback", {
            "response_id": response_id,
            "feedback_kind": kind,
            **fields,
        })

    def log_train(self, batch_id: str, objective: str, loss: float,
                  batch_size: int, commit_token: int) -> int:
        return self.log_event("train_step", {
            "batch_id": batch_id,
            "objective": objective,
            "loss": float(loss),
            "batch_size": int(batch_size),
            "commit_token": int(commit_token),
        })

    # ------------------------------------------------------------------ readers
    def iter_events(self, since_offset: int = 0) -> Iterable[dict[str, Any]]:
        with self.path.open("rb") as f:
            f.seek(since_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("malformed trajectory line at offset %d", f.tell())

    def iter_with_offsets(
        self,
        since_offset: int = 0,
        kinds: set[str] | None = None,
    ) -> Iterable[tuple[int, dict[str, Any]]]:
        """Yield ``(line_start_offset, record)`` so callers can deduplicate by offset.

        Used by the idle replay scheduler (§T4.1) to track per-record replay
        counts: the line's byte offset is its stable identifier for the
        lifetime of the log file. ``kinds`` restricts yielded records to the
        listed event kinds (e.g. ``{"feedback"}``); ``None`` yields all.
        """
        with self.path.open("rb") as f:
            f.seek(since_offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    log.warning("malformed trajectory line at offset %d", pos)
                    continue
                if kinds is not None and record.get("kind") not in kinds:
                    continue
                yield pos, record

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        all_events = list(self.iter_events())
        return all_events[-n:]

    def size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0
