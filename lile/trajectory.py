"""Append-only trajectory log.

Every interaction (chat completion, training event, feedback, snapshot, merge) is
written here as a JSON line. The log is the source of truth for replay and audit.

We use append-only flat files so that:

* Snapshot reconstruction can read offsets directly (see :mod:`lile.snapshot`).
* The log is human-greppable.
* Crash recovery is trivial (truncate the partial last line, continue).

Records have a stable schema: ``{ts, kind, ...payload}`` where ``ts`` is monotonic
nanoseconds since epoch and ``kind`` is a short tag (``chat``, ``train``,
``feedback``, ``merge``, ``snapshot``).
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


_KIND_VALUES = frozenset(
    {"chat", "train", "feedback", "merge", "snapshot", "restore", "info"}
)


def _now_ns() -> int:
    return time.time_ns()


@dataclass(frozen=True)
class Record:
    """A single trajectory record."""

    ts: int
    kind: str
    payload: dict[str, Any]
    offset: int  # byte offset into the underlying file at write time

    def to_json(self) -> str:
        body = {"ts": self.ts, "kind": self.kind, **self.payload}
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


class TrajectoryLog:
    """Thread-safe append-only JSONL log.

    Designed to be opened once at daemon startup and shared across handlers. The
    underlying file handle stays open in append mode; ``fsync`` is called on
    explicit ``flush`` and on close (not per-write — it would crater throughput).
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Binary append; we own the encoding so we can compute exact byte
        # offsets that survive Python's text-mode rewrites of newlines.
        self._fh: io.BufferedWriter = open(self.path, "ab", buffering=0)
        self._lock = threading.Lock()
        self._closed = False

    def append(self, kind: str, **payload: Any) -> Record:
        if kind not in _KIND_VALUES:
            raise ValueError(f"Unknown trajectory kind: {kind!r}")
        ts = _now_ns()
        with self._lock:
            if self._closed:
                raise RuntimeError("TrajectoryLog is closed")
            offset = self._fh.tell()
            line = json.dumps(
                {"ts": ts, "kind": kind, **payload},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            self._fh.write(line)
        return Record(ts=ts, kind=kind, payload=payload, offset=offset)

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            finally:
                self._fh.close()
                self._closed = True

    def current_offset(self) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("TrajectoryLog is closed")
            return self._fh.tell()

    @staticmethod
    def iter_from(
        path: str | os.PathLike[str],
        start_offset: int = 0,
        *,
        kinds: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    ) -> Iterator[Record]:
        """Yield records starting from ``start_offset`` (inclusive).

        ``kinds`` (optional) filters to records whose ``kind`` is in the set.
        Used by the idle replay scheduler to scan only ``feedback`` / ``chat``
        records without paying the deserialization cost on the rest.
        """
        path = Path(path)
        if not path.exists():
            return
        kind_set = frozenset(kinds) if kinds is not None else None
        with open(path, "rb") as f:
            f.seek(start_offset)
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # Partial trailing line from a crash mid-write; ignore.
                    break
                obj = json.loads(line.decode("utf-8"))
                if kind_set is not None and obj["kind"] not in kind_set:
                    continue
                yield Record(
                    ts=obj["ts"],
                    kind=obj["kind"],
                    payload={k: v for k, v in obj.items() if k not in {"ts", "kind"}},
                    offset=offset,
                )

    # Context manager sugar so callers can use `with TrajectoryLog(...) as log:`.
    def __enter__(self) -> "TrajectoryLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
