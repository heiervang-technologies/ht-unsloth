"""ARC-AGI-3 task loader.

Reads a JSON list of tasks from disk; defaults to the frozen subset at
``tasks_v0.json`` next to this module. The frozen subset is hand-written
so the smoke path is hermetic (no network, no live ARC API). Each task
carries a small set of train pairs plus a single test input/output for
exact-match scoring.

The schema is intentionally narrow — just what the smoke runner needs.
The full ARC-AGI-3 agentic format (action histories, env feedback) will
be a superset added in the agentic follow-up.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


Grid = list[list[int]]


class GridPair(TypedDict):
    input: Grid
    output: Grid


class Task(TypedDict):
    id: str
    train_pairs: list[GridPair]
    test_input: Grid
    test_output: Grid


_DEFAULT_PATH = Path(__file__).parent / "tasks_v0.json"


def _validate_grid(g: object, *, where: str) -> Grid:
    if not isinstance(g, list) or not g:
        raise ValueError(f"{where}: expected non-empty list of rows, got {type(g).__name__}")
    width: int | None = None
    for i, row in enumerate(g):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{where}: row {i} is not a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{where}: ragged grid (row {i} width {len(row)} != {width})")
        for j, v in enumerate(row):
            if not isinstance(v, int) or v < 0 or v > 9:
                raise ValueError(f"{where}: cell ({i},{j})={v!r} not an int in 0..9")
    return g  # type: ignore[return-value]


def _validate_task(raw: object, *, idx: int) -> Task:
    if not isinstance(raw, dict):
        raise ValueError(f"task {idx}: expected dict, got {type(raw).__name__}")
    if "id" not in raw or not isinstance(raw["id"], str):
        raise ValueError(f"task {idx}: missing/invalid 'id'")
    pairs_raw = raw.get("train_pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ValueError(f"task {raw['id']}: 'train_pairs' must be a non-empty list")
    pairs: list[GridPair] = []
    for k, p in enumerate(pairs_raw):
        if not isinstance(p, dict) or "input" not in p or "output" not in p:
            raise ValueError(f"task {raw['id']}: train_pair {k} missing input/output")
        pairs.append({
            "input": _validate_grid(p["input"], where=f"task {raw['id']} train_pair[{k}].input"),
            "output": _validate_grid(p["output"], where=f"task {raw['id']} train_pair[{k}].output"),
        })
    return {
        "id": raw["id"],
        "train_pairs": pairs,
        "test_input": _validate_grid(raw.get("test_input"), where=f"task {raw['id']} test_input"),
        "test_output": _validate_grid(raw.get("test_output"), where=f"task {raw['id']} test_output"),
    }


def load_tasks(path: Path | None = None) -> list[Task]:
    """Load and validate ARC tasks from ``path`` (default: frozen subset).

    The default path resolves to ``tasks_v0.json`` next to this module —
    a hand-pinned set of simple transforms (identity, color-swap,
    horizontal reflection) used by the smoke harness.
    """
    src = Path(path) if path is not None else _DEFAULT_PATH
    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{src}: top-level JSON must be a list of tasks")
    return [_validate_task(t, idx=i) for i, t in enumerate(raw)]
