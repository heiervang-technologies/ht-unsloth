"""ARC-AGI-3 single-turn runner.

Drives one task end-to-end: build the prompt, call the controller's
async ``generate``, parse a grid out of the response, exact-match it
against the gold test output. Parse failures are reported as
``correct=False`` with ``parse_error`` populated; the runner never
raises into the caller.

The agentic multi-turn loop is layered on top of this same primitive.
Until that arrives, this is the canonical ARC adapter for both the
eval harness and the RLVR scheduler.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from .loader import Grid, Task
from .prompts import build_prompt


class RunResult(TypedDict):
    task_id: str
    predicted: Grid | None
    expected: Grid
    correct: bool
    parse_error: str | None


# Match a fenced ```json ... ``` block; tolerate the bare ``` fence too.
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)
# Fallback: a top-level list-of-lists somewhere in the text.
_LIST_OF_LISTS = re.compile(r"\[\s*\[.*?\]\s*\]", re.DOTALL)


def _validate_grid(parsed: Any) -> Grid:
    """Return ``parsed`` as a Grid or raise ``ValueError``."""
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("not a non-empty list")
    width: int | None = None
    for i, row in enumerate(parsed):
        if not isinstance(row, list) or not row:
            raise ValueError(f"row {i} is not a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"ragged grid (row {i} width {len(row)} != {width})")
        for j, v in enumerate(row):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"cell ({i},{j}) is not an int")
            if v < 0 or v > 9:
                raise ValueError(f"cell ({i},{j})={v} out of 0..9")
    return parsed  # type: ignore[return-value]


def parse_grid(text: str) -> Grid:
    """Pull a Grid out of ``text``. Raises ValueError on any failure."""
    if not text:
        raise ValueError("empty response")
    m = _JSON_FENCE.search(text)
    candidate = m.group(1) if m else None
    # Also try a raw list-of-lists slice if no fence matched.
    if candidate is None:
        m2 = _LIST_OF_LISTS.search(text)
        if m2 is None:
            raise ValueError("no JSON fence and no list-of-lists found")
        candidate = m2.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"json decode failed: {exc.msg}") from exc
    return _validate_grid(parsed)


def _coerce_response_text(result: Any) -> str:
    """Pull the most informative text out of ``controller.generate``'s dict.

    The contract returns ``{"response", "reasoning_content", "raw"}``. We
    prefer ``response`` (the post-thinking content), fall back to
    ``reasoning_content`` (some models emit the JSON inside the thinking
    span), and finally ``raw``. A bare string is also accepted so this
    function is usable with monkey-patched fakes.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("response", "reasoning_content", "raw"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return ""


async def run_task(
    controller: Any,
    task: Task,
    *,
    max_tokens: int = 800,
) -> RunResult:
    """Run ``task`` against ``controller`` and exact-match the prediction.

    ``controller`` only needs an async ``.generate(messages, **kwargs)``
    method that returns a dict with ``response`` / ``reasoning_content``
    / ``raw`` (or a bare string). This loose duck-type lets unit tests
    monkeypatch a tiny stand-in without importing torch.
    """
    prompt = build_prompt(task)
    messages = [{"role": "user", "content": prompt}]
    expected = task["test_output"]
    try:
        result = await controller.generate(
            messages=messages,
            temperature=0.0,
            max_new_tokens=max_tokens,
        )
    except Exception as exc:
        return {
            "task_id": task["id"],
            "predicted": None,
            "expected": expected,
            "correct": False,
            "parse_error": f"generate raised: {exc!r}",
        }
    text = _coerce_response_text(result)
    try:
        predicted = parse_grid(text)
    except ValueError as exc:
        return {
            "task_id": task["id"],
            "predicted": None,
            "expected": expected,
            "correct": False,
            "parse_error": str(exc),
        }
    return {
        "task_id": task["id"],
        "predicted": predicted,
        "expected": expected,
        "correct": predicted == expected,
        "parse_error": None,
    }
