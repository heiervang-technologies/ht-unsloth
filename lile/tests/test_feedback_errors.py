"""Controller-level tests: /v1/feedback raises structured errors.

Pins the pre-fix bug where unknown response_id returned HTTP 200 with a
``{"error": "..."}`` body. After #12 the controller raises
``UnknownResponseIdError`` / ``InvalidInputError`` and the server envelope
handler converts them to 404 / 400 responses.

These tests hit the pure-Python Controller path (no model loaded).

Run with: pytest lile/tests/test_feedback_errors.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lile.config import ServeConfig
from lile.controller import Controller
from lile.errors import InvalidInputError, UnknownResponseIdError

pytestmark = pytest.mark.cpu_only


def _make_controller(tmp_path: Path) -> Controller:
    cfg = ServeConfig(data_dir=tmp_path)
    return Controller(cfg)


def test_unknown_response_id_raises_structured_error(tmp_path):
    c = _make_controller(tmp_path)
    with pytest.raises(UnknownResponseIdError) as ei:
        asyncio.run(c.submit_feedback({
            "response_id": "r_nonexistent",
            "kind": "binary",
        }))
    assert "r_nonexistent" in str(ei.value)


def test_missing_response_id_and_prompt_raises(tmp_path):
    c = _make_controller(tmp_path)
    # No response_id, no prompt — can't possibly identify a prior turn.
    with pytest.raises(UnknownResponseIdError):
        asyncio.run(c.submit_feedback({"kind": "binary"}))


def test_underspecified_kind_raises_invalid_input(tmp_path):
    c = _make_controller(tmp_path)
    # Provide prompt so we bypass the response_id lookup, but pass an
    # unsupported feedback kind so feedback_to_batch returns None.
    with pytest.raises(InvalidInputError) as ei:
        asyncio.run(c.submit_feedback({
            "prompt": "Q",
            "response": "A",
            "kind": "totally_made_up_kind",
        }))
    assert "totally_made_up_kind" in str(ei.value)
