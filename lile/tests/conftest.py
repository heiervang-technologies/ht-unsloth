"""pytest conftest for the lile test suite.

Most of this suite imports ``torch`` / ``unsloth`` (directly or through
``lile.controller`` / ``lile.state``) at module scope. On a CI runner
without those heavy deps we still want the ``eval`` harness slice plus
the few stdlib-only slices to collect cleanly, so when torch is missing
we fall back to a whitelist of files known to import without it.

The ``eval`` bucket only needs stdlib + pytest — ``lile.teach.eval`` is
urllib-based and has no torch dependency.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _missing(*modules: str) -> bool:
    return any(importlib.util.find_spec(m) is None for m in modules)


collect_ignore_glob: list[str] = []

# Files that can be imported on a torchless runner. Everything else in
# ``lile/tests/`` will be skipped at collection time when torch is absent.
_TORCHLESS_OK = {
    "test_errors.py",          # lazy lile.errors import inside tests
    "test_error_middleware.py", # same; also uses FastAPI/TestClient
    "test_eval_harness.py",    # harness smoke — urllib-only
    "test_queue_cursor.py",    # lile.queue is pure Python
    "test_reasoning.py",       # lile.reasoning is pure Python
    "test_trajectory_tail.py", # lile.trajectory is pure Python
    "conftest.py",
    "__init__.py",
}

if _missing("torch"):
    here = Path(__file__).parent
    for p in here.glob("*.py"):
        if p.name not in _TORCHLESS_OK:
            collect_ignore_glob.append(p.name)
