"""pytest conftest for the lile test suite.

Most of this suite imports ``torch`` / ``unsloth`` (directly or through
``lile.controller`` / ``lile.state``) at module scope. On a CI runner
without those heavy deps we still want the ``eval`` harness slice plus
the few stdlib-only slices to collect cleanly, so when torch is missing
we fall back to a whitelist of files known to import without it.

The ``eval`` bucket only needs stdlib + pytest — ``lile.teach.eval`` is
urllib-based and has no torch dependency.

Self-validation
---------------

When torch is absent we also validate that the ``_TORCHLESS_OK`` whitelist
actually matches what's collectible on this runner. A file that declares
``pytestmark = pytest.mark.cpu_only`` but is missing from the whitelist
*and* imports cleanly without torch would silently fail to collect in the
cpu_only CI bucket — exactly the regression PR #36 shipped. The check
raises at conftest load so the miss is loud, not a silent skip.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def _missing(*modules: str) -> bool:
    return any(importlib.util.find_spec(m) is None for m in modules)


collect_ignore_glob: list[str] = []

# Files that can be imported on a torchless runner. Everything else in
# ``lile/tests/`` will be skipped at collection time when torch is absent.
_TORCHLESS_OK = {
    "test_errors.py",               # lazy lile.errors import inside tests
    "test_error_middleware.py",     # same; also uses FastAPI/TestClient
    "test_eval_harness.py",         # harness smoke — urllib-only
    "test_logging_backends.py",     # lile.logging_backends is lazy on heavy deps
    "test_queue_cursor.py",         # lile.queue is pure Python
    "test_queue_graceful_drain.py", # lile.queue drain path, asyncio-only
    "test_reasoning.py",            # lile.reasoning is pure Python
    "test_replay_streams.py",       # replay_streams scaffold — stdlib-only imports
    "test_trajectory_tail.py",      # lile.trajectory is pure Python
    "test_whitelist_consistency.py", # self-validation of _TORCHLESS_OK
    "test_commits_sse_stream.py",   # lile.commit_stream is pure asyncio + FastAPI
    "conftest.py",
    "__init__.py",
}


def _ast_has_cpu_only_pytestmark(tree: ast.AST) -> bool:
    """Return True if ``tree`` has a module-level ``pytestmark`` assignment
    that references ``pytest.mark.cpu_only``.

    Matches the three shapes found in the suite today:

    - ``pytestmark = pytest.mark.cpu_only``
    - ``pytestmark = [pytest.mark.cpu_only, pytest.mark.eval]``
    - ``pytestmark = (pytest.mark.cpu_only,)``

    We walk the RHS looking for any ``Attribute`` named ``cpu_only``;
    that's stricter than a text grep (which would also match comments)
    and looser than trying to pattern-match every legal list shape.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "pytestmark":
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Attribute) and sub.attr == "cpu_only":
                        return True
    return False


def _cpu_only_whitelist_violations(
    tests_dir: Path, whitelist: set[str]
) -> list[str]:
    """Return test files that claim ``cpu_only`` but aren't whitelisted yet
    *and* import cleanly in the current (torchless) interpreter.

    The import probe is load-bearing. A file may carry ``cpu_only`` because
    the author intends it for the cpu bucket, but still import heavy deps
    at module scope (in which case filtering it out is correct, and adding
    it to the whitelist would break CI). Only files that both advertise
    cpu_only *and* would actually collect here are violations.

    Boundary: this only checks *module-level* importability. A file whose
    module scope is stdlib-only but whose test bodies lazy-import torch or
    prometheus_client will pass the probe and still fail at test-run time.
    That's a misclassification bug in the test file itself (the
    ``cpu_only`` marker is lying); the fix is to drop the marker or make
    the dep available to the cpu_only CI env, not to extend this probe.
    """
    violations: list[str] = []
    for p in sorted(tests_dir.glob("test_*.py")):
        if p.name in whitelist:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        if not _ast_has_cpu_only_pytestmark(tree):
            continue
        spec = importlib.util.spec_from_file_location(
            f"_lile_whitelist_probe_{p.stem}", p
        )
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ImportError:
            # Heavy-dep file; filtering it out is correct behavior.
            continue
        except Exception:
            # Imports cleanly enough to reach some other error — still a
            # whitelist miss, because the test module surface exists.
            pass
        violations.append(p.name)
    return violations


if _missing("torch"):
    here = Path(__file__).parent
    for p in here.glob("*.py"):
        if p.name not in _TORCHLESS_OK:
            collect_ignore_glob.append(p.name)

    _violations = _cpu_only_whitelist_violations(here, _TORCHLESS_OK)
    if _violations:
        raise AssertionError(
            "lile/tests/conftest.py: _TORCHLESS_OK is out of sync with the "
            "cpu_only-marked test files.\n"
            f"Files marked cpu_only that import cleanly without torch but "
            f"are not in _TORCHLESS_OK: {_violations}\n"
            "Fix: add each file to _TORCHLESS_OK (if it really is stdlib-only) "
            "or drop the cpu_only marker (if it needs torch to run)."
        )
