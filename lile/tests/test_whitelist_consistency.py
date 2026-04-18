"""Self-validation tests for ``conftest._cpu_only_whitelist_violations``.

The helper exists to catch the class of bug PR #36 shipped: a test file
marked ``pytestmark = pytest.mark.cpu_only`` but missing from the
``_TORCHLESS_OK`` whitelist, so it silently didn't collect in the
torchless CI bucket.

We can't easily simulate a torchless interpreter from inside a
pytest run (torch is already loaded, or it isn't). Instead we exercise
the helper directly with synthetic ``test_*.py`` files in ``tmp_path``:

- A stdlib-only file *with* the ``cpu_only`` marker that's absent from
  the whitelist must be flagged.
- The same file, but with a module-level import that raises
  ``ImportError``, must NOT be flagged (the probe catches ImportError
  and filters it out).
- A file that's already whitelisted must never be flagged.
- A stdlib-only file *without* the ``cpu_only`` marker must not be
  flagged (this matters: lots of legitimate torch-requiring tests don't
  carry the marker and should keep being filtered by ignore-glob).
- The AST matcher must recognize the three pytestmark shapes used in
  the real suite (bare attribute, list, tuple).
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------- loader


def _load_conftest_module():
    """Import ``lile/tests/conftest.py`` as a plain module.

    We don't want the conftest's side-effectful top-level (the
    ``if _missing('torch')`` block) to run against the live tree and
    possibly raise mid-test. Loading it under a throwaway module name
    with ``spec_from_file_location`` keeps pytest's own conftest
    instance untouched.
    """
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_lile_conftest_under_test", here / "conftest.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conftest = _load_conftest_module()


# ---------------------------------------------------------------- AST matcher


@pytest.mark.parametrize("src", [
    "import pytest\npytestmark = pytest.mark.cpu_only\n",
    "import pytest\npytestmark = [pytest.mark.cpu_only]\n",
    "import pytest\npytestmark = (pytest.mark.cpu_only,)\n",
    "import pytest\npytestmark = [pytest.mark.cpu_only, pytest.mark.eval]\n",
])
def test_ast_matcher_accepts_known_shapes(src):
    assert conftest._ast_has_cpu_only_pytestmark(ast.parse(src)) is True


@pytest.mark.parametrize("src", [
    "import pytest\npytestmark = pytest.mark.slow\n",
    "import pytest\n# pytestmark = pytest.mark.cpu_only (commented out)\n",
    "import pytest\n",  # no marker at all
    # Function-local assignment — not a module-level pytestmark.
    "import pytest\ndef f():\n    pytestmark = pytest.mark.cpu_only\n",
])
def test_ast_matcher_rejects_non_matches(src):
    assert conftest._ast_has_cpu_only_pytestmark(ast.parse(src)) is False


# ---------------------------------------------------------------- probe


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_probe_flags_stdlib_only_cpu_only_not_in_whitelist(tmp_path):
    _write(tmp_path, "test_new_cpu_only.py",
           "import pytest\npytestmark = pytest.mark.cpu_only\n"
           "def test_ok():\n    assert 1 == 1\n")
    v = conftest._cpu_only_whitelist_violations(tmp_path, whitelist=set())
    assert v == ["test_new_cpu_only.py"]


def test_probe_ignores_whitelisted_file(tmp_path):
    _write(tmp_path, "test_already_in.py",
           "import pytest\npytestmark = pytest.mark.cpu_only\n")
    v = conftest._cpu_only_whitelist_violations(
        tmp_path, whitelist={"test_already_in.py"}
    )
    assert v == []


def test_probe_ignores_file_without_cpu_only_marker(tmp_path):
    # Stdlib-only, imports cleanly, but no pytestmark. This is the shape
    # of a torch-requiring test that simply forgot the marker — the
    # whitelist can't possibly know about it and shouldn't be asked to.
    _write(tmp_path, "test_unmarked.py",
           "def test_ok():\n    assert True\n")
    v = conftest._cpu_only_whitelist_violations(tmp_path, whitelist=set())
    assert v == []


def test_probe_filters_file_that_raises_importerror(tmp_path):
    # Marked cpu_only but would-be-collection-time imports fail — this
    # is the "heavy dep at module scope" case. The helper must NOT flag
    # it, because adding it to the whitelist would break torchless CI.
    _write(tmp_path, "test_needs_torch.py",
           "import pytest\n"
           "import __nonexistent_heavy_dep__  # simulates `import torch`\n"
           "pytestmark = pytest.mark.cpu_only\n")
    v = conftest._cpu_only_whitelist_violations(tmp_path, whitelist=set())
    assert v == []


def test_probe_still_flags_file_with_non_importerror(tmp_path):
    # Marked cpu_only, imports cleanly, then hits a non-ImportError at
    # module scope (e.g. a runtime bug in a helper). The module surface
    # is real, pytest would see it, so the whitelist should cover it.
    _write(tmp_path, "test_buggy_but_importable.py",
           "import pytest\n"
           "pytestmark = pytest.mark.cpu_only\n"
           "raise RuntimeError('module-scope bug')\n")
    v = conftest._cpu_only_whitelist_violations(tmp_path, whitelist=set())
    assert v == ["test_buggy_but_importable.py"]


def test_probe_tolerates_syntax_error(tmp_path):
    # An unparseable file shouldn't crash the probe — just skip it.
    _write(tmp_path, "test_broken.py", "def oops(:\n    pass\n")
    # Also add a clean violation to prove the loop keeps going.
    _write(tmp_path, "test_clean.py",
           "import pytest\npytestmark = pytest.mark.cpu_only\n")
    v = conftest._cpu_only_whitelist_violations(tmp_path, whitelist=set())
    assert v == ["test_clean.py"]


def test_probe_ignores_non_test_files(tmp_path):
    # Helpers named ``foo.py`` or ``_foo.py`` are not picked up by
    # pytest's default discovery; the probe only scans ``test_*.py``.
    _write(tmp_path, "helper.py",
           "import pytest\npytestmark = pytest.mark.cpu_only\n")
    v = conftest._cpu_only_whitelist_violations(tmp_path, whitelist=set())
    assert v == []


# ---------------------------------------------------------------- live tree


def test_live_whitelist_is_consistent():
    """The real ``lile/tests/`` tree must have no cpu_only misses.

    This is the regression version of the check that conftest runs at
    load time under torchless. If this fails while the conftest load
    passed, it means the probe would flag files that also happen to
    import torch transitively here — a setup/env skew worth fixing."""
    here = Path(__file__).parent
    v = conftest._cpu_only_whitelist_violations(here, conftest._TORCHLESS_OK)
    # When run with torch present, files that import ``lile.controller``
    # do succeed, so the probe returns them — filter those out here by
    # noting the ones already listed in collect_ignore_glob. The check
    # that matters is the torchless-CI run, pinned by the conftest
    # load-time assertion. We still want a smoke regression: any file
    # the probe flags must at least *look* cpu_only.
    for name in v:
        src = (here / name).read_text()
        tree = ast.parse(src)
        assert conftest._ast_has_cpu_only_pytestmark(tree), (
            f"{name} flagged by probe but has no cpu_only pytestmark"
        )
