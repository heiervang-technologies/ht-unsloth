"""Jurigged-backed hot reload for the lile daemon.

When enabled, ``jurigged.watch(pattern)`` patches live function code objects
in place on file save. Every caller — including references already captured
in the ``OBJECTIVES`` registry, in closures, or in already-bound methods —
gets the new code without a reimport. The model, optimizer, residual, and
trajectory state stay warm.

Limitations (jurigged cannot patch):
- New module files (requires an import somewhere)
- New top-level names imported via ``from X import Y`` elsewhere
- Class hierarchy changes (new base, new abstract method, MRO shift)
- Module-level side-effectful code (e.g. mutations to the OBJECTIVES dict literal)
- Metaclass / dataclass / decorator shape changes
- **FastAPI route handlers defined as closures inside ``create_app()``** — the
  decorator captures the closure's code object at registration time, so edits
  to the route body don't reach the running app. Edit the function those
  closures *call* (controller methods, objective functions) instead; those
  are module-scoped and hot-reload fine.

For those cases, the process must bounce. ``autosave_on_exit`` + ``autoload_on_boot``
make that bounce lossless: weights, optimizer state, residual, and trajectory
offset are restored byte-exact from the ``_autosave`` snapshot on next boot.

Enable via ``cfg.dev_autoreload=True`` or ``LILE_DEV_AUTORELOAD=1``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Watch the lile package directory. Jurigged's ``to_filter`` treats a
# directory path as "everything under it recursively", which covers
# ``lile/server.py`` at the top level *and* ``lile/objectives/kl.py``
# nested. A glob like ``lile/**/*.py`` would miss the top-level files
# because ``**`` requires at least one intermediate directory.
_DEFAULT_PATTERN = "lile/"


def enable(pattern: str | None = None) -> bool:
    """Start jurigged watching. Returns True if enabled, False otherwise.

    Safe to call when ``jurigged`` isn't installed — logs a warning and
    returns False so the daemon keeps running without hot reload.
    """
    try:
        import jurigged  # type: ignore[import-not-found]
    except ImportError:
        log.warning(
            "dev_autoreload requested but jurigged is not installed. "
            "Install with: uv pip install jurigged  (or: uv sync --extra dev)"
        )
        return False

    pat = pattern or os.environ.get("LILE_AUTORELOAD_PATTERN", _DEFAULT_PATTERN)
    # Resolve relative to the lile package parent so the daemon can be launched
    # from any cwd and jurigged still watches the right tree. A trailing
    # slash on the path tells jurigged's ``to_filter`` to treat it as a
    # directory and match everything beneath it recursively.
    here = Path(__file__).resolve().parents[2]  # repo root
    abs_pattern = str(here / pat)
    if pat.endswith("/") and not abs_pattern.endswith("/"):
        abs_pattern += "/"
    # poll=True because inotify via watchdog silently drops modifications
    # on some filesystems / editor-save patterns (observed on btrfs with
    # atomic-rename saves). Polling is a few ms of scan every debounce
    # window — negligible next to a 9B model step — and reliable.
    jurigged.watch(pattern=abs_pattern, logger=_jurigged_logger, poll=True)
    # Use print so the message survives uvicorn's log reconfiguration.
    print(f"[lile.dev.autoreload] watching {abs_pattern} (poll=True)", flush=True)
    log.info("dev_autoreload enabled — watching %s", abs_pattern)
    return True


def _jurigged_logger(event) -> None:
    """Compact log line per jurigged event — full repr dumps the module namespace."""
    try:
        kind = type(event).__name__
        name = getattr(getattr(event, "defn", None), "name", None)
        filename = getattr(event, "filename", None) or getattr(
            getattr(event, "defn", None), "filename", None
        )
        print(f"[jurigged] {kind} {name or ''} {filename or ''}".rstrip(), flush=True)
    except Exception:
        pass
