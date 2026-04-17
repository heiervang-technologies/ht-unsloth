"""Executable-subset Python verifier.

Claims prompts whose last line looks like "Expected output: <...>". Verifies
that the fenced Python block in the candidate, when run under a restricted
``exec`` sandbox with a wall-clock budget, prints exactly the expected
output.

The sandbox is intentionally small: no ``__import__``, no filesystem, no
network, no ``open``. It is **not** a security boundary — it's a cheap
"does this candidate do the right thing" check for pseudo-reward training.
For adversarial inputs (user-submitted code in prod), wrap this in a real
sandbox (nsjail, firecracker). That wrapping is PR L's problem, not the
registry's.
"""
from __future__ import annotations

import io
import re
import signal
import threading
from contextlib import redirect_stdout

from . import register

_EXPECTED = re.compile(r"(?i)\bexpected(?:\s+output)?\s*[:=]\s*(.+?)(?:\n|$)")
_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_EXEC_TIMEOUT_S = 1


class _ExecTimeout(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover — signal path
    raise _ExecTimeout()


_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "int", "len", "list", "map", "max", "min", "print", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    )
}


def extract_expected(prompt: str) -> str | None:
    """Return the ``Expected:`` stanza from ``prompt`` (stripped), or None."""
    m = _EXPECTED.search(prompt)
    return m.group(1).strip() if m else None


def extract_code(candidate: str) -> str | None:
    """Return the first fenced Python block in ``candidate``, or None."""
    m = _CODE_FENCE.search(candidate)
    return m.group(1) if m else None


def _run_sandboxed(code: str) -> str:
    """Run ``code`` with restricted builtins and a wall-clock timeout.

    Returns captured stdout on clean exit; raises on any error (caller
    translates to a False verdict).

    The SIGALRM wall-clock budget only arms on the main thread — Python's
    ``signal.signal`` raises ``ValueError`` from any other thread. Worker-
    thread callers (dataloader workers, asyncio executors, ComputeQueue
    worker) lose the timeout guard; PR L is expected to swap this path
    for ``multiprocessing.Process`` + ``.join(timeout=…)`` before accepting
    adversarial candidates. The no-timeout fallback is acceptable for the
    seed because every caller today runs against trusted candidates.
    """
    buf = io.StringIO()
    ns = {"__builtins__": _SAFE_BUILTINS}
    armed = threading.current_thread() is threading.main_thread()
    prev = None
    if armed:
        prev = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_EXEC_TIMEOUT_S)
    try:
        with redirect_stdout(buf):
            exec(code, ns, ns)
    finally:
        if armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
    return buf.getvalue().strip()


def claims(prompt: str) -> bool:
    return extract_expected(prompt) is not None


@register("code")
def verify(prompt: str, candidate: str) -> bool | None:
    expected = extract_expected(prompt)
    if expected is None:
        return None
    code = extract_code(candidate)
    if code is None:
        return False
    try:
        got = _run_sandboxed(code)
    except Exception:
        return False
    return got == expected


verify.claims = claims  # type: ignore[attr-defined]
