"""Verifier registry — pass/fail (or graded) checks on candidate responses
where an objective ground-truth exists.

Consumed by TTRL-style pseudo-reward training (roadmap PR L) and future RL
workstreams. A verifier answers one question:

    Given ``prompt``, does ``candidate`` satisfy the domain-specific check?

Return values:

- ``True`` / ``1.0`` — candidate verifies cleanly
- ``False`` / ``0.0`` — candidate fails the check
- ``None`` — verifier is not applicable to this prompt (caller should skip;
  ``None`` is never coerced to False downstream)

Verifiers are adapters: they must never raise into the caller. The top-level
:func:`verify` dispatcher catches any exception and returns ``None`` so a
bad verifier can't take down the train loop.

Registering a custom verifier:

    from lile.objectives.verifiers import register

    @register("my_domain")
    def verify(prompt: str, candidate: str) -> bool | float | None:
        ...
"""
from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

Verifier = Callable[[str, str], "bool | float | None"]

# Populated at module import by the seed modules below.
VERIFIERS: dict[str, Verifier] = {}


def register(domain: str) -> Callable[[Verifier], Verifier]:
    """Decorator that registers ``fn`` as the verifier for ``domain``."""
    def _wrap(fn: Verifier) -> Verifier:
        VERIFIERS[domain] = fn
        return fn
    return _wrap


def verify(domain: str, prompt: str, candidate: str) -> bool | float | None:
    """Dispatch to ``VERIFIERS[domain]``, swallowing adapter exceptions.

    Returns ``None`` when no verifier is registered for ``domain`` or the
    verifier itself raises — both indicate "can't judge this", never "fail".
    """
    fn = VERIFIERS.get(domain)
    if fn is None:
        return None
    try:
        return fn(prompt, candidate)
    except Exception as exc:
        log.warning("verifier %r raised %s — returning None", domain, exc)
        return None


def select(prompt: str) -> str | None:
    """Return the first registered verifier whose domain claims ``prompt``.

    Each seed verifier exposes a ``claims(prompt) -> bool`` sibling used here
    to let TTRL pick a verifier without the caller hard-coding domains.
    Custom verifiers without a ``claims`` attribute are skipped — register
    your own ``select`` if you need richer routing.
    """
    for domain, fn in VERIFIERS.items():
        claims = getattr(fn, "claims", None)
        if callable(claims) and claims(prompt):
            return domain
    return None


# Seed the registry. Order matters for :func:`select` — cheapest/strictest
# claim first, so "math" wins over "code" on ambiguous prompts.
from . import _math  # noqa: F401, E402
from . import _code  # noqa: F401, E402
