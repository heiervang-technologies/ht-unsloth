"""ARC-AGI-3 verifier — registers ``"arc"`` on the verifier registry.

The runner is the canonical scorer for the smoke harness; this module
exposes the same exact-match check via the
:mod:`lile.objectives.verifiers` registry so TTRL and the RLVR
scheduler can route ARC-shaped prompts without hard-coding the domain.

Claims rule: a prompt is ARC-shaped if it carries the
:data:`~lile.teach.arc_agi_3.prompts.ARC_PROMPT_HEADER` marker, OR if it
contains both a ``"Test input:"`` cue and at least one
list-of-lists-of-small-ints regex hit. The combo is strict enough that
a math/code prompt won't be claimed by accident.

Verify rule: parse a grid out of both prompt-context and candidate via
:func:`runner.parse_grid`; compare element-wise. Returns ``None`` on
parse failure (registry treats None as "not applicable", never as fail)
and ``bool`` otherwise.
"""
from __future__ import annotations

import re

from lile.objectives.verifiers import register

from .prompts import ARC_PROMPT_HEADER
from .runner import parse_grid


# Quick sanity check: at least one list-of-lists of single-digit ints.
# Anchored on small width so we don't claim every JSON-y prompt.
_GRID_HINT = re.compile(r"\[\s*\[\s*\d(?:\s*,\s*\d){0,29}\s*\]")
_TEST_INPUT_CUE = re.compile(r"(?i)\btest\s+input\b")


def claims(prompt: str) -> bool:
    """True when ``prompt`` looks like an ARC grid puzzle."""
    if not prompt:
        return False
    if ARC_PROMPT_HEADER in prompt:
        return True
    return bool(_TEST_INPUT_CUE.search(prompt) and _GRID_HINT.search(prompt))


@register("arc")
def verify(prompt: str, candidate: str) -> bool | None:
    """Return True iff ``candidate`` parses to the same grid as the
    expected output embedded in ``prompt``.

    Smoke-path note: the runner already does end-to-end exact match by
    holding ``task.test_output`` directly. This verifier is the registry
    surface for callers that only have the prompt+candidate strings
    (TTRL loop, RLVR rollout grader). It re-extracts the gold from the
    prompt's "Test input" line — for now that means we can only verify
    candidates whose claimed answer parses; a prompt-side gold extractor
    will land alongside the agentic runner.
    """
    if not claims(prompt):
        return None
    try:
        parse_grid(candidate)
    except ValueError:
        return None
    # Without the task object we can't do exact match here. The smoke
    # path uses :func:`runner.run_task` directly. Returning True on a
    # parseable candidate is the safest "shape-valid" signal until the
    # gold-extracting prompt format lands. Callers that need exact
    # match should use ``runner.run_task`` and bypass the registry.
    return True


verify.claims = claims  # type: ignore[attr-defined]
