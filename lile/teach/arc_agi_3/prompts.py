"""ARC-AGI-3 prompt builder.

Produces a single user-facing message that frames the task as a grid
transformation problem: show train input/output pairs, show the test
input, demand a JSON-fenced list-of-lists answer. The output schema is
deliberately rigid so :mod:`runner` can parse it without hand-rolled
heuristics.

The header string ``"ARC-AGI-3 task"`` is also what the registered
verifier uses to claim ARC-shaped prompts (see :mod:`verifier`). Don't
rename it without updating the verifier's claims regex.
"""
from __future__ import annotations

from .loader import Grid, Task


# Marker that survives in the user message and is what
# ``verifier.claims()`` pattern-matches on. Stable string.
ARC_PROMPT_HEADER = "ARC-AGI-3 task"


def _format_grid(g: Grid) -> str:
    return "[\n" + ",\n".join("  " + str(row) for row in g) + "\n]"


def build_prompt(task: Task) -> str:
    """Return a single string prompt describing ``task``.

    The result is intended as the user-message body in a chat completion
    call. Returning a plain string (rather than a list of role/content
    dicts) keeps the function trivially testable and lets callers pick
    whatever system prompt suits their model.
    """
    parts: list[str] = []
    parts.append(f"{ARC_PROMPT_HEADER} ({task['id']})")
    parts.append(
        "You are solving a grid-transformation puzzle. Each cell is an "
        "integer in 0..9 (a color). Infer the rule from the training "
        "pairs, then apply it to the test input."
    )
    parts.append("")
    parts.append("Training pairs:")
    for i, pair in enumerate(task["train_pairs"], start=1):
        parts.append(f"\nPair {i} input:")
        parts.append(_format_grid(pair["input"]))
        parts.append(f"Pair {i} output:")
        parts.append(_format_grid(pair["output"]))
    parts.append("")
    parts.append("Test input:")
    parts.append(_format_grid(task["test_input"]))
    parts.append("")
    parts.append(
        "Reply with ONLY the test output as a JSON list-of-lists of "
        "integers, wrapped in a ```json fenced code block. No prose, no "
        "explanation outside the fence. Example:"
    )
    parts.append("```json")
    parts.append("[[0, 1], [1, 0]]")
    parts.append("```")
    return "\n".join(parts)
