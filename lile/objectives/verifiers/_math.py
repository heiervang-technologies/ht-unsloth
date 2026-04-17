"""GSM8K-style numeric-answer verifier.

Claims prompts that look like arithmetic word problems (or any prompt
containing a number plus a question verb). Verifies that the candidate
response exposes a final, extractable numeric answer.

Equivalence semantics are deliberately minimal here — the verifier only
reports *extractability*. TTRL layers equivalence hashing on top by
calling :func:`extract_answer` directly.
"""
from __future__ import annotations

import re

from . import register

# Explicit boxed / labelled answer patterns, tried in order.
#
# Intentionally narrow: scientific notation (``1.5e10`` → ``1.5``) and
# fractions (``3/4`` → ``3``) are treated as the integer/decimal prefix.
# Fine for GSM8K-style benches where the final answer is always a plain
# decimal; TTRL layers equivalence hashing on top of ``extract_answer``
# when richer number shapes matter.
_ANSWER_PATTERNS = (
    re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}"),
    re.compile(r"(?i)\banswer(?:\s+is)?\s*[:=]?\s*(-?\d[\d,]*(?:\.\d+)?)"),
)
# Fallback: last number anywhere in the response.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Cheap prompt classifier — question verb + a digit or arithmetic noun.
_PROMPT_CLAIM = re.compile(
    r"(?i)(how many|how much|what is|what's|find|compute|calculate|solve|"
    r"sum of|product of|average|mean|total|percentage|fraction)",
)


def _normalize(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def extract_answer(text: str) -> str | None:
    """Return the normalized final numeric answer in ``text`` or ``None``."""
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return _normalize(m.group(1))
    matches = _NUMBER.findall(text)
    if matches:
        return _normalize(matches[-1])
    return None


def claims(prompt: str) -> bool:
    """True when ``prompt`` looks like a math problem (heuristic)."""
    return bool(_PROMPT_CLAIM.search(prompt)) and bool(_NUMBER.search(prompt))


@register("math")
def verify(prompt: str, candidate: str) -> bool | None:
    if not claims(prompt):
        return None
    return extract_answer(candidate) is not None


verify.claims = claims  # type: ignore[attr-defined]
