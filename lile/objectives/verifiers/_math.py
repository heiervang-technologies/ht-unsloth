"""GSM8K-style numeric-answer verifier.

Claims prompts that look like arithmetic word problems (or any prompt
containing a number plus a question verb). Verifies that the candidate
response exposes a final, extractable numeric answer.

Equivalence semantics are deliberately minimal here — the verifier only
reports *extractability*. TTRL layers equivalence hashing on top by
calling :func:`extract_answer` directly. Callers that have a reference
answer and want tolerant equivalence should use :func:`answers_match`,
which handles fraction canonicalization and a configurable ``rtol``.
"""
from __future__ import annotations

import math
import re

from . import register

# Explicit boxed / labelled answer patterns, tried in order.
#
# Scientific notation (``1.5e10`` → ``1.5``) is treated as the decimal
# prefix. Fractions (``3/4``) have their own preferred patterns — see
# :func:`extract_answer` — applied *first* when the prompt signals a
# fraction answer ("what fraction", "in lowest terms", etc.). Otherwise
# we fall back to the decimal-only patterns below, keeping the classic
# GSM8K short-answer shape.
_ANSWER_PATTERNS = (
    re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}"),
    re.compile(r"(?i)\banswer(?:\s+is)?\s*[:=]?\s*(-?\d[\d,]*(?:\.\d+)?)"),
)

# Fraction variants of the same anchors. Preferred over the decimal
# patterns only when the prompt asks for a fraction, to keep the default
# GSM8K-exact-match path untouched.
_FRACTION_ANCHORED = (
    re.compile(r"####\s*(-?\d+\s*/\s*\d+)"),
    re.compile(r"\\boxed\{\s*(-?\d+\s*/\s*\d+)\s*\}"),
    re.compile(r"(?i)\banswer(?:\s+is)?\s*[:=]?\s*(-?\d+\s*/\s*\d+)"),
)

# Fallback: last number anywhere in the response.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Fallback for fraction mode: last ``a/b`` anywhere in the response.
# Anchored on word-like boundaries so we don't eat dates ("3/4/2020").
_FRACTION = re.compile(r"(?<![\d/])-?\d+\s*/\s*\d+(?![\d/])")

# Cheap prompt classifier — question verb + a digit or arithmetic noun.
_PROMPT_CLAIM = re.compile(
    r"(?i)(how many|how much|what is|what's|find|compute|calculate|solve|"
    r"sum of|product of|average|mean|total|percentage|fraction)",
)

# Prompt phrasing that signals the expected answer is a fraction. Narrow
# on purpose: "fraction" appearing incidentally ("what is the decimal
# equivalent of the fraction 3/4") shouldn't flip the mode. Must be an
# explicit answer-shape directive.
_FRACTION_PROMPT = re.compile(
    r"(?i)\b("
    r"what\s+fraction|"
    r"as\s+(?:a|the)\s+fraction|"
    r"express\s+(?:the\s+answer\s+)?as\s+a\s+fraction|"
    r"(?:give|write|leave)\s+(?:your\s+answer|the\s+answer)\s+as\s+a\s+fraction|"
    r"in\s+(?:lowest|simplest)\s+(?:terms|form)"
    r")\b"
)


def _normalize_decimal(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _normalize_fraction(s: str) -> str:
    """Strip whitespace in ``a / b`` → ``a/b``. Does not reduce the fraction."""
    return s.replace(" ", "")


def _is_fraction_prompt(prompt: str | None) -> bool:
    return bool(prompt) and bool(_FRACTION_PROMPT.search(prompt))


def extract_answer(text: str, prompt: str | None = None) -> str | None:
    """Return the normalized final answer in ``text`` or ``None``.

    When ``prompt`` asks for a fraction ("what fraction ...", "as a
    fraction", "in lowest terms"), fraction patterns (``a/b``) are tried
    first so responses like "#### 5/8" return ``"5/8"`` rather than
    being split by the decimal regex into the numerator ``"5"`` alone.

    Without a ``prompt`` hint the behavior is unchanged: the classic
    ``####``, ``\\boxed{...}``, and "answer: N" patterns are tried in
    order, then the last number in the text as a fallback. This keeps
    GSM8K-style short-answer bench behaviour byte-stable.
    """
    if _is_fraction_prompt(prompt):
        for pat in _FRACTION_ANCHORED:
            m = pat.search(text)
            if m:
                return _normalize_fraction(m.group(1))
        fracs = _FRACTION.findall(text)
        if fracs:
            return _normalize_fraction(fracs[-1])
        # Fraction mode, no fraction in candidate. Fall through to the
        # anchored decimal patterns only (the model may have answered with
        # an integer like ``#### 1``). The last-number fallback is *not*
        # used here — in fraction mode an unanchored year or incidental
        # number is almost always spurious.
        for pat in _ANSWER_PATTERNS:
            m = pat.search(text)
            if m:
                return _normalize_decimal(m.group(1))
        return None
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return _normalize_decimal(m.group(1))
    matches = _NUMBER.findall(text)
    if matches:
        return _normalize_decimal(matches[-1])
    return None


def claims(prompt: str) -> bool:
    """True when ``prompt`` looks like a math problem (heuristic)."""
    return bool(_PROMPT_CLAIM.search(prompt)) and bool(_NUMBER.search(prompt))


@register("math")
def verify(prompt: str, candidate: str) -> bool | None:
    if not claims(prompt):
        return None
    return extract_answer(candidate, prompt=prompt) is not None


verify.claims = claims  # type: ignore[attr-defined]


# ------------------------------------------------------------------ equivalence


def _parse_number(s: str) -> float | None:
    """Parse ``s`` as a float. Accepts plain decimals, comma grouping, and
    simple ``a/b`` fractions. Returns ``None`` on any parse failure."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if not s:
        return None
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            d = float(den)
            if d == 0.0:
                return None
            return float(num) / d
        except (TypeError, ValueError):
            return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def answers_match(
    reference: object,
    candidate_answer: object,
    *,
    rtol: float = 1e-3,
    abs_tol: float = 0.0,
) -> bool:
    """Tolerant equivalence for math answers.

    Accepts:

    - Exact string match after whitespace / comma-grouping normalization.
    - Fraction canonicalization: ``"5 / 8"`` and ``"5/8"`` are equal, and
      a fraction like ``"5/8"`` matches its decimal equivalent ``"0.625"``
      via the numeric path.
    - Numeric tolerance: both sides are float-parsed and compared with
      ``math.isclose(rel_tol=rtol, abs_tol=abs_tol)``. The default
      ``rtol=1e-3`` reclaims mini-GSM8K misses like ``43.98`` vs
      ``43.98226`` where the gold carries extra decimals. Eval harnesses
      that want stricter equality can tighten via the kwarg.

    Returns ``False`` for unparseable inputs or type mismatches. Never
    raises.
    """
    if reference is None or candidate_answer is None:
        return False
    ref_s = str(reference).strip()
    cand_s = str(candidate_answer).strip()
    if not ref_s or not cand_s:
        return False

    # Quick win: post-normalization string compare. Handles the exact-integer
    # and exact-fraction cases without paying the float-parse cost.
    ref_norm = _normalize_fraction(ref_s) if "/" in ref_s else ref_s.replace(",", "")
    cand_norm = _normalize_fraction(cand_s) if "/" in cand_s else cand_s.replace(",", "")
    if ref_norm == cand_norm:
        return True

    # Numeric compare with the caller-specified tolerance.
    r = _parse_number(ref_s)
    c = _parse_number(cand_s)
    if r is None or c is None:
        return False
    return math.isclose(r, c, rel_tol=rtol, abs_tol=abs_tol)
