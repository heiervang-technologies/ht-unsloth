"""Multi-role teacher client for the RLVR online loop (Track B).

A single OpenRouter call to ``openai/gpt-oss-120b`` returns four signals at
once — per-rollout grade, per-rollout critique, per-rollout counterfactual,
and a canonical demonstration. This is one HTTP round-trip per RLVR step,
not four; the combined-loss engine then routes each piece to the
appropriate objective in a single backward pass.

Stdlib HTTP only. ``lile.teach.tutor.run`` uses ``httpx`` which is fine for
the tutor distillation script but adds a heavy dep to anything that imports
it. The RLVR scheduler runs inside the daemon process, so we keep this
module on ``urllib.request`` to match the ``lile.teach.eval`` pattern and
stay torchless / dep-light.

API
---

::

    from lile.teach.teacher_oss120b import judge

    result = judge(
        prompt="What is 17 * 23?",
        rollouts=["391", "390", "I'm not sure"],
        domain="math",
    )
    # result["grades"]            -> ["correct", "wrong", "ambiguous"]
    # result["critiques"]         -> [None, "Off by 1; ...", None]
    # result["counterfactuals"]   -> [None, "390", None]
    # result["demonstration"]     -> "17 * 23 = 391"

The function raises ``RuntimeError`` if ``OPENROUTER_API_KEY`` is missing,
the response can't be parsed as JSON, or the per-rollout list lengths
don't match. Callers should treat it like any other network call: catch
``RuntimeError`` and ``urllib.error.URLError`` if they want to soft-fail.

Caching
-------

Repeated calls with identical ``(prompt, rollouts)`` are served from an
in-process LRU cache (max 256 entries) so retries don't double-bill.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Literal, TypedDict

__all__ = ["judge", "JudgeResult"]


_DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-oss-120b"

Domain = Literal["math", "code", "arc", "general"]
Grade = Literal["correct", "wrong", "ambiguous"]


class JudgeResult(TypedDict):
    grades: list[Grade]
    critiques: list[str | None]
    counterfactuals: list[str | None]
    demonstration: str


_SYSTEM_PROMPT = (
    "You are an expert teacher grading a student's attempted answers to a "
    "single problem. You will receive the problem and N candidate rollouts. "
    "Return STRICT JSON matching exactly this schema and nothing else:\n"
    "{\n"
    '  "grades": [<one of "correct"|"wrong"|"ambiguous"> for each rollout],\n'
    '  "critiques": [<short string explaining the error if "wrong", else null>],\n'
    '  "counterfactuals": [<a short near-miss bad output if "wrong", else null>],\n'
    '  "demonstration": <the canonical correct answer as a string>\n'
    "}\n"
    "Lengths of grades, critiques, counterfactuals must each equal N. "
    "Use \"ambiguous\" only when the rollout is unparseable, refuses, or is "
    "off-topic — not for partially-correct work (which is \"wrong\")."
)


def _format_user_message(prompt: str, rollouts: list[str], domain: Domain) -> str:
    parts = [f"DOMAIN: {domain}", "", f"PROBLEM:\n{prompt}", "", "ROLLOUTS:"]
    for i, r in enumerate(rollouts):
        parts.append(f"[{i}] {r}")
    parts.append("")
    parts.append("Respond with the JSON object only.")
    return "\n".join(parts)


def _rollouts_hash(rollouts: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(rollouts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _post_openrouter(
    url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout: float,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/heiervang-technologies/ht-unsloth",
            "X-Title": "lile rlvr teacher",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - controlled URL
        raw = resp.read().decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"teacher_oss120b: OpenRouter response was not valid JSON: {exc}"
        ) from exc


def _extract_message_content(envelope: dict) -> str:
    try:
        choices = envelope["choices"]
        msg = choices[0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"teacher_oss120b: malformed OpenRouter envelope: {envelope!r}"
        ) from exc
    # Some reasoning models put the answer on ``reasoning_content`` when
    # ``content`` is empty — same fallback as ``tutor/run.py``.
    text = msg.get("content") or msg.get("reasoning_content") or ""
    if not text:
        raise RuntimeError(
            "teacher_oss120b: OpenRouter returned empty message content"
        )
    return text


def _parse_judge_payload(text: str, n_rollouts: int) -> JudgeResult:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"teacher_oss120b: judge payload was not strict JSON: {exc}; "
            f"first 200 chars: {text[:200]!r}"
        ) from exc

    required = ("grades", "critiques", "counterfactuals", "demonstration")
    missing = [k for k in required if k not in obj]
    if missing:
        raise RuntimeError(
            f"teacher_oss120b: judge payload missing keys {missing}; got "
            f"keys={sorted(obj.keys())}"
        )

    grades = obj["grades"]
    critiques = obj["critiques"]
    counterfactuals = obj["counterfactuals"]
    demonstration = obj["demonstration"]

    if not (
        isinstance(grades, list)
        and isinstance(critiques, list)
        and isinstance(counterfactuals, list)
    ):
        raise RuntimeError(
            "teacher_oss120b: grades/critiques/counterfactuals must all be lists"
        )

    lens = (len(grades), len(critiques), len(counterfactuals))
    if any(L != n_rollouts for L in lens):
        raise RuntimeError(
            f"teacher_oss120b: per-rollout list lengths {lens} do not match "
            f"n_rollouts={n_rollouts}"
        )

    valid_grades = {"correct", "wrong", "ambiguous"}
    bad_grades = [g for g in grades if g not in valid_grades]
    if bad_grades:
        raise RuntimeError(
            f"teacher_oss120b: invalid grade values {bad_grades!r}; "
            f"expected one of {sorted(valid_grades)}"
        )

    if not isinstance(demonstration, str):
        raise RuntimeError(
            "teacher_oss120b: demonstration must be a string"
        )

    # Normalize critique/counterfactual entries: anything not a non-empty
    # string is None. (Some upstreams emit "" instead of null.)
    def _norm(xs: list) -> list[str | None]:
        out: list[str | None] = []
        for x in xs:
            if isinstance(x, str) and x.strip():
                out.append(x)
            else:
                out.append(None)
        return out

    return JudgeResult(
        grades=list(grades),
        critiques=_norm(critiques),
        counterfactuals=_norm(counterfactuals),
        demonstration=demonstration,
    )


@lru_cache(maxsize=256)
def _cached_judge(
    prompt: str,
    rollouts_key: str,
    rollouts_blob: str,
    domain: Domain,
    timeout: float,
    model: str,
    url: str,
) -> JudgeResult:
    """LRU-cached inner. Keyed on ``(prompt, sha256(rollouts))`` per spec.

    ``rollouts_blob`` is the JSON-encoded rollouts list — it's an argument
    (not closed over) so the cache key stays a tuple of hashable scalars,
    while still letting us reconstruct the original list for the request
    body. ``rollouts_key`` is what makes the cache identity unique; the
    blob itself is functionally derived from it.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "teacher_oss120b: OPENROUTER_API_KEY is not set in the daemon "
            "process environment. Add it to compose.lile-dev.yaml's "
            "environment block (or export it before running outside Docker)."
        )

    rollouts: list[str] = json.loads(rollouts_blob)
    n = len(rollouts)
    user_msg = _format_user_message(prompt, rollouts, domain)
    envelope = _post_openrouter(
        url=url,
        api_key=api_key,
        model=model,
        system=_SYSTEM_PROMPT,
        user=user_msg,
        timeout=timeout,
    )
    text = _extract_message_content(envelope)
    return _parse_judge_payload(text, n_rollouts=n)


def judge(
    prompt: str,
    rollouts: list[str],
    *,
    domain: Domain,
    timeout: float = 60.0,
    model: str = _DEFAULT_MODEL,
    url: str = _DEFAULT_URL,
) -> JudgeResult:
    """Issue one OpenRouter call returning all four teacher signals.

    Parameters
    ----------
    prompt
        The problem statement shown to the student.
    rollouts
        Student's candidate answers (length N, N >= 1).
    domain
        Routes prompt-construction nuance; currently informational only.
    timeout
        HTTP timeout in seconds. Default 60s — gpt-oss-120b is fast but
        OpenRouter cold-starts can spike.
    model, url
        Override the OpenRouter target. Defaults match the production
        ``ServeConfig.teacher_model`` / ``teacher_url``.

    Returns
    -------
    JudgeResult
        ``grades``, ``critiques``, ``counterfactuals`` are each length N;
        ``demonstration`` is a single string.

    Raises
    ------
    RuntimeError
        If ``OPENROUTER_API_KEY`` is unset, the response is not strict
        JSON, required keys are missing, list lengths don't match N, or
        a grade is outside ``{correct, wrong, ambiguous}``.
    """
    if not isinstance(rollouts, list) or len(rollouts) == 0:
        raise ValueError("teacher_oss120b: rollouts must be a non-empty list")

    rollouts_key = _rollouts_hash(rollouts)
    rollouts_blob = json.dumps(rollouts, sort_keys=True, ensure_ascii=False)
    return _cached_judge(
        prompt,
        rollouts_key,
        rollouts_blob,
        domain,
        timeout,
        model,
        url,
    )


def _clear_cache_for_tests() -> None:
    """Reset the LRU cache. Tests use this between scenarios."""
    _cached_judge.cache_clear()
