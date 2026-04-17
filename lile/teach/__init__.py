"""Teaching procedures — compose the lile API into surgical memorization tools.

Out-of-API recipes for imprinting atomic facts with minimal collateral drift.

The core primitive is :func:`teach_entity`: given a set of paraphrased
questions and a set of accepted answer surface forms, it loops
(train → merge → greedy eval) until every question elicits some surface
form, while a separate **probe prompt** is re-evaluated every iteration
to detect representational collapse. On collapse, we snapshot-rollback
and abort — the fact is not worth degrading unrelated knowledge.

Design (from the in-session conversation):

- **Salient-only loss** is free: ``weighted_sft`` via ``build_chat_inputs``
  already masks prompt tokens with -100, so gradient flows only through
  the response (= surface form) tokens.
- **Paraphrase spray**: N question templates × M surface forms = N*M
  weighted_sft samples per step. Decouples the fact from prompt shape.
- **Collapse probe**: we snapshot before teaching, baseline-generate on
  an unrelated probe, and every iteration check that the probe answer
  still contains its anchor substring. If not, roll back.
- **Sequence-level success**: we check the full surface form appears in
  the greedy output, not just the first token — prevents the "9 → digit"
  failure mode where the salient integer is right but the transition
  token fights the model's prior.

The wrapper :func:`teach_number` generates the numeric surface forms
automatically ("9 billion" / "9B" / "9,000,000,000" / "nine billion")
so the caller only needs the target integer.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------- HTTP client
class _Client:
    """Minimal stdlib HTTP client for the lile FastAPI server."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _req(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self.base_url + path
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            body = e.read()
            raise RuntimeError(
                f"{method} {path} -> {e.code}: {body.decode('utf-8', 'replace')}"
            ) from None
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def snapshot_save(self, name: str) -> dict:
        return self._req("POST", "/v1/state/snapshot/save", {"name": name})

    def snapshot_load(self, name: str) -> dict:
        return self._req("POST", "/v1/state/snapshot/load", {"name": name})

    def train(self, objective: str, samples: list[dict], chunk_size: int = 8) -> dict:
        return self._req("POST", "/v1/train", {
            "objective": objective, "samples": samples,
            "batch_objectives": [], "kwargs": {}, "chunk_size": chunk_size,
        })

    def wait_for(self, token: int, timeout: float = 120.0) -> dict:
        return self._req("POST", f"/v1/wait?token={int(token)}&timeout={timeout}")

    def merge(self) -> dict:
        return self._req("POST", "/v1/state/merge")

    def chat(self, prompt: str, max_tokens: int = 48,
             temperature: float = 0.05, top_p: float = 1.0) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        body = self._req("POST", "/v1/chat/completions", payload)
        return body["choices"][0]["message"]["content"]


# --------------------------------------------------------------------- result types
@dataclass
class IterLog:
    step: int
    loss: float | None
    question_hits: list[tuple[str, bool, str]]   # (question, matched, greedy_output)
    probe_output: str
    probe_ok: bool


@dataclass
class TeachResult:
    success: bool
    aborted_on_collapse: bool
    iterations: int
    baseline_probe: str
    final_probe: str
    history: list[IterLog] = field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------- core
def teach_entity(
    question_templates: list[str],
    surface_forms: list[str],
    *,
    probe_prompt: str = "What is the capital of France?",
    probe_anchor: str | None = None,
    validator: Callable[[str], bool] | None = None,
    max_iters: int = 30,
    weight: float = 2.0,
    base_url: str = "http://127.0.0.1:8765",
    snapshot_name: str = "pre_teach",
    greedy_temp: float = 0.05,
    max_answer_tokens: int = 32,
    max_probe_tokens: int = 48,
    verbose: bool = True,
) -> TeachResult:
    """Iteratively teach that any ``question_templates`` question answers
    with any string in ``surface_forms``, without collapsing the probe.

    Parameters
    ----------
    question_templates
        Paraphrases of the same question. All must elicit the fact.
    surface_forms
        Accepted answer strings. At least one must appear in the greedy
        output for each question to declare success.
    probe_prompt
        Unrelated control prompt; regenerated every iteration.
    probe_anchor
        Substring that must keep appearing in the probe output. If the
        baseline contains it and a later iteration does not, we treat
        that as representational collapse and roll back. If ``None``,
        the anchor is auto-extracted as the first three words of the
        baseline (a weak default — provide one explicitly when possible).
    validator
        Override for the per-question success test. Default: any
        ``surface_form`` appears as substring in the greedy output.
    max_iters
        Hard cap. On hit without success, rolls back and returns failure.
    weight
        ``weighted_sft`` sample weight. 2-4 accelerates convergence on
        short atomic facts.
    snapshot_name
        Name used for the pre-teach snapshot / rollback target.
    """
    if not question_templates:
        raise ValueError("need at least one question template")
    if not surface_forms:
        raise ValueError("need at least one accepted surface form")

    client = _Client(base_url)

    if validator is None:
        def validator(output: str) -> bool:  # noqa: E306
            low = output.lower()
            return any(sf.lower() in low for sf in surface_forms)

    def _log(msg: str) -> None:
        if verbose:
            print(f"[teach] {msg}")

    # ---------- 1. Baseline: snapshot + probe
    _log(f"snapshot → {snapshot_name!r}")
    client.snapshot_save(snapshot_name)

    baseline_probe = client.chat(
        probe_prompt, max_tokens=max_probe_tokens, temperature=greedy_temp,
    ).strip()
    _log(f"probe baseline: {baseline_probe[:120]!r}")

    if probe_anchor is None:
        # Auto-pick: first ~3 words or first 24 chars, whichever shorter.
        words = baseline_probe.split()
        probe_anchor = " ".join(words[:3]) if len(words) >= 3 else baseline_probe[:24]
        _log(f"probe_anchor (auto): {probe_anchor!r}")

    # ---------- 2. Build the training batch (paraphrase × surface_form)
    samples = [
        {"prompt": q, "response": sf, "weight": float(weight)}
        for q in question_templates
        for sf in surface_forms
    ]

    history: list[IterLog] = []
    final_probe = baseline_probe

    for step in range(1, max_iters + 1):
        # ---------- a. train one step
        tr = client.train("weighted_sft", samples)
        commit_token = tr.get("commit_token")
        if commit_token is not None:
            client.wait_for(int(commit_token), timeout=120.0)
        # ---------- b. merge
        client.merge()

        # ---------- c. evaluate every question
        q_hits: list[tuple[str, bool, str]] = []
        for q in question_templates:
            out = client.chat(
                q, max_tokens=max_answer_tokens, temperature=greedy_temp,
            ).strip()
            q_hits.append((q, validator(out), out))

        # ---------- d. probe
        probe_out = client.chat(
            probe_prompt, max_tokens=max_probe_tokens, temperature=greedy_temp,
        ).strip()
        final_probe = probe_out
        probe_ok = probe_anchor.lower() in probe_out.lower()

        history.append(IterLog(
            step=step, loss=None, question_hits=q_hits,
            probe_output=probe_out, probe_ok=probe_ok,
        ))

        hit_count = sum(1 for _, ok, _ in q_hits if ok)
        _log(f"iter {step:>2} · hits {hit_count}/{len(q_hits)} · probe_ok={probe_ok}")

        # ---------- e. check stop conditions
        if not probe_ok:
            _log(f"COLLAPSE: probe anchor {probe_anchor!r} lost from output "
                 f"({probe_out[:100]!r}) — rolling back")
            client.snapshot_load(snapshot_name)
            return TeachResult(
                success=False, aborted_on_collapse=True, iterations=step,
                baseline_probe=baseline_probe, final_probe=probe_out,
                history=history, note="probe_anchor lost",
            )

        if all(ok for _, ok, _ in q_hits):
            _log(f"SUCCESS at iter {step}")
            return TeachResult(
                success=True, aborted_on_collapse=False, iterations=step,
                baseline_probe=baseline_probe, final_probe=probe_out,
                history=history, note="",
            )

    # ---------- hit cap
    _log(f"cap {max_iters} reached without full memorization — rolling back")
    client.snapshot_load(snapshot_name)
    return TeachResult(
        success=False, aborted_on_collapse=False, iterations=max_iters,
        baseline_probe=baseline_probe, final_probe=final_probe,
        history=history, note="max_iters reached",
    )


# --------------------------------------------------------------------- number wrapper
_SMALL_WORD = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve",
}


def numeric_surface_forms(n: int, noun: str = "") -> list[str]:
    """Common English surface forms for an integer.

    ``noun`` is appended to magnitude-suffixed forms: ``numeric_surface_forms(9_000_000_000, "parameters")``
    includes "9 billion parameters" / "9B parameters" / etc.
    """
    forms: list[str] = [f"{n}", f"{n:,}"]
    suffix = f" {noun}" if noun else ""

    def _add_unit(value: int, unit_word: str, unit_short: str) -> None:
        forms.extend([
            f"{value} {unit_word}",
            f"{value}{unit_short}",
            f"{value} {unit_word}{suffix}" if suffix else f"{value} {unit_word}",
            f"{value}{unit_short}{suffix}" if suffix else f"{value}{unit_short}",
        ])
        if value in _SMALL_WORD:
            forms.append(f"{_SMALL_WORD[value]} {unit_word}")
            if suffix:
                forms.append(f"{_SMALL_WORD[value]} {unit_word}{suffix}")

    if n >= 1_000_000_000 and n % 1_000_000_000 == 0:
        _add_unit(n // 1_000_000_000, "billion", "B")
    if n >= 1_000_000 and n % 1_000_000 == 0:
        _add_unit(n // 1_000_000, "million", "M")
    if n >= 1_000 and n % 1_000 == 0:
        _add_unit(n // 1_000, "thousand", "K")

    # Dedupe, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for f in forms:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def teach_number(
    question_templates: list[str],
    answer_int: int,
    *,
    noun: str = "",
    extra_surface_forms: list[str] | None = None,
    **kwargs: Any,
) -> TeachResult:
    """Teach an integer fact. Auto-generates numeric surface forms.

    ``noun``: optional noun appended to magnitude forms (e.g. "parameters"
    so "9 billion parameters" is accepted).

    ``extra_surface_forms``: caller-supplied additional accepted strings
    (e.g. ``["roughly 9 billion"]``).
    """
    forms = numeric_surface_forms(answer_int, noun=noun)
    if extra_surface_forms:
        forms.extend(extra_surface_forms)
    return teach_entity(question_templates, forms, **kwargs)
