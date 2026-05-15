"""ARC-AGI-3 smoke eval — Track E wiring check.

A one-shot CLI that runs the pinned ARC tasks against a live lile daemon
via the OpenAI-compatible ``/v1/chat/completions`` endpoint and reports a
pass-rate. lm-eval has no ARC-AGI-3 wrapper, so this bypasses it and
goes straight through :func:`lile.teach.arc_agi_3.runner.run_task`.

Use:

    python -m lile.teach.eval_arc_smoke --n 3
    python -m lile.teach.eval_arc_smoke --n 3 --out /tmp/arc.json

The runner only needs an async ``.generate(messages, **kwargs)`` shaped
like the Controller. We adapt the existing stdlib HTTP client used by
:mod:`lile.teach` (``_Client``) into that shape — no new HTTP plumbing,
no third-party deps.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from lile.teach import _Client
from lile.teach.arc_agi_3 import loader as arc_loader
from lile.teach.arc_agi_3 import runner as arc_runner


# ----------------------------------------------------------------- adapter
class _DaemonClient:
    """Async ``.generate`` adapter over the lile daemon's chat endpoint.

    Wraps :class:`lile.teach._Client` so :func:`runner.run_task` can call
    ``await client.generate(messages=..., temperature=..., ...)`` exactly
    like the in-process ``Controller``. We forward ``temperature``,
    ``top_p``, and ``max_new_tokens`` — the runner sets these explicitly
    for deterministic decoding (temperature=0.0).

    Returns the response in the dict shape the runner already accepts:
    ``{"response": str, "reasoning_content": None, "raw": str}``.
    """

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._http = _Client(base_url, timeout=timeout)

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 800,
        **_unused: Any,
    ) -> dict[str, Any]:
        payload = {
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        # `_Client._req` is sync stdlib; punt it to a worker thread so the
        # caller's event loop is not blocked.
        body = await asyncio.to_thread(self._http._req, "POST", "/v1/chat/completions", payload)
        msg = body["choices"][0]["message"]
        # Qwen3.5 reasoning model: the assistant turn often emits everything
        # as reasoning_content with content empty. Fall back so the runner
        # sees the actual generation.
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        text = content if content.strip() else reasoning
        return {"response": text, "reasoning_content": reasoning, "raw": text}


# ----------------------------------------------------------------- core
async def run_arc_eval(
    daemon_url: str = "http://127.0.0.1:8768",
    n: int = 3,
    *,
    client: Any | None = None,
    max_tokens: int = 800,
) -> dict[str, Any]:
    """Run up to ``n`` pinned ARC tasks and return a structured summary.

    Parameters
    ----------
    daemon_url
        Base URL of the lile daemon (no trailing ``/v1``). Used only when
        ``client`` is ``None``.
    n
        Number of pinned tasks to run; capped at the loader's count.
    client
        Optional pre-built generate-shaped client (used by tests to mock
        the HTTP plane). Must expose async ``.generate(messages=…, …)``.
    max_tokens
        Forwarded to the runner.

    Returns
    -------
    dict
        ``{"daemon": str, "n": int, "correct": int, "total": int,
           "pass_rate": float, "tasks": [{"task_id", "correct",
           "parse_error"}, ...]}``
    """
    tasks = arc_loader.load_tasks()
    n_capped = max(0, min(int(n), len(tasks)))
    tasks = tasks[:n_capped]
    if client is None:
        client = _DaemonClient(daemon_url)

    task_results: list[dict[str, Any]] = []
    correct = 0
    for task in tasks:
        result = await arc_runner.run_task(client, task, max_tokens=max_tokens)
        task_results.append({
            "task_id": result["task_id"],
            "correct": bool(result["correct"]),
            "parse_error": result["parse_error"],
        })
        if result["correct"]:
            correct += 1

    total = len(task_results)
    pass_rate = (correct / total) if total > 0 else 0.0
    return {
        "daemon": daemon_url,
        "n": total,
        "correct": correct,
        "total": total,
        "pass_rate": pass_rate,
        "tasks": task_results,
    }


# ----------------------------------------------------------------- CLI
def _print_summary(summary: dict[str, Any]) -> None:
    for t in summary["tasks"]:
        print(
            f"[{t['task_id']}] correct={t['correct']} "
            f"parse_error={t['parse_error']!r}"
        )
    correct = summary["correct"]
    total = summary["total"]
    pct = summary["pass_rate"] * 100.0
    print(f"correct: {correct}/{total} ({pct:.1f}%)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lile.teach.eval_arc_smoke")
    p.add_argument(
        "--n", type=int, default=3,
        help="Number of pinned tasks to run (capped at loader count).",
    )
    p.add_argument(
        "--daemon", default="http://127.0.0.1:8768",
        help="lile daemon base URL (no trailing /v1).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Optional JSON path to write full results to.",
    )
    args = p.parse_args(argv)

    summary = asyncio.run(run_arc_eval(daemon_url=args.daemon, n=args.n))
    _print_summary(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
