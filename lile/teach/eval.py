"""Offline eval harness for lile — regression-check at n=100 per task.

CLI entry point for the harness specified in
`lile/docs/research/eval-harness.md`. Hits the OpenAI-compatible endpoint
of a running lile daemon (or any server that speaks /v1/chat/completions)
and runs four verifiable tasks whose results are tied to the daemon's
`commit_cursor` for A/B reproducibility.

    uv run python -m lile.teach.eval \\
        --endpoint http://127.0.0.1:8768/v1 \\
        --model unsloth/Qwen3-9B \\
        --tasks hellaswag,arc_easy,arc_challenge,gsm8k \\
        --code-tasks humaneval_plus \\
        --limit 100 \\
        --out lile_data/evals/baseline.json

Deps are opt-in via the `lile[eval]` extra (lm-eval, evalplus). Running
with those unavailable prints a useful stub message per task rather than
crashing — so scaffolding lands before the full pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------- task registry
LM_EVAL_TASKS: dict[str, dict[str, Any]] = {
    "hellaswag":     {"metric": "acc_norm",    "lm_eval_name": "hellaswag"},
    "arc_easy":      {"metric": "acc_norm",    "lm_eval_name": "arc_easy"},
    "arc_challenge": {"metric": "acc_norm",    "lm_eval_name": "arc_challenge"},
    "gsm8k":         {"metric": "exact_match", "lm_eval_name": "gsm8k_cot_zeroshot"},
}

CODE_TASKS: dict[str, dict[str, Any]] = {
    "humaneval_plus": {"metric": "pass@1", "evalplus_dataset": "humaneval"},
}


# ----------------------------------------------------------------- result types
@dataclass
class TaskResult:
    task: str
    metric: str
    value: float
    n: int
    wall_clock_s: float
    raw: dict[str, Any] = field(default_factory=dict)
    stub: bool = False


@dataclass
class RunResult:
    run_id: str
    timestamp: str
    endpoint: str
    model: str
    commit_cursor_before: int | None
    commit_cursor_after: int | None
    tasks: list[TaskResult] = field(default_factory=list)


# ----------------------------------------------------------------- daemon probe
def _get_commit_cursor(endpoint: str) -> int | None:
    """GET {endpoint_root}/health → commit_cursor, or None if endpoint does not expose it."""
    root = endpoint.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(root + "/health", timeout=5.0) as r:
            body = json.loads(r.read().decode("utf-8"))
        return int(body.get("commit_cursor")) if "commit_cursor" in body else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


# ----------------------------------------------------------------- runners
def _run_lm_eval(task: str, endpoint: str, model: str, limit: int,
                 batch_size: int) -> TaskResult:
    """Run a single lm-eval-harness task via local-chat-completions adapter.

    lm-eval's `local-chat-completions` model backend speaks the OpenAI chat
    API. Invocation (once deps are installed):

        from lm_eval import simple_evaluate
        res = simple_evaluate(
            model="local-chat-completions",
            model_args=f"model={model},base_url={endpoint}/chat/completions",
            tasks=[LM_EVAL_TASKS[task]["lm_eval_name"]],
            limit=limit,
            batch_size=batch_size,
        )

    Scaffolding returns a stub result when lm-eval is not importable.
    """
    meta = LM_EVAL_TASKS[task]
    t0 = time.time()
    try:
        from lm_eval import simple_evaluate  # type: ignore[import-not-found]
    except ImportError:
        return TaskResult(
            task=task, metric=meta["metric"], value=float("nan"), n=0,
            wall_clock_s=0.0, stub=True,
            raw={"note": "lm-eval not installed; run `uv sync --extra eval`"},
        )
    res = simple_evaluate(
        model="local-chat-completions",
        model_args=f"model={model},base_url={endpoint}/chat/completions",
        tasks=[meta["lm_eval_name"]],
        limit=limit,
        batch_size=batch_size,
    )
    results = res["results"][meta["lm_eval_name"]]
    value = float(results.get(meta["metric"], results.get(f"{meta['metric']},none", 0.0)))
    return TaskResult(
        task=task, metric=meta["metric"], value=value, n=limit,
        wall_clock_s=time.time() - t0, raw=results,
    )


def _run_evalplus(task: str, endpoint: str, model: str,
                  limit: int) -> TaskResult:
    """Run an evalplus code task via its OpenAI-compat backend.

        python -m evalplus.evaluate --dataset humaneval --model {model} \\
            --backend openai --base-url {endpoint}

    Scaffolding returns a stub when evalplus is not importable.
    """
    meta = CODE_TASKS[task]
    t0 = time.time()
    try:
        import evalplus  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return TaskResult(
            task=task, metric=meta["metric"], value=float("nan"), n=0,
            wall_clock_s=0.0, stub=True,
            raw={"note": "evalplus not installed; run `uv sync --extra eval`"},
        )
    raise NotImplementedError(
        "evalplus integration: call evalplus.evaluate.entry_point with "
        f"dataset={meta['evalplus_dataset']!r}, backend='openai', "
        f"base_url={endpoint!r}, limit={limit}. Time: {time.time() - t0:.1f}s."
    )


# ----------------------------------------------------------------- driver
def run(endpoint: str, model: str, tasks: list[str], code_tasks: list[str],
        limit: int, batch_size: int) -> RunResult:
    run_id = f"eval-{uuid.uuid4().hex[:8]}"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cursor_before = _get_commit_cursor(endpoint)

    results: list[TaskResult] = []
    for task in tasks:
        if task not in LM_EVAL_TASKS:
            raise SystemExit(f"unknown lm-eval task: {task!r} (known: {sorted(LM_EVAL_TASKS)})")
        print(f"[eval] {task} (n={limit})", file=sys.stderr)
        results.append(_run_lm_eval(task, endpoint, model, limit, batch_size))

    for task in code_tasks:
        if task not in CODE_TASKS:
            raise SystemExit(f"unknown code task: {task!r} (known: {sorted(CODE_TASKS)})")
        print(f"[eval] {task} (n={limit})", file=sys.stderr)
        results.append(_run_evalplus(task, endpoint, model, limit))

    cursor_after = _get_commit_cursor(endpoint)
    return RunResult(
        run_id=run_id, timestamp=ts, endpoint=endpoint, model=model,
        commit_cursor_before=cursor_before, commit_cursor_after=cursor_after,
        tasks=results,
    )


def _jsonable_value(v: float) -> float | None:
    """Stub tasks set value=NaN, but NaN is not valid JSON. Emit null instead."""
    return None if isinstance(v, float) and math.isnan(v) else v


def _emit(result: RunResult, out: Path | None) -> None:
    payload = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "endpoint": result.endpoint,
        "model": result.model,
        "commit_cursor_before": result.commit_cursor_before,
        "commit_cursor_after": result.commit_cursor_after,
        "tasks": [{
            "task": t.task, "metric": t.metric, "value": _jsonable_value(t.value),
            "n": t.n, "wall_clock_s": round(t.wall_clock_s, 2),
            "stub": t.stub, "raw": t.raw,
        } for t in result.tasks],
    }
    if out is None:
        json.dump(payload, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"wrote {out}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m lile.teach.eval")
    p.add_argument("--endpoint", default="http://127.0.0.1:8768/v1",
                   help="OpenAI-compatible endpoint root (include /v1)")
    p.add_argument("--model", required=True,
                   help="Model name passed to the endpoint")
    p.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,gsm8k",
                   help="Comma-separated lm-eval tasks")
    p.add_argument("--code-tasks", default="humaneval_plus",
                   help="Comma-separated code tasks (evalplus)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=100,
                   help="Samples per task; 100 = research-check, 250+ = promotion")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path; prints to stdout if omitted")
    args = p.parse_args()

    lm_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    code_tasks = [t.strip() for t in args.code_tasks.split(",") if t.strip()]
    result = run(args.endpoint, args.model, lm_tasks, code_tasks,
                 args.limit, args.batch_size)
    _emit(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
