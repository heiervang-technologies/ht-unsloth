"""Tutor → student distillation runner.

Pipeline:

  1. Load seed prompts (JSONL with `domain`, `split`, `prompt`).
  2. For each `split=train` prompt, call the tutor (OpenRouter → gpt-oss-120b)
     to produce a gold response.
  3. Save all (prompt, response) to `tutor_responses.jsonl`.
  4. For each `split=eval` prompt, capture a **pre-training** student
     generation via the lile daemon's /v1/chat/completions.
  5. POST each tutor pair to the daemon /v1/train as an SFT sample, waiting
     for the commit token so we observe loss per step.
  6. After all train samples commit, capture a **post-training** student
     generation for each eval prompt.
  7. Emit `results.json` summarizing commit_cursor delta, train losses, and
     pre/post student answers for eval prompts.

Usage:
    OPENROUTER_API_KEY=... \\
    .venv/bin/python -m lile.teach.tutor.run \\
        --prompts lile/teach/tutor/seed_prompts.jsonl \\
        --daemon http://127.0.0.1:8766 \\
        --tutor openai/gpt-oss-120b \\
        --out lile_data/tutor_run_01

The script is intentionally small, synchronous, and uses only stdlib +
`httpx`. No training framework glue; everything goes through the daemon's
HTTP surface, so the daemon's safety machinery (queue, mode_lock, KL
anchor) is engaged the same way a live user would engage it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("tutor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


TUTOR_SYSTEM = (
    "You are an expert tutor. Answer the student's question with a clear, "
    "correct, and reasonably concise explanation. Show key steps or reasoning. "
    "For code, return a complete minimal implementation. Do not ask clarifying "
    "questions — give your best answer to the question as stated."
)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_tutor(client: httpx.Client, model: str, prompt: str,
               api_key: str, timeout: float = 120.0) -> str:
    """One tutor call via OpenRouter. Returns the assistant message text."""
    r = client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/heiervang-technologies/ht-unsloth",
            "X-Title": "lile tutor distillation",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": TUTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def call_student(client: httpx.Client, daemon: str, prompt: str,
                 max_tokens: int = 512) -> str:
    r = client.post(
        f"{daemon}/v1/chat/completions",
        json={
            "model": "student",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        },
        timeout=180.0,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # Qwen3.5 reasoning models emit their answer on `reasoning_content` when
    # no explicit thinking-mode suffix is requested; `content` can be empty.
    # Fall back so we see something in the eval.
    return msg.get("content") or msg.get("reasoning_content") or ""


def train_one(client: httpx.Client, daemon: str, prompt: str, response: str,
              objective: str = "sft", timeout: float = 180.0) -> dict[str, Any]:
    """Submit one SFT sample, wait for commit, return the trajectory entry."""
    payload = {
        "objective": objective,
        "samples": [{"prompt": prompt, "response": response}],
    }
    r = client.post(f"{daemon}/v1/train", json=payload, timeout=timeout)
    r.raise_for_status()
    submit = r.json()
    token = submit.get("commit_token", submit.get("token"))
    if token is None:
        raise RuntimeError(f"no commit token in /v1/train response: {submit}")

    # Block until the step commits.
    w = client.post(
        f"{daemon}/v1/wait",
        params={"token": token, "timeout": timeout},
        timeout=timeout + 5,
    )
    w.raise_for_status()
    wait_result = w.json()
    if not wait_result.get("committed"):
        raise RuntimeError(f"train step did not commit: {wait_result}")

    # Pull the latest trajectory entry (which should be this step).
    t = client.get(f"{daemon}/v1/state/trajectory/tail", params={"n": 1},
                   timeout=10.0)
    t.raise_for_status()
    events = t.json().get("events", [])
    return events[-1] if events else {}


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m lile.teach.tutor.run")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--daemon", default="http://127.0.0.1:8766")
    p.add_argument("--tutor", default="openai/gpt-oss-120b")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of train prompts (for dry-runs)")
    p.add_argument("--eval-max-tokens", type=int, default=400)
    p.add_argument("--objective", default="sft")
    p.add_argument("--skip-pre-eval", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY (or OPENAI_API_KEY) must be set")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_prompts(args.prompts)
    train_rows = [r for r in rows if r.get("split") == "train"]
    eval_rows = [r for r in rows if r.get("split") == "eval"]
    if args.limit is not None:
        train_rows = train_rows[: args.limit]

    log.info("loaded %d prompts (%d train, %d eval)",
             len(rows), len(train_rows), len(eval_rows))

    client = httpx.Client()

    # 0. Daemon health.
    r = client.get(f"{args.daemon}/health", timeout=5.0)
    r.raise_for_status()
    health_before = r.json()
    log.info("daemon before: %s", health_before)

    # 1. Tutor generations for train rows.
    tutor_path = args.out / "tutor_responses.jsonl"
    tutor_pairs: list[dict[str, Any]] = []
    t0 = time.time()
    with tutor_path.open("w") as f:
        for i, row in enumerate(train_rows):
            prompt = row["prompt"]
            try:
                resp = call_tutor(client, args.tutor, prompt, api_key)
            except Exception as exc:
                log.warning("[%d/%d] tutor failed for %r: %s",
                            i + 1, len(train_rows), row.get("domain"), exc)
                continue
            entry = {
                "domain": row["domain"],
                "prompt": prompt,
                "response": resp,
            }
            tutor_pairs.append(entry)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            log.info("[%d/%d] tutor %s — %d chars",
                     i + 1, len(train_rows), row["domain"], len(resp))
    log.info("tutor generation done: %d pairs in %.1fs",
             len(tutor_pairs), time.time() - t0)

    # 2. Pre-training student generations on eval rows.
    pre_evals: list[dict[str, Any]] = []
    if not args.skip_pre_eval:
        log.info("collecting pre-training student eval responses...")
        for row in eval_rows:
            try:
                ans = call_student(client, args.daemon, row["prompt"],
                                   max_tokens=args.eval_max_tokens)
            except Exception as exc:
                log.warning("pre-eval failed for %r: %s", row["domain"], exc)
                ans = f"<error: {exc}>"
            pre_evals.append({
                "domain": row["domain"],
                "prompt": row["prompt"],
                "pre": ans,
            })
    (args.out / "eval_pre.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in pre_evals) + "\n"
        if pre_evals else ""
    )

    # 3. Train one at a time, block on commit, capture trajectory entry.
    train_log_path = args.out / "train_log.jsonl"
    train_log: list[dict[str, Any]] = []
    if not args.skip_train:
        log.info("training %d samples one at a time...", len(tutor_pairs))
        with train_log_path.open("w") as f:
            for i, pair in enumerate(tutor_pairs):
                t_step = time.time()
                try:
                    entry = train_one(
                        client, args.daemon,
                        pair["prompt"], pair["response"],
                        objective=args.objective,
                    )
                except Exception as exc:
                    log.warning("[%d/%d] train failed: %s",
                                i + 1, len(tutor_pairs), exc)
                    continue
                entry["domain"] = pair["domain"]
                entry["prompt_chars"] = len(pair["prompt"])
                entry["response_chars"] = len(pair["response"])
                entry["wall_s"] = time.time() - t_step
                train_log.append(entry)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                loss = entry.get("loss")
                log.info("[%d/%d] %s loss=%s wall=%.1fs",
                         i + 1, len(tutor_pairs), pair["domain"],
                         f"{loss:.4f}" if isinstance(loss, float) else loss,
                         entry["wall_s"])

    # 4. Post-training student generations on eval rows.
    post_evals: list[dict[str, Any]] = []
    log.info("collecting post-training student eval responses...")
    for row in eval_rows:
        try:
            ans = call_student(client, args.daemon, row["prompt"],
                               max_tokens=args.eval_max_tokens)
        except Exception as exc:
            log.warning("post-eval failed for %r: %s", row["domain"], exc)
            ans = f"<error: {exc}>"
        post_evals.append({
            "domain": row["domain"],
            "prompt": row["prompt"],
            "post": ans,
        })
    (args.out / "eval_post.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in post_evals) + "\n"
    )

    # 5. Final daemon state + summary.
    health_after = client.get(f"{args.daemon}/health", timeout=5.0).json()
    losses = [e["loss"] for e in train_log if isinstance(e.get("loss"), (int, float))]
    summary = {
        "daemon_before": health_before,
        "daemon_after": health_after,
        "train_samples": len(train_log),
        "tutor_pairs": len(tutor_pairs),
        "eval_samples": len(eval_rows),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary))

    # 6. Side-by-side pre/post for human inspection.
    side_by_side_path = args.out / "eval_side_by_side.md"
    lines = ["# Tutor distillation — pre/post eval\n",
             f"Trained {len(train_log)} samples on domains: "
             f"{sorted({e['domain'] for e in train_log})}\n",
             f"First-step loss: {summary['loss_first']}  |  "
             f"Last-step loss: {summary['loss_last']}  |  "
             f"Mean: {summary['loss_mean']}\n\n"]
    post_by_prompt = {(e["domain"], e["prompt"]): e["post"] for e in post_evals}
    for pre in pre_evals:
        key = (pre["domain"], pre["prompt"])
        post = post_by_prompt.get(key, "<missing>")
        lines.append(f"## [{pre['domain']}] {pre['prompt']}\n\n")
        lines.append(f"**Pre:**\n\n```\n{pre['pre']}\n```\n\n")
        lines.append(f"**Post:**\n\n```\n{post}\n```\n\n")
        lines.append("---\n\n")
    side_by_side_path.write_text("".join(lines))
    log.info("wrote %s", side_by_side_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
