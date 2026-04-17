"""Build `mixed_500.jsonl` per the spec in this directory's README.

Composition (exact):
    200 binary  | KTO
    150 nl_critique | CoH
    100 rewrite | weighted_sft (w=3.0)
     50 preferred | hinge

Each kind split 25/25/25/25 across math / code / common-sense / general
(rounded by largest remainder to preserve per-kind totals).

Dependencies:
    - datasets (HF): GSM8K, HumanEval+MBPP, HellaSwag, MMLU.
    - An OpenAI-compatible endpoint for the cold base model (the lile daemon
      on :8768 with no active adapter — temporarily `snapshot_load` a
      `cold_base` snapshot, or run a parallel base-model-only server).
    - A teacher endpoint for labels / critiques / rewrites. Default:
      claude-opus-4-7 via the Anthropic API.

This script is idempotent: it caches teacher responses by (prompt, response,
kind) hash in `.cache/teacher/*.json` next to the output file. Re-running
with the same --seed and --out regenerates the stream deterministically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# ------------------------------------------------------------------ composition
@dataclass(frozen=True)
class Composition:
    total: int = 500
    by_kind: tuple[tuple[str, int], ...] = (
        ("binary", 200),
        ("nl_critique", 150),
        ("rewrite", 100),
        ("preferred", 50),
    )
    domains: tuple[str, ...] = ("math", "code", "common-sense", "general")

    def per_kind_by_domain(self) -> dict[str, dict[str, int]]:
        """Largest-remainder split of each kind across domains."""
        out: dict[str, dict[str, int]] = {}
        n_domains = len(self.domains)
        for kind, n in self.by_kind:
            base = n // n_domains
            rem = n - base * n_domains
            split = {d: base for d in self.domains}
            # Deterministic tiebreak: award remainder to domains in declared order.
            for d in self.domains[:rem]:
                split[d] += 1
            out[kind] = split
            assert sum(split.values()) == n, (kind, split, n)
        return out


COMPOSITION = Composition()


# ------------------------------------------------------------------ sources
# Prompt-source specs. The generator resolves each to a list of user prompts.
# Prompts are drawn from indices [1000, 1000+N) to stay disjoint from the
# eval harness's `--limit 250` slice (indices [0, 250)).
SOURCE_OFFSET = 1000

SOURCES: dict[str, dict[str, Any]] = {
    "math": {
        "hf_dataset": "gsm8k",
        "hf_config": "main",
        "hf_split": "train",
        "prompt_field": "question",
        "ground_truth_field": "answer",  # GSM8K format: "...#### <number>"
    },
    "code": {
        "hf_dataset": "mbpp",        # HumanEval has no train split; MBPP fills in.
        "hf_config": "sanitized",
        "hf_split": "train",
        "prompt_field": "prompt",
        "ground_truth_field": "test_list",
    },
    "common-sense": {
        "hf_dataset": "hellaswag",
        "hf_config": None,
        "hf_split": "train",
        "prompt_field": "ctx",       # context → "what comes next?"
        "ground_truth_field": "label",
    },
    "general": {
        "hf_dataset": "cais/mmlu",
        "hf_config": "all",
        "hf_split": "test",          # MMLU's only labeled split for non-aux
        "prompt_field": "question",
        "ground_truth_field": "answer",
    },
}


# ------------------------------------------------------------------ io helpers
def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ stubs
#
# The calls below are intentionally stubbed — implementing them requires live
# model endpoints. The run_stream function composes them into a deterministic
# pipeline so swapping stubs for real calls is a mechanical change.
#
# Each stub's docstring describes the exact contract.


def load_source_prompts(domain: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Return `n` prompts for `domain` with their ground-truth refs.

    Each returned dict has keys: `prompt`, `ground_truth` (domain-specific),
    `source_idx` (absolute index into the HF split).
    Deterministic on `seed` — random.Random(seed).sample over the source slice.
    """
    raise NotImplementedError(
        "Wire to datasets.load_dataset(...) per SOURCES[domain]; "
        f"sample from indices [{SOURCE_OFFSET}, {SOURCE_OFFSET}+N) with "
        "random.Random(seed).sample to keep ordering reproducible."
    )


def cold_generate(prompt: str, domain: str, endpoint: str, seed: int) -> str:
    """Call the cold base model (no adapter) via OpenAI-compat endpoint.

    Generation params per-domain (from README):
        math:         max_tokens=512, temperature=0.7, top_p=0.95
        code:         max_tokens=768
        common-sense: max_tokens=128
        general:      max_tokens=256
    """
    raise NotImplementedError(
        "POST to {endpoint}/chat/completions with {'model': 'cold', "
        "'messages': [{'role': 'user', 'content': prompt}], ...}. "
        "Use a snapshot_load of `cold_base` to ensure no adapter is active."
    )


def teacher_label_binary(prompt: str, response: str, domain: str,
                         ground_truth: Any, teacher_model: str) -> str:
    """Return "up" or "down" for binary feedback.

    For math: exact-match on final numeric answer extracted from `response`.
    For code: execute against `ground_truth["test_list"]` in sandboxed subprocess.
    For common-sense / general: teacher rubric check (non-refusal + factual).
    """
    raise NotImplementedError(
        "Ground-truth check for math/code; teacher API for fuzzy domains."
    )


def teacher_rewrite(prompt: str, response: str, domain: str,
                    teacher_model: str) -> str:
    """Teacher-generated better response for `rewrite` records."""
    raise NotImplementedError(
        "Anthropic API call: 'Rewrite to be correct/helpful, preserve format.'"
    )


def teacher_critique(prompt: str, response: str, domain: str,
                     teacher_model: str) -> str:
    """Teacher-generated 1-3 sentence critique for `nl_critique` records."""
    raise NotImplementedError(
        "Anthropic API call: 'Write a 1-3 sentence specific critique.'"
    )


def teacher_preferred_pair(prompt: str, domain: str, endpoint: str,
                           teacher_model: str, seed: int) -> tuple[str, str]:
    """Return (chosen, rejected) for `preferred` records.

    Samples two responses at temperature=0.7; chosen is the one that passes
    ground-truth or teacher rubric. If both pass or both fail, re-sample.
    """
    raise NotImplementedError(
        "Two cold_generate calls with different generation seeds; "
        "teacher_label_binary picks the winner; loop until tiebreak resolves."
    )


# ------------------------------------------------------------------ pipeline
def _record_id(idx: int) -> str:
    return f"mixed_500-{idx:04d}"


def build_records(endpoint: str, teacher_model: str,
                  seed: int) -> list[dict[str, Any]]:
    """Compose the full stream in declared order. Shuffling happens at write."""
    rng = random.Random(seed)
    split = COMPOSITION.per_kind_by_domain()
    records: list[dict[str, Any]] = []
    idx = 0

    for kind, _total in COMPOSITION.by_kind:
        for domain in COMPOSITION.domains:
            n = split[kind][domain]
            prompts = load_source_prompts(domain, n, seed=seed + hash(kind) % 10_000)
            for item in prompts:
                idx += 1
                prompt = item["prompt"]
                response = cold_generate(prompt, domain, endpoint,
                                         seed=seed + idx)
                base: dict[str, Any] = {
                    "response_id": _record_id(idx),
                    "feedback_kind": kind,
                    "prompt": prompt,
                    "response": response,
                    "domain": domain,
                    "source_idx": item["source_idx"],
                }
                if kind == "binary":
                    base["value"] = teacher_label_binary(
                        prompt, response, domain,
                        item["ground_truth"], teacher_model,
                    )
                elif kind == "rewrite":
                    base["better_response"] = teacher_rewrite(
                        prompt, response, domain, teacher_model,
                    )
                    base["weight"] = 3.0
                elif kind == "preferred":
                    chosen, rejected = teacher_preferred_pair(
                        prompt, domain, endpoint, teacher_model,
                        seed=seed + idx,
                    )
                    base["chosen"] = chosen
                    base["rejected"] = rejected
                elif kind == "nl_critique":
                    base["critique"] = teacher_critique(
                        prompt, response, domain, teacher_model,
                    )
                    # 20% upgrade to nl_critique_with_rewrite per README.
                    if rng.random() < 0.20:
                        base["feedback_kind"] = "nl_critique_with_rewrite"
                        base["better_response"] = teacher_rewrite(
                            prompt, response, domain, teacher_model,
                        )
                else:
                    raise AssertionError(f"unknown kind {kind!r}")
                records.append(base)

    rng.shuffle(records)
    return records


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m lile.teach.replay_streams.build_mixed_500")
    p.add_argument("--endpoint", default="http://127.0.0.1:8768/v1",
                   help="OpenAI-compat endpoint of the cold base model")
    p.add_argument("--teacher-model", default="claude-opus-4-7",
                   help="Teacher model used for labels / critiques / rewrites")
    p.add_argument("--out",
                   default="lile/teach/replay_streams/mixed_500.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="print the composition split and exit without calling models")
    args = p.parse_args()

    if args.dry_run:
        split = COMPOSITION.per_kind_by_domain()
        print(f"total: {COMPOSITION.total}")
        for kind, n in COMPOSITION.by_kind:
            print(f"  {kind:>12}  {n:>4}  "
                  + " ".join(f"{d}={split[kind][d]}" for d in COMPOSITION.domains))
        print(f"seed: {args.seed}")
        print(f"source offset: [{SOURCE_OFFSET}, {SOURCE_OFFSET}+N)")
        return 0

    records = build_records(args.endpoint, args.teacher_model, seed=args.seed)
    out = Path(args.out)
    n = _write_jsonl(out, records)
    print(f"wrote {n} records to {out}")
    if n != COMPOSITION.total:
        print(f"WARNING: expected {COMPOSITION.total}, got {n}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
