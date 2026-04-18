"""Build `mixed_500.jsonl` per the spec in this directory's README.

Composition (exact):
    200 binary  | KTO
    150 nl_critique | CoH
    100 rewrite | weighted_sft (w=3.0)
     50 preferred | hinge

Each kind split 25/25/25/25 across math / code / common-sense / general
(rounded by largest remainder to preserve per-kind totals).

Dependencies:
    - datasets (HF): GSM8K, MBPP-sanitized, HellaSwag, MMLU. Imported lazily
      inside `load_source_prompts` so this module stays importable on a
      torchless / datasets-less CI runner.
    - openai (Python SDK): imported lazily inside the teacher client. The
      teacher talks to either Anthropic's OpenAI-compat endpoint
      (ANTHROPIC_API_KEY) or OpenAI (OPENAI_API_KEY, gpt-4o-mini default).
    - A cold Qwen3 daemon on a disjoint port for `cold_generate`. This is
      the remaining runtime dependency; see `cold_generate` docstring.

Deterministic on `--seed` (default 42). Shuffling happens once at write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


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


# ------------------------------------------------------------------ prompts
def _hellaswag_prompt(ctx: str) -> str:
    """Frame a HellaSwag context as a single-turn user prompt."""
    ctx = (ctx or "").strip()
    return (
        f"Continue the following passage with one plausible next sentence:\n\n"
        f"{ctx}\n\nNext:"
    )


def _mmlu_prompt(item: dict[str, Any]) -> str:
    """Frame an MMLU row as a single-turn user prompt."""
    q = (item.get("question") or "").strip()
    choices = item.get("choices") or []
    if choices:
        labeled = "\n".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
        )
        return (
            f"{q}\n\n{labeled}\n\n"
            "Answer with the single letter (A, B, C, or D) and a brief justification."
        )
    return q


def load_source_prompts(domain: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Return `n` prompts for `domain` with their ground-truth refs.

    Each returned dict has keys: `prompt`, `ground_truth` (domain-specific),
    `source_idx` (absolute index into the HF split).

    Deterministic on `seed` via `random.Random(seed).sample` over the source
    slice `[SOURCE_OFFSET, SOURCE_OFFSET + n_window)`, where `n_window` is a
    capped view into the split so the eval slice `[0, 250)` stays disjoint.
    """
    if domain not in SOURCES:
        raise KeyError(f"unknown domain {domain!r}; known: {sorted(SOURCES)}")
    spec = SOURCES[domain]

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover - scaffold guard
        raise ImportError(
            "load_source_prompts needs the `datasets` package. "
            "Install with `uv pip install datasets`."
        ) from e

    ds = load_dataset(
        spec["hf_dataset"],
        spec["hf_config"],
        split=spec["hf_split"],
    )
    total = len(ds)
    # Pool candidate indices from the disjoint range; cap at what the split has.
    pool_end = min(total, SOURCE_OFFSET + max(n * 4, 200))
    if pool_end <= SOURCE_OFFSET:
        raise ValueError(
            f"{domain}: source split has {total} rows, cannot allocate the "
            f"disjoint range [{SOURCE_OFFSET}, ...) (eval uses [0, 250))"
        )
    pool = list(range(SOURCE_OFFSET, pool_end))
    if len(pool) < n:
        raise ValueError(
            f"{domain}: asked for {n} prompts but the disjoint pool only has "
            f"{len(pool)} rows (split total = {total})"
        )
    rng = random.Random(seed)
    idxs = rng.sample(pool, n)

    pfield = spec["prompt_field"]
    gfield = spec["ground_truth_field"]
    out: list[dict[str, Any]] = []
    for i in idxs:
        row = ds[i]
        if domain == "common-sense":
            prompt = _hellaswag_prompt(row[pfield])
        elif domain == "general":
            prompt = _mmlu_prompt(row)
        else:
            prompt = (row[pfield] or "").strip()
        out.append({
            "prompt": prompt,
            "ground_truth": row.get(gfield),
            "source_idx": int(i),
        })
    # Assert disjointness — defensive; tests pin this too.
    for rec in out:
        assert rec["source_idx"] >= SOURCE_OFFSET, rec
    return out


# ------------------------------------------------------------------ cold model
def cold_generate(prompt: str, domain: str, endpoint: str, seed: int) -> str:
    """Call the cold base model (no adapter) via OpenAI-compat endpoint.

    This stub is intentionally not implemented until a cold Qwen3-9B snapshot
    and a dedicated cold daemon are available. Recipe to fill it in:

    1. Produce a cold snapshot (no active adapter). Either start the trainer
       daemon with a clean base, POST an empty `/v1/snapshots/save`, or
       `snapshot_load` a committed `cold_base` snapshot at runtime. The goal
       is to pin weights == base model + zero delta.
    2. Spin a *second* daemon instance on a port disjoint from the training
       daemon (e.g. cold on :8768, trainer on :8766). If VRAM fits, same GPU
       is fine; otherwise point at a second device via CUDA_VISIBLE_DEVICES.
    3. POST `{endpoint}/chat/completions` with
       `{'model': 'current', 'messages': [{'role': 'user', 'content': prompt}]}`
       — the lile server ignores the model name and uses the live weights.
       Pass `seed`, `temperature`, `top_p`, and `max_tokens` per domain.
    4. Wire this function to call that endpoint (use `httpx` or `openai`).

    Generation params per-domain (from README):
        math:         max_tokens=512, temperature=0.7, top_p=0.95
        code:         max_tokens=768, temperature=0.7, top_p=0.95
        common-sense: max_tokens=128, temperature=0.7, top_p=0.95
        general:      max_tokens=256, temperature=0.7, top_p=0.95
    """
    raise NotImplementedError(
        "cold_generate requires a cold Qwen3 daemon on a disjoint port. "
        "See the docstring for the cold-daemon recipe."
    )


# ------------------------------------------------------------------ teacher client
class TeacherClient(Protocol):
    """Protocol for the teacher — any object with these 4 methods plugs in."""

    def label_binary(self, *, prompt: str, response: str, domain: str,
                     ground_truth: Any) -> str: ...

    def rewrite(self, *, prompt: str, response: str, domain: str) -> str: ...

    def critique(self, *, prompt: str, response: str, domain: str) -> str: ...

    def preferred_pair(self, *, prompt: str, response_a: str, response_b: str,
                       domain: str) -> tuple[str, str]: ...


_BINARY_SYSTEM = (
    "You grade a single model response as 'up' (helpful and correct) or "
    "'down' (incorrect, unsafe, or unhelpful). Reply with exactly 'up' or 'down'."
)
_REWRITE_SYSTEM = (
    "Rewrite the response so it is correct, helpful, and preserves the "
    "original format. Reply with only the rewritten response — no preface."
)
_CRITIQUE_SYSTEM = (
    "Write a 1-3 sentence specific, non-vacuous critique of the response "
    "explaining what is wrong or could be improved. Reply with only the critique."
)
_PREF_SYSTEM = (
    "Compare two candidate responses to the prompt and pick the better one. "
    "Reply with exactly 'A' or 'B'."
)


@dataclass
class OpenAICompatTeacher:
    """Default teacher client: OpenAI-compat SDK with pluggable base_url."""

    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512

    def _chat(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - scaffold guard
            raise ImportError(
                "OpenAICompatTeacher needs the `openai` package. "
                "Install with `uv pip install openai`."
            ) from e
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def label_binary(self, *, prompt: str, response: str, domain: str,
                     ground_truth: Any) -> str:
        gt = "" if ground_truth is None else f"\n\nReference: {ground_truth!s}"
        user = f"Prompt:\n{prompt}\n\nResponse:\n{response}{gt}"
        out = self._chat(_BINARY_SYSTEM, user).lower()
        return "up" if "up" in out and "down" not in out else "down"

    def rewrite(self, *, prompt: str, response: str, domain: str) -> str:
        user = (
            f"Prompt:\n{prompt}\n\n"
            f"Original response:\n{response}\n\nRewritten response:"
        )
        return self._chat(_REWRITE_SYSTEM, user)

    def critique(self, *, prompt: str, response: str, domain: str) -> str:
        user = f"Prompt:\n{prompt}\n\nResponse:\n{response}\n\nCritique:"
        return self._chat(_CRITIQUE_SYSTEM, user)

    def preferred_pair(self, *, prompt: str, response_a: str,
                       response_b: str, domain: str) -> tuple[str, str]:
        user = (
            f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\n"
            f"Response B:\n{response_b}\n\nWhich is better? Reply A or B."
        )
        pick = self._chat(_PREF_SYSTEM, user).strip().upper()
        # Tie / malformed → keep declared order so the run is deterministic.
        if pick.startswith("B"):
            return response_b, response_a
        return response_a, response_b


def default_teacher(model: str) -> TeacherClient:
    """Construct a teacher client from env.

    Lookup order:
    1. ANTHROPIC_API_KEY → Anthropic's OpenAI-compat endpoint with `model`.
    2. OPENAI_API_KEY    → OpenAI; if `model` looks Anthropic (claude-*),
                           fall back to `gpt-4o-mini`.

    Raises `RuntimeError` with a clear message if neither is set.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return OpenAICompatTeacher(
            model=model,
            api_key=anthropic_key,
            base_url="https://api.anthropic.com/v1/",
        )
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        fallback = "gpt-4o-mini" if model.startswith("claude") else model
        return OpenAICompatTeacher(model=fallback, api_key=openai_key)
    raise RuntimeError(
        "No teacher API key in env. Set ANTHROPIC_API_KEY "
        "(preferred, matches README) or OPENAI_API_KEY (falls back to "
        "gpt-4o-mini when --teacher-model is a Claude model)."
    )


# ------------------------------------------------------------------ pipeline
def _record_id(idx: int) -> str:
    return f"mixed_500-{idx:04d}"


PromptLoader = Callable[[str, int, int], list[dict[str, Any]]]
ColdGen = Callable[[str, str, str, int], str]


def build_records(
    endpoint: str,
    teacher_model: str,
    seed: int,
    composition: Composition = COMPOSITION,
    *,
    teacher: TeacherClient | None = None,
    prompts_fn: PromptLoader | None = None,
    cold_fn: ColdGen | None = None,
) -> list[dict[str, Any]]:
    """Compose the full stream in declared order. Shuffling happens at write.

    Injection points (all optional — defaults construct the production impls):
      * `teacher`      — custom `TeacherClient` (tests pass a mock).
      * `prompts_fn`   — custom prompt loader (tests pass a fixture slice).
      * `cold_fn`      — custom cold-model callable (tests pass a stub).
    """
    teacher = teacher if teacher is not None else default_teacher(teacher_model)
    prompts_fn = prompts_fn if prompts_fn is not None else load_source_prompts
    cold_fn = cold_fn if cold_fn is not None else cold_generate

    rng = random.Random(seed)
    split = composition.per_kind_by_domain()
    records: list[dict[str, Any]] = []
    idx = 0

    for kind, _total in composition.by_kind:
        for domain in composition.domains:
            n = split[kind][domain]
            if n == 0:
                continue
            prompts = prompts_fn(domain, n, seed + hash(kind) % 10_000)
            for item in prompts:
                idx += 1
                prompt = item["prompt"]
                response = cold_fn(prompt, domain, endpoint, seed + idx)
                base: dict[str, Any] = {
                    "response_id": _record_id(idx),
                    "feedback_kind": kind,
                    "prompt": prompt,
                    "response": response,
                    "domain": domain,
                    "source_idx": item["source_idx"],
                }
                if kind == "binary":
                    base["value"] = teacher.label_binary(
                        prompt=prompt, response=response, domain=domain,
                        ground_truth=item.get("ground_truth"),
                    )
                elif kind == "rewrite":
                    base["better_response"] = teacher.rewrite(
                        prompt=prompt, response=response, domain=domain,
                    )
                    base["weight"] = 3.0
                elif kind == "preferred":
                    alt = cold_fn(prompt, domain, endpoint, seed + idx + 10_000)
                    chosen, rejected = teacher.preferred_pair(
                        prompt=prompt, response_a=response, response_b=alt,
                        domain=domain,
                    )
                    base["chosen"] = chosen
                    base["rejected"] = rejected
                elif kind == "nl_critique":
                    base["critique"] = teacher.critique(
                        prompt=prompt, response=response, domain=domain,
                    )
                    # 20% upgrade to nl_critique_with_rewrite per README.
                    if rng.random() < 0.20:
                        base["feedback_kind"] = "nl_critique_with_rewrite"
                        base["better_response"] = teacher.rewrite(
                            prompt=prompt, response=response, domain=domain,
                        )
                else:
                    raise AssertionError(f"unknown kind {kind!r}")
                records.append(base)

    rng.shuffle(records)
    return records


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m lile.teach.replay_streams.build_mixed_500",
    )
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
