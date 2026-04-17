"""§11 — Ranking-reliability benchmark for the implicit critique reward r_c.

This is the gating empirical question for CCPD v2 (see ``LIVELEARN.md`` §11). We
sample ``k`` candidate responses from ``π(·|x, c)``, score each with the
length-normalised log-ratio

    r_c(y) = β · [ log π(y|x, c) − log π(y|x) ] / |y|

and measure the Spearman rank correlation between the ``r_c`` rank and a
ground-truth rank derived from a deterministic per-prompt criterion (e.g. "fewer
words is better" for a "be more concise" critique).

Usage
-----

    python benchmarks/ccpd_ranking_reliability.py \\
        --model unsloth/qwen3-0.6b-unsloth-bnb-4bit \\
        --k 8 \\
        --max-new-tokens 96 \\
        --out benchmarks/ccpd_ranking_reliability.json

Decision thresholds (from §11)
------------------------------

* mean Spearman > 0.5  → ship CCPD v2 as the T2.1 default.
* mean Spearman 0.2-0.5 → ship CCPD v2 only with k ≥ 8; T2.2 hinge primary.
* mean Spearman < 0.2  → CCPD v2 ships as opt-in experimental flag.

Design notes
------------

1. We disable Qwen3's "thinking" mode (``enable_thinking=False``) so r_c reflects
   the full visible response, not a hidden CoT preamble.
2. Critique is placed in the *system* slot of the chat template; the prompt is in
   the user slot. The "without critique" comparison uses the user slot only.
3. We reuse the same sampled candidates for r_c scoring AND ground-truth scoring
   — that's the experiment: do the rankings agree?
4. Length-normalisation in r_c is essential to avoid trivially preferring short
   responses; we divide log-prob sums by the candidate's token count.
5. The dataset is small but covers heterogeneous critique types (length,
   formatting, content, style). All criteria are deterministic functions; no
   judge LLM dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

# Triton spam suppression for cleaner logs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from unsloth import FastLanguageModel  # noqa: E402


# --- Dataset ---------------------------------------------------------------

@dataclass
class CritiqueCase:
    """One (prompt, critique, ground-truth-criterion) tuple."""

    name: str
    prompt: str
    critique: str
    # The *positive* score for a candidate response — higher == better-satisfies-critique.
    score: Callable[[str], float]
    notes: str = ""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def _has_bullets(text: str) -> bool:
    return bool(re.search(r"^\s*[-*\u2022]\s", text, re.MULTILINE))


def _has_numbered_list(text: str) -> bool:
    return bool(re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE))


def _has_code_fence(text: str) -> bool:
    return "```" in text or bool(re.search(r"^[ \t]{4,}\S", text, re.MULTILINE))


def _is_yes_no(text: str) -> bool:
    stripped = text.strip().lower().rstrip(".!?")
    return stripped in {"yes", "no", "yeah", "nope", "y", "n"}


def _starts_with(text: str, prefix: str) -> bool:
    return text.strip().lower().startswith(prefix.lower())


def build_dataset() -> list[CritiqueCase]:
    """Hand-crafted but heterogeneous dataset.

    Each case is designed so the ground-truth score is a deterministic function
    of the response text. This avoids judge-LLM dependency for the gating
    benchmark; it does mean our criteria are surface-form proxies for the actual
    critique meaning, which is conservative — if r_c ranks well against
    surface-form criteria, that's encouraging; if it ranks poorly, the same
    criteria are at least the easy case for a critique to land.
    """
    return [
        # --- Length / conciseness critiques -------------------------------
        CritiqueCase(
            name="concise_paris",
            prompt="Tell me about the city of Paris.",
            critique="Be extremely concise. Use as few words as possible.",
            score=lambda r: -_word_count(r),
            notes="shorter is better",
        ),
        CritiqueCase(
            name="concise_history",
            prompt="What caused World War I?",
            critique="Be extremely concise. One sentence maximum.",
            score=lambda r: -_word_count(r),
        ),
        CritiqueCase(
            name="concise_explain",
            prompt="Explain photosynthesis.",
            critique="Be very brief — a single sentence under twenty words.",
            score=lambda r: -_word_count(r),
        ),
        CritiqueCase(
            name="one_word_answer",
            prompt="What is 2+2?",
            critique="Answer in exactly one word, nothing else.",
            score=lambda r: 1.0 if _is_yes_no(r) or _word_count(r) == 1 else -_word_count(r),
        ),
        # --- Format critiques ---------------------------------------------
        CritiqueCase(
            name="bullets_pros",
            prompt="What are the pros and cons of Python?",
            critique="Format your answer as a bulleted list.",
            score=lambda r: float(_has_bullets(r)) - 0.001 * _word_count(r),
        ),
        CritiqueCase(
            name="bullets_steps",
            prompt="How do I bake a cake?",
            critique="Format your answer as a bulleted list of steps.",
            score=lambda r: float(_has_bullets(r)) - 0.001 * _word_count(r),
        ),
        CritiqueCase(
            name="numbered_list",
            prompt="What are the planets of our solar system?",
            critique="Use a numbered list.",
            score=lambda r: float(_has_numbered_list(r)) - 0.001 * _word_count(r),
        ),
        CritiqueCase(
            name="no_code",
            prompt="Explain how to compute factorial.",
            critique="Do not include any code. Use prose only.",
            score=lambda r: -float(_has_code_fence(r)),
        ),
        CritiqueCase(
            name="include_code",
            prompt="Show me how to print hello world.",
            critique="Provide your answer as a code block.",
            score=lambda r: float(_has_code_fence(r)),
        ),
        # --- Style critiques ----------------------------------------------
        CritiqueCase(
            name="start_with_certainly",
            prompt="What is the capital of Japan?",
            critique="Begin your response with the word 'Certainly'.",
            score=lambda r: float(_starts_with(r, "certainly")) - 0.001 * _word_count(r),
        ),
        CritiqueCase(
            name="start_with_honestly",
            prompt="Is jogging good for you?",
            critique="Start your answer with 'Honestly'.",
            score=lambda r: float(_starts_with(r, "honestly")) - 0.001 * _word_count(r),
        ),
        CritiqueCase(
            name="end_with_question",
            prompt="Tell me about cats.",
            critique="End your reply with a question mark.",
            score=lambda r: float(r.strip().endswith("?")),
        ),
        # --- Content / restriction critiques ------------------------------
        CritiqueCase(
            name="no_apology",
            prompt="Why is the sky blue?",
            critique="Do not start with apologies or hedging language. Be direct.",
            score=lambda r: -float(any(
                _starts_with(r, w) for w in ("sorry", "i apologize", "apologies", "actually", "well,")
            )),
        ),
        CritiqueCase(
            name="metric_units",
            prompt="How tall is Mount Everest?",
            critique="Use metric units (metres / kilometres) only, never feet or miles.",
            score=lambda r: (
                float(("metre" in r.lower()) or ("meter" in r.lower()) or ("km" in r.lower()) or ("metr" in r.lower()))
                - float(("feet" in r.lower()) or ("ft" in r.lower()) or ("mile" in r.lower()))
            ),
        ),
        CritiqueCase(
            name="emoji_only_no",
            prompt="What's your favourite colour?",
            critique="Do not use any emoji or unicode pictographs.",
            score=lambda r: -float(any(ord(ch) > 0x2600 for ch in r)),
        ),
    ]


# --- Logprob computation ---------------------------------------------------

def _format_with_critique(tokenizer, prompt: str, critique: str | None) -> str:
    """Build the chat-template string. ``critique=None`` ⇒ no system message."""
    msgs: list[dict[str, str]] = []
    if critique:
        msgs.append({"role": "system", "content": critique})
    msgs.append({"role": "user", "content": prompt})
    # Qwen3 chat template supports ``enable_thinking``; we disable it so
    # candidates are pure responses (no <think>...</think> preamble).
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    return text


@torch.no_grad()
def _logprob_of_completion(model, tokenizer, prefix_text: str, completion: str) -> tuple[float, int]:
    """Compute sum log p(completion | prefix_text) and the completion's token count.

    Returns ``(logprob_sum, n_tokens)``. ``logprob_sum / n_tokens`` is the
    length-normalised average log-prob.
    """
    device = next(model.parameters()).device
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prefix_text + completion, return_tensors="pt").input_ids.to(device)
    # The completion's tokens are the suffix.
    n_prefix = prefix_ids.shape[1]
    if full_ids.shape[1] <= n_prefix:
        return 0.0, 0
    n_completion = full_ids.shape[1] - n_prefix
    # Forward; logits are shifted by one for next-token prediction.
    out = model(full_ids).logits  # [1, T, V]
    log_probs = torch.log_softmax(out[0, n_prefix - 1 : -1, :].float(), dim=-1)  # [n_completion, V]
    target = full_ids[0, n_prefix:]  # [n_completion]
    chosen = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [n_completion]
    return float(chosen.sum().item()), n_completion


# --- Generation -----------------------------------------------------------

@torch.no_grad()
def _sample_candidates(
    model, tokenizer, prefix_text: str, k: int, max_new_tokens: int, temperature: float, seed: int,
) -> list[str]:
    device = next(model.parameters()).device
    inputs = tokenizer(prefix_text, return_tensors="pt").to(device)
    candidates: list[str] = []
    # Distinct seeds per candidate so we get diversity even at same temperature.
    for i in range(k):
        torch.manual_seed(seed + i)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        candidates.append(text.strip())
    return candidates


# --- Spearman -------------------------------------------------------------

def _ranks(xs: Sequence[float]) -> list[float]:
    """Average-rank-on-ties to match scipy.stats.rankdata default."""
    arr = np.array(xs, dtype=np.float64)
    order = arr.argsort()
    ranks = np.empty_like(arr)
    ranks[order] = np.arange(len(arr))
    # Average ties.
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    for r, group in zip(ranks, inv):
        sums[group] += r
    avg = sums / counts
    return [float(avg[g]) for g in inv]


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    rxa, rya = np.array(rx), np.array(ry)
    rxa -= rxa.mean()
    rya -= rya.mean()
    denom = float((rxa.std() * rya.std()) * len(rxa))
    if denom == 0.0:
        return float("nan")
    return float((rxa * rya).sum() / denom)


# --- Run ------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    n_candidates: int
    spearman: float
    r_c_top: str
    r_c_bottom: str
    gt_top: str
    gt_bottom: str
    candidates: list[str]
    r_c_scores: list[float]
    gt_scores: list[float]
    elapsed_s: float


def run_benchmark(
    model_name: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    out_path: Path,
    max_seq_length: int = 1024,
) -> dict:
    print(f"[load] {model_name}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    print(f"[load] OK in {time.time() - t0:.1f}s; vram={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    cases = build_dataset()
    print(f"[run]  {len(cases)} cases × k={k} candidates")

    results: list[CaseResult] = []
    for ci, case in enumerate(cases):
        t1 = time.time()
        prefix_with = _format_with_critique(tokenizer, case.prompt, case.critique)
        prefix_without = _format_with_critique(tokenizer, case.prompt, None)

        candidates = _sample_candidates(
            model, tokenizer, prefix_with, k=k,
            max_new_tokens=max_new_tokens, temperature=temperature,
            seed=seed + ci * 1000,
        )

        # Score each candidate
        r_c_scores: list[float] = []
        gt_scores: list[float] = []
        for cand in candidates:
            ll_with, n = _logprob_of_completion(model, tokenizer, prefix_with, cand)
            ll_without, _ = _logprob_of_completion(model, tokenizer, prefix_without, cand)
            if n == 0:
                r_c = 0.0
            else:
                r_c = (ll_with - ll_without) / n
            r_c_scores.append(r_c)
            gt_scores.append(float(case.score(cand)))

        rho = _spearman(r_c_scores, gt_scores)
        elapsed = time.time() - t1

        # Pick top/bottom-by-rc and top/bottom-by-gt for visual inspection.
        rc_order = np.argsort(r_c_scores)
        gt_order = np.argsort(gt_scores)
        results.append(CaseResult(
            name=case.name,
            n_candidates=len(candidates),
            spearman=rho,
            r_c_top=candidates[int(rc_order[-1])],
            r_c_bottom=candidates[int(rc_order[0])],
            gt_top=candidates[int(gt_order[-1])],
            gt_bottom=candidates[int(gt_order[0])],
            candidates=candidates,
            r_c_scores=r_c_scores,
            gt_scores=gt_scores,
            elapsed_s=elapsed,
        ))
        print(f"[{ci+1:02d}/{len(cases)}] {case.name:24s}  ρ={rho:+.3f}  ({elapsed:.1f}s)")

    valid = [r.spearman for r in results if not (r.spearman != r.spearman)]
    mean = statistics.mean(valid) if valid else float("nan")
    median = statistics.median(valid) if valid else float("nan")
    pos_frac = sum(1 for r in valid if r > 0) / max(len(valid), 1)

    summary = {
        "model": model_name,
        "k": k,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "n_cases": len(results),
        "n_valid": len(valid),
        "mean_spearman": mean,
        "median_spearman": median,
        "frac_positive": pos_frac,
        "decision": _decision(mean),
        "cases": [asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f"[summary] mean ρ = {mean:+.3f}  median ρ = {median:+.3f}  positive {pos_frac:.0%}")
    print(f"[summary] decision: {summary['decision']}")
    print(f"[summary] wrote {out_path}")
    return summary


def _decision(mean: float) -> str:
    if mean != mean:  # NaN
        return "INDETERMINATE"
    if mean > 0.5:
        return "SHIP_CCPD_V2_DEFAULT"
    if mean >= 0.2:
        return "SHIP_CCPD_V2_GATED"
    return "FALLBACK_T2_2_HINGE"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/qwen3-0.6b-unsloth-bnb-4bit")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument(
        "--out", type=Path,
        default=Path("benchmarks/ccpd_ranking_reliability.json"),
    )
    args = p.parse_args()
    run_benchmark(
        model_name=args.model,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        out_path=args.out,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
