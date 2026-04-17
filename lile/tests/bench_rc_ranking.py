"""§11 benchmark — does r_c rank reliably?

This is the gating experiment: we measure Spearman rank correlation between
the detached critique-conditional reward r_c and a ground-truth ordering of
candidate responses to (x, c).

The ground-truth oracle here is a **critique-satisfaction rule**: we
synthesize pairs where we know what "satisfies the critique" means.
Specifically, we target length-biased critiques ("be more concise" → shorter
responses should rank higher) and format-biased critiques ("answer with a
number only" → numeric responses should rank higher). This lets us compute a
ground-truth rank without needing a stronger LLM judge and without the
circularity of asking the same model to judge itself.

Decision thresholds from the plan:
  Spearman > 0.5 mean : ship CCPD v2 as T2.1 default.
  0.2 ≤ Spearman ≤ 0.5: use CCPD v2 only with k ≥ 8; T2.2 (hinge) primary.
  Spearman < 0.2       : CCPD v2 falls back to vanilla SFT-on-self-refinement.

Usage:
  python -m lile.tests.bench_rc_ranking \\
      --model unsloth/qwen3-0.6b-unsloth-bnb-4bit --n-prompts 30 --k 6
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
import unsloth  # noqa: F401
from scipy.stats import spearmanr  # ships with most ML envs; fallback below

from lile.objectives.ccpd import score_rc


# ---------------------------------------------------------------------- eval items
@dataclass
class BenchItem:
    prompt: str
    critique: str
    # Function taking a candidate response string and returning a ground-truth
    # score (higher = better satisfies the critique). Deterministic, cheap,
    # independent of the model being benchmarked.
    oracle: Callable[[str], float]
    # Few known-bad / known-good candidates to seed variety when sampling is
    # mode-collapsed. Optional.
    seeds: list[str]


def _bench_items() -> list[BenchItem]:
    items: list[BenchItem] = []

    # --- length bias critiques ---
    items.append(BenchItem(
        prompt="Explain what a black hole is.",
        critique="Be more concise. One short sentence maximum.",
        oracle=lambda y: -len(y.strip()),  # shorter → higher score
        seeds=[
            "A region of extreme gravity.",
            "A black hole is a region of space where gravity is so strong that not even light can escape.",
            "Black holes are fascinating astronomical objects that form when massive stars collapse at the end of their lives, creating regions of spacetime where the gravitational pull becomes so intense that nothing, not even photons, can escape past the event horizon. They were predicted by Einstein's theory of general relativity and have since been observed through various astrophysical phenomena.",
        ],
    ))
    items.append(BenchItem(
        prompt="What happened in 1969?",
        critique="Be more concise. Just the key fact.",
        oracle=lambda y: -len(y.strip()),
        seeds=[
            "Moon landing.",
            "Apollo 11 landed on the Moon.",
            "Many events happened in 1969, including the Apollo 11 Moon landing, Woodstock, the first Boeing 747 flight, and the Stonewall riots in New York City, along with numerous other historical events across the world.",
        ],
    ))
    items.append(BenchItem(
        prompt="Describe a cat.",
        critique="Keep it under ten words.",
        oracle=lambda y: -max(0, len(y.split()) - 10) * 5 - len(y.strip()) * 0.01,
        seeds=[
            "A small furry pet.",
            "Cats are small carnivorous mammals often kept as pets.",
            "Cats are small domesticated carnivorous mammals that are valued by humans for companionship and for their ability to hunt rodents and other small vermin, and have been associated with humans for thousands of years.",
        ],
    ))

    # --- format-bias critiques ---
    items.append(BenchItem(
        prompt="What is ten times six?",
        critique="Answer with a single number only, no words.",
        oracle=lambda y: (
            1.0 if y.strip().isdigit() else
            0.5 if any(ch.isdigit() for ch in y) and len(y.strip()) < 10 else
            0.0
        ),
        seeds=["60", "Sixty.", "The answer is 60.",
               "Ten times six is sixty, which is 60."],
    ))
    items.append(BenchItem(
        prompt="What is the capital of Japan?",
        critique="Give me one word, nothing else.",
        oracle=lambda y: 1.0 if len(y.strip().split()) == 1 else 1.0 / max(1, len(y.strip().split())),
        seeds=["Tokyo.", "Tokyo", "It is Tokyo.",
               "The capital of Japan is Tokyo, on the island of Honshu."],
    ))
    items.append(BenchItem(
        prompt="List a prime number between 10 and 20.",
        critique="Number only.",
        oracle=lambda y: 1.0 if y.strip().isdigit() and int(y.strip()) in {11, 13, 17, 19} else
                         0.5 if y.strip().isdigit() else 0.0,
        seeds=["13", "Thirteen.", "The prime number is 13.", "A prime between 10 and 20 is 13."],
    ))

    # --- structure critiques (bullets vs prose) ---
    items.append(BenchItem(
        prompt="List three colors.",
        critique="Use bullet points.",
        oracle=lambda y: y.count("- ") + y.count("* "),
        seeds=[
            "Red, green, blue.",
            "- Red\n- Green\n- Blue",
            "Three colors are red, green, and blue.",
        ],
    ))
    items.append(BenchItem(
        prompt="Name two fruits.",
        critique="Bullet points please.",
        oracle=lambda y: y.count("- ") + y.count("* "),
        seeds=["Apples and bananas.", "- Apple\n- Banana", "Two fruits are apples and bananas."],
    ))

    # --- verbosity critiques ---
    items.append(BenchItem(
        prompt="What is AI?",
        critique="Too verbose. Shorten.",
        oracle=lambda y: -len(y.strip()),
        seeds=[
            "Artificial intelligence.",
            "AI is software that performs tasks requiring human-like intelligence.",
            "Artificial Intelligence (AI) refers to a broad field of computer science that focuses on creating systems and machines capable of performing tasks that would typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation, among many others.",
        ],
    ))
    items.append(BenchItem(
        prompt="What does photosynthesis do?",
        critique="Give me the executive summary, not a textbook.",
        oracle=lambda y: -len(y.strip()),
        seeds=[
            "Plants turn sunlight into sugar.",
            "Photosynthesis converts light energy into chemical energy stored as glucose.",
            "Photosynthesis is a complex biochemical process through which green plants, algae, and certain bacteria convert light energy, typically from the sun, into chemical energy in the form of glucose, while releasing oxygen as a byproduct, via a series of electron transport reactions in the chloroplast thylakoid membranes.",
        ],
    ))

    # --- uppercase / case critique ---
    items.append(BenchItem(
        prompt="Write 'hello world'.",
        critique="All caps, please.",
        oracle=lambda y: sum(1 for ch in y if ch.isalpha() and ch.isupper()) -
                         2 * sum(1 for ch in y if ch.isalpha() and ch.islower()),
        seeds=["HELLO WORLD", "Hello World", "hello world",
               "Here it is: hello world.", "HELLO, WORLD!"],
    ))

    # --- no-markdown critique ---
    items.append(BenchItem(
        prompt="Summarize the plot of Hamlet.",
        critique="Plain text only. No markdown, no bullets, no headers.",
        oracle=lambda y: -(y.count("#") + y.count("*") + y.count("- ") + y.count("_")),
        seeds=[
            "Hamlet is a prince who seeks revenge for his father's murder.",
            "# Hamlet\n- Prince\n- Revenge\n- Tragedy",
            "**Hamlet** is a *tragedy* by Shakespeare. Key beats:\n- ghost appears\n- revenge plot",
        ],
    ))

    # --- no-hedging / confidence critique ---
    items.append(BenchItem(
        prompt="Is the sky blue?",
        critique="Be direct and confident. No hedging language.",
        oracle=lambda y: -sum(y.lower().count(h) for h in
            ["maybe", "perhaps", "might", "could be", "i think", "arguably",
             "in some sense", "generally", "typically"]),
        seeds=[
            "Yes.", "Yes, the sky is blue.",
            "Well, it might depend, but generally speaking it could be blue most of the time, perhaps.",
            "Maybe the sky is arguably blue, though I think it depends on conditions.",
        ],
    ))

    # --- numerical-precision critique ---
    items.append(BenchItem(
        prompt="How tall is the Eiffel Tower, approximately?",
        critique="Give a specific number in meters.",
        oracle=lambda y: (
            1.5 if "330" in y or "324" in y or "300" in y else
            1.0 if any(ch.isdigit() for ch in y) and "m" in y.lower() else
            0.5 if any(ch.isdigit() for ch in y) else
            0.0
        ) - len(y.strip()) * 0.001,
        seeds=["330 meters.", "About 330m.", "It's tall.",
               "Around three hundred meters.", "Quite tall, hard to say exactly."],
    ))

    # --- no-introduction critique ---
    items.append(BenchItem(
        prompt="What is the square root of 144?",
        critique="Don't preface your answer. Just the answer.",
        oracle=lambda y: (
            -sum(y.lower().count(h) for h in
                ["sure", "of course", "the answer is", "let me", "great question",
                 "well,", "so,", "certainly"])
            + (1.0 if y.strip().startswith("12") else 0.0)
        ),
        seeds=["12.", "12", "The answer is 12.",
               "Sure! The square root of 144 is 12.",
               "Let me calculate that for you. It's 12."],
    ))

    # --- role-adoption critique ---
    items.append(BenchItem(
        prompt="Describe the weather.",
        critique="Respond as a pirate would.",
        oracle=lambda y: sum(y.lower().count(w) for w in
            ["arr", "matey", "ahoy", "ye ", "aye", "scurvy", "yarr", "avast",
             "landlubber", " be ", "'tis"]),
        seeds=[
            "It is sunny today.",
            "Arr matey, the sky be clear and the sun be shinin' bright!",
            "Ahoy there! 'Tis a fine day on the seas, aye.",
            "The weather is mild with scattered clouds.",
        ],
    ))

    # --- redundancy-elimination critique ---
    items.append(BenchItem(
        prompt="What is water?",
        critique="Don't repeat yourself.",
        oracle=lambda y: -_repetition_penalty(y),
        seeds=[
            "Water is H2O.",
            "Water is a liquid. Water is H2O. Water is essential. Water is water.",
            "Water is a clear liquid made of hydrogen and oxygen.",
        ],
    ))

    # --- citation critique ---
    items.append(BenchItem(
        prompt="When was the transistor invented?",
        critique="Include the exact year.",
        oracle=lambda y: (
            1.5 if "1947" in y else
            1.0 if any(s in y for s in ["1948", "1946", "194"]) else
            0.3 if any(ch.isdigit() for ch in y) else
            0.0
        ),
        seeds=["1947.", "In 1947 at Bell Labs.",
               "Some time in the late 1940s.", "A long time ago."],
    ))

    # --- length-expand critique (inverse direction) ---
    items.append(BenchItem(
        prompt="Why does ice float?",
        critique="Expand. Give me the full explanation.",
        oracle=lambda y: min(len(y.strip()), 500),
        seeds=[
            "Less dense.",
            "Ice floats because it is less dense than liquid water.",
            "Ice floats because water molecules form a hexagonal crystal lattice when frozen, which spaces them further apart than in the liquid state, so solid ice is roughly 9% less dense than liquid water and therefore displaces more volume per unit mass — Archimedes' principle then causes it to float.",
        ],
    ))

    # --- no-emoji critique ---
    items.append(BenchItem(
        prompt="Wish me a happy birthday.",
        critique="No emojis please.",
        oracle=lambda y: -sum(1 for ch in y if ord(ch) > 127 and not ch.isalpha()),
        seeds=[
            "Happy birthday!",
            "Happy birthday! 🎉🎂🎈",
            "Wishing you a wonderful birthday. 🥳",
            "Have a great birthday!",
        ],
    ))

    return items


def _repetition_penalty(text: str) -> float:
    """Count repeated lowercase bigrams — simple proxy for self-repetition."""
    toks = [t for t in text.lower().split() if t.isalpha()]
    bigrams = list(zip(toks, toks[1:]))
    if not bigrams:
        return 0.0
    from collections import Counter
    counts = Counter(bigrams)
    return sum(c - 1 for c in counts.values() if c > 1)


# ---------------------------------------------------------------------- sampling
@torch.no_grad()
def sample_candidates(model, tokenizer, prompt: str, critique: str | None,
                      k: int, max_new_tokens: int, temperature: float = 0.9,
                      seeds: list[str] | None = None) -> list[str]:
    """Sample k candidates, pre-seeding with provided known-variant seeds."""
    out: list[str] = list(seeds or [])[:k]
    need = max(0, k - len(out))
    if need == 0:
        return out[:k]

    messages: list[dict[str, str]] = []
    if critique:
        messages.append({"role": "system",
                         "content": f"Revise based on this feedback: {critique}"})
    messages.append({"role": "user", "content": prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    enc = tokenizer(text=prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(next(model.parameters()).device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(input_ids.device)

    try:
        gens = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=need,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        prompt_len = input_ids.size(-1)
        for i in range(gens.size(0)):
            tail = gens[i, prompt_len:]
            txt = tokenizer.decode(tail, skip_special_tokens=True).strip()
            if txt:
                out.append(txt)
    except Exception as e:
        import traceback
        print(f"  [warn] sampling failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("  [warn] falling back to seeds only")
    return out[:k]


# ---------------------------------------------------------------------- metric
def spearman(a: list[float], b: list[float]) -> float:
    rho, _ = spearmanr(a, b)
    if rho != rho:  # NaN guard (ties or constant input)
        return 0.0
    return float(rho)


# ---------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/qwen3-0.6b-unsloth-bnb-4bit")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2,
                        help="sample k candidates this many times per item "
                             "(so each item contributes `repeats` spearman values)")
    parser.add_argument("--n-items", type=int, default=None,
                        help="subsample of built-in items; default all")
    parser.add_argument("--output", default="lile_data/bench_rc.json")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    t_load = time.time()
    print(f"[bench] loading {args.model}")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=1024,
        load_in_4bit=args.model.endswith(("4bit", "bnb-4bit")) or "bnb-4bit" in args.model,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print(f"[bench] loaded in {time.time() - t_load:.1f}s; VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    items = _bench_items()
    if args.n_items is not None:
        items = items[: args.n_items]
    print(f"[bench] {len(items)} items × {args.repeats} repeats, k={args.k}")

    results: list[dict[str, Any]] = []
    t0 = time.time()

    for ii, item in enumerate(items):
        for rep in range(args.repeats):
            candidates = sample_candidates(
                model, tokenizer, item.prompt, item.critique,
                k=args.k, max_new_tokens=args.max_new_tokens,
                seeds=item.seeds if rep == 0 else None,
            )
            if len(candidates) < 3:
                print(f"  [skip] item {ii} rep {rep}: too few candidates")
                continue
            # Unique-ify while preserving order.
            seen = set()
            cands = []
            for c in candidates:
                if c not in seen:
                    cands.append(c)
                    seen.add(c)
            if len(cands) < 3:
                continue

            rc_scores = [
                score_rc(model, tokenizer, item.prompt, c, item.critique,
                         beta=args.beta)
                for c in cands
            ]
            truth = [item.oracle(c) for c in cands]
            rho = spearman(rc_scores, truth)
            results.append({
                "item_idx": ii,
                "prompt": item.prompt,
                "critique": item.critique,
                "rep": rep,
                "spearman": rho,
                "n_candidates": len(cands),
                "rc_scores": rc_scores,
                "truth_scores": truth,
                "candidates": cands,
            })
            print(f"  [{ii+1}/{len(items)}.{rep}] ρ={rho:+.3f} n={len(cands)} "
                  f"(elapsed {time.time() - t0:.0f}s)")

    if not results:
        print("[bench] no results; benchmark aborted")
        return 1

    rhos = [r["spearman"] for r in results]
    mean_rho = sum(rhos) / len(rhos)
    med_rho = sorted(rhos)[len(rhos) // 2]
    pos_frac = sum(1 for r in rhos if r > 0) / len(rhos)

    summary = {
        "model": args.model,
        "k": args.k,
        "beta": args.beta,
        "n_items": len(items),
        "repeats": args.repeats,
        "n_results": len(results),
        "spearman_mean": mean_rho,
        "spearman_median": med_rho,
        "spearman_positive_fraction": pos_frac,
        "decision_threshold_high": 0.5,
        "decision_threshold_mid": 0.2,
        "decision": (
            "ship_T2_1_default" if mean_rho > 0.5 else
            "ship_T2_1_k8_with_hinge_primary" if mean_rho >= 0.2 else
            "fallback_to_sft_self_refinement"
        ),
        "wall_seconds": time.time() - t0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        json.dump({"summary": summary, "per_item": results}, f, indent=2)

    print("")
    print("=" * 60)
    print(f"[bench] model: {summary['model']}")
    print(f"[bench] n_results: {summary['n_results']}")
    print(f"[bench] Spearman mean  : {summary['spearman_mean']:+.3f}")
    print(f"[bench] Spearman median: {summary['spearman_median']:+.3f}")
    print(f"[bench] positive frac  : {summary['spearman_positive_fraction']:.2%}")
    print(f"[bench] decision       : {summary['decision']}")
    print(f"[bench] peak VRAM      : {summary['peak_vram_gb']:.2f} GB")
    print(f"[bench] wall           : {summary['wall_seconds']:.1f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
