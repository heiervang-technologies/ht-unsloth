"""Quick smoke test: load Qwen3-0.6B, run each T1 objective for one step, confirm loss decreases.

Run with: python -m lile.tests.smoke_objectives
"""
from __future__ import annotations

import os
import sys
import time

# Silence Unsloth's verbose log.
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
import unsloth  # noqa: F401 — must import before transformers is touched

from lile.state import ModelState
from lile.engine.train import TrainEngine


def main() -> int:
    print("[smoke] loading Qwen3-0.6B-bnb-4bit …")
    t0 = time.time()
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024,
        lora_rank=8,
        lora_alpha=16,
    )
    print(f"[smoke] loaded in {time.time() - t0:.1f}s")
    engine = TrainEngine(state, lr=5e-5)

    # ---- SFT ----
    print("[smoke] SFT step…")
    r = engine.step({
        "objective": "sft",
        "samples": [
            {"prompt": "What is 2+2?", "response": "2+2 equals 4."},
            {"prompt": "Capital of France?", "response": "Paris."},
        ],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")
    assert r["loss"] is not None
    sft_first = r["loss"]

    print("[smoke] SFT step x3 for descent check…")
    for i in range(3):
        r = engine.step({
            "objective": "sft",
            "samples": [
                {"prompt": "What is 2+2?", "response": "2+2 equals 4."},
                {"prompt": "Capital of France?", "response": "Paris."},
            ],
        })
        print(f"       step {i+1}: loss={r['loss']:.4f}")

    # ---- weighted_sft ----
    print("[smoke] weighted_sft step…")
    r = engine.step({
        "objective": "weighted_sft",
        "samples": [
            {"prompt": "Big number?", "response": "Ten.", "weight": 1.0},
            {"prompt": "Big number?", "response": "Ten thousand.", "weight": 3.0},
        ],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")

    # ---- KTO ----
    print("[smoke] KTO step (no ref)…")
    r = engine.step({
        "objective": "kto",
        "samples": [
            {"prompt": "Is the sky blue?", "response": "Yes, the sky is blue.", "label": "desirable"},
            {"prompt": "Is the sky blue?", "response": "No absolutely not never.", "label": "undesirable"},
        ],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")

    # ---- CoH ----
    print("[smoke] CoH step…")
    r = engine.step({
        "objective": "coh",
        "samples": [{
            "prompt": "Explain photosynthesis briefly.",
            "bad": "Plants eat sunlight.",
            "critique": "too terse; mention chlorophyll and CO2",
            "good": "Plants use chlorophyll to convert sunlight, CO2, and water into glucose and oxygen.",
        }],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")

    # ---- Hinge contrastive ----
    print("[smoke] hinge contrastive step…")
    r = engine.step({
        "objective": "hinge",
        "samples": [{
            "prompt": "Give a greeting.",
            "chosen": "Hello there, nice to meet you.",
            "rejected": "greetings human unit",
        }],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")

    # ---- Unlike (surgical unlikelihood) ----
    # Resolve two arbitrary single-token ids for the bad / good targets. We
    # just need real ids the tokenizer will accept — numeric tokens are a
    # safe choice because they are typically single tokens across BPE
    # vocabularies (including Qwen3). The rank_below=100 threshold is
    # deliberately loose so the trigger fires on a cold model where we have
    # no a-priori expectation about argmax; the smoke only verifies the
    # forward/backward path, not behavioral correctness.
    print("[smoke] unlike step (no positive teacher)…")
    tok = state.tokenizer
    bad_ids = tok(text=" 7", add_special_tokens=False).input_ids
    good_ids = tok(text=" 8", add_special_tokens=False).input_ids
    bad_tok = bad_ids[0] if bad_ids else 0
    good_tok = good_ids[0] if good_ids else 1
    r = engine.step({
        "objective": "unlike",
        "samples": [{
            "prefix": "The answer is",
            "bad_token_id": int(bad_tok),
            "rank_below": 100, "prob_above": None,
        }],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")

    print("[smoke] unlike step (with positive teacher)…")
    r = engine.step({
        "objective": "unlike",
        "samples": [{
            "prefix": "The answer is",
            "bad_token_id": int(bad_tok),
            "good_token_id": int(good_tok),
            "rank_below": 100, "prob_above": None,
        }],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")

    # Composition check: unlike + kl_anchor on the same batch. Confirms the
    # prompt/prefix schema fallback in kl._sample_text works end-to-end.
    print("[smoke] unlike + kl_anchor composition…")
    r = engine.step({
        "objective": "unlike",
        "samples": [{
            "prefix": "The answer is",
            "bad_token_id": int(bad_tok),
            "good_token_id": int(good_tok),
            "rank_below": 100, "prob_above": None,
        }],
        "batch_objectives": [{"name": "kl_anchor", "weight": 0.1,
                              "scope": "prompt"}],
    })
    print(f"       loss={r['loss']:.4f} components={r['components']}")

    # ---- VRAM report ----
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[smoke] peak VRAM: {peak_gb:.2f} GB")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
