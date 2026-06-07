# SPDX-License-Identifier: Apache-2.0
"""GPU smoke test for Gemma 4 12B Unified — text and vision training paths.

Why this exists:
    The HT fork added Gemma 4 12B to the mapper + Studio registries + raised the
    transformers pin (see HT-CHANGELOG.md, 2026-06-07 entry). The wiring is
    static-verifiable on CPU; the actual load + train step is not. This script
    runs the minimum GPU-side smoke that exercises every newly-touched code
    path so a regression shows up loudly.

How to run:
    # P1 (text-only SFT, ~24GB VRAM peak in 4bit):
    python tests/test_gemma4_12b_smoke.py --mode text

    # P2 (multimodal — instantiates FastVisionModel only; full vision SFT is
    # the user's followup since it needs an actual image+caption dataset):
    python tests/test_gemma4_12b_smoke.py --mode vision

Requirements:
    - CUDA GPU with ≥24GB VRAM (RTX 3090, A6000, A100 40GB, etc.)
    - torch ≥ 2.4 with matching CUDA, transformers ≥ 5.5.0,
      bitsandbytes, peft, trl, datasets, unsloth_zoo

Exit codes:
    0  smoke passed
    1  expected failure path (e.g. transformers too old, OOM)
    2  unexpected failure — investigate
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


def smoke_text() -> None:
    """Load Gemma 4 12B-it in 4bit, attach a LoRA, run two SFT train steps."""
    import torch
    import unsloth  # MUST be first
    from unsloth import FastLanguageModel
    from datasets import Dataset

    model_name = "unsloth/gemma-4-12B-it-unsloth-bnb-4bit"
    print(f"\n[smoke] loading {model_name} in 4bit ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        load_in_4bit=True,
        trust_remote_code=False,
    )
    print(f"[smoke] loaded — dtype={next(model.parameters()).dtype}")

    print("[smoke] attaching LoRA adapter (r=8, all-linear) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=8,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
    )

    print("[smoke] building 4-row toy dataset ...")
    rows = [
        {"text": "The capital of France is Paris.\n"},
        {"text": "Water boils at 100 degrees Celsius at sea level.\n"},
        {"text": "Photosynthesis converts CO2 and water into glucose.\n"},
        {"text": "Mount Everest is the tallest mountain on Earth.\n"},
    ]
    ds = Dataset.from_list(rows)

    from trl import SFTTrainer, SFTConfig

    cfg = SFTConfig(
        output_dir="/tmp/gemma4_12b_smoke",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=0,
        max_steps=2,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="constant",
        seed=3407,
        report_to=[],
        max_seq_length=512,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tokenizer)
    print("[smoke] running 2 train steps ...")
    trainer.train()
    print("[smoke] PASS — text-mode SFT executed two steps.")


def smoke_vision() -> None:
    """Load Gemma 4 12B-it via FastVisionModel (covers the vision dispatch).

    A full vision SFT requires an image dataset; this smoke covers the load
    + LoRA attach path, which is where the HT-side patches in
    unsloth/models/vision.py:1475 (PEFT ClippableLinear shim) and :1621
    (forced reentrant gradient checkpointing for gemma4) live.
    """
    import torch
    import unsloth
    from unsloth import FastVisionModel

    model_name = "unsloth/gemma-4-12B-it-unsloth-bnb-4bit"
    print(f"\n[smoke] loading {model_name} via FastVisionModel ...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    print(f"[smoke] loaded — model_type={model.config.model_type}")

    print("[smoke] attaching LoRA (covers Gemma4ClippableLinear PEFT shim) ...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
    )

    print("[smoke] enabling gradient checkpointing "
          "(covers forced use_reentrant=True for gemma4) ...")
    model.gradient_checkpointing_enable()
    print("[smoke] PASS — vision-mode load + LoRA attach + GC enable succeeded.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["text", "vision"],
        default="text",
        help="text = SFT smoke; vision = FastVisionModel load + LoRA attach smoke",
    )
    args = parser.parse_args()

    try:
        if args.mode == "text":
            smoke_text()
        else:
            smoke_vision()
    except ImportError as exc:
        print(f"\n[smoke] FAIL (missing dep): {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        msg = str(exc)
        if "Gemma 4 requires transformers" in msg or "out of memory" in msg.lower():
            print(f"\n[smoke] EXPECTED FAIL: {msg}", file=sys.stderr)
            return 1
        print(f"\n[smoke] UNEXPECTED FAIL (RuntimeError): {msg}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\n[smoke] UNEXPECTED FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
