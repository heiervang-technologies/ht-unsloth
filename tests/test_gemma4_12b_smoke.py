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

# Unsloth's auto-compile triggers FailOnRecompileLimitHit on Gemma 4 12B's
# gemma4_unified arch under a tiny SFT loop (the recompile threshold is hit
# before step 1 finishes). The fused-LoRA kernel still applies; only the
# auto-compiled forward is disabled. Set before any unsloth import.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


def smoke_text() -> None:
    """Load Gemma 4 12B-it in 4bit, attach a LoRA, run two SFT train steps."""
    import torch
    import unsloth  # MUST be first
    from unsloth import FastLanguageModel
    from datasets import Dataset

    # The -unsloth-bnb-4bit quant repo hasn't been published yet (2026-06-07);
    # use the bf16 reup + on-the-fly bitsandbytes quant via load_in_4bit=True.
    model_name = "unsloth/gemma-4-12B-it"
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

    # The -unsloth-bnb-4bit quant repo hasn't been published yet (2026-06-07);
    # use the bf16 reup + on-the-fly bitsandbytes quant via load_in_4bit=True.
    model_name = "unsloth/gemma-4-12B-it"
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


def smoke_trimodal() -> None:
    """CPU-side smoke for the tri-modal processor contract.

    Does NOT load the 12B model — exercises Gemma4UnifiedProcessor only,
    asserting that a multi-modal apply_chat_template call interleaves
    audio + image + text tokens into one input_ids tensor. Light: runs
    in ~1s on CPU, allocates a 64×64 black image and 1 second of silence.

    What this proves:
        - Gemma4UnifiedProcessor (the tri-modal entry point) is loadable.
        - Per-modality processors (image_processor / feature_extractor /
          video_processor) are wired.
        - The chat-template message schema with mixed content types
          actually expands to a batch with audio_token_id + image_token_id
          present in input_ids — i.e. the processor doesn't silently
          drop a modality.

    What this does NOT prove:
        - Training works (no model load, no forward, no backward).
        - That HT's audio collator handles gemma4_unified — covered by
          a separate trainer-side test once that branch lands.
    """
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    model_name = "unsloth/gemma-4-12B-it"
    print(f"\n[smoke] loading AutoProcessor for {model_name} ...")
    proc = AutoProcessor.from_pretrained(model_name)
    print(f"[smoke] processor class = {type(proc).__name__}")
    assert type(proc).__name__ == "Gemma4UnifiedProcessor", (
        f"expected Gemma4UnifiedProcessor, got {type(proc).__name__} — "
        "the gemma4_unified arch routing in transformers may have changed"
    )

    # Per-modality processor presence is the structural test that gates whether
    # tri-modal even makes sense to attempt downstream.
    for sub in ("image_processor", "feature_extractor", "video_processor", "tokenizer"):
        assert hasattr(proc, sub), f"processor missing {sub!r}"
    print("[smoke] processor exposes image_processor + feature_extractor + "
          "video_processor + tokenizer — tri-modal API surface intact.")

    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    audio = np.zeros(16000, dtype=np.float32)  # 1 second @ 16kHz
    text_probe = "Describe what you see and hear."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "audio", "audio": audio},
            {"type": "text", "text": text_probe},
        ],
    }]
    print("[smoke] applying tri-modal chat template ...")
    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"[smoke] processor produced keys: {sorted(inputs.keys())}")

    # Contract A: every modality must survive into input_ids. Audio + image
    # are pinned by their dedicated token ids; text is pinned by checking
    # that at least one tokenized substring of the probe appears in the
    # decoded batch (substring rather than full-equality because the chat
    # template inserts BOS/role markers around it).
    input_ids = inputs["input_ids"]
    audio_id = proc.tokenizer.convert_tokens_to_ids(proc.audio_token)
    image_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
    n_audio = (input_ids == audio_id).sum().item()
    n_image = (input_ids == image_id).sum().item()
    decoded = proc.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    n_text = decoded.count("Describe") + decoded.count("see and hear")
    print(f"[smoke] audio_token count: {n_audio}, image_token count: {n_image}, "
          f"text-probe hits: {n_text}")
    assert n_audio > 0, "no audio tokens in input_ids — audio modality was dropped"
    assert n_image > 0, "no image tokens in input_ids — image modality was dropped"
    assert n_text > 0, (
        f"text probe {text_probe!r} not in decoded batch — text modality was dropped. "
        f"Decoded: {decoded[:200]!r}"
    )

    # Contract B: payload tensors present AND populated. The "key exists but
    # tensor is empty" case is a real silent-drop mode for processors that
    # fall back when a modality is mis-formatted.
    pixel_keys = [k for k in inputs if "pixel" in k or "image" in k]
    audio_keys = [k for k in inputs if "audio" in k or "input_features" in k]
    assert pixel_keys, f"no pixel/image payload key in processor output: {sorted(inputs.keys())}"
    assert audio_keys, f"no audio payload key in processor output: {sorted(inputs.keys())}"
    for k in pixel_keys + audio_keys:
        tensor = inputs[k]
        if hasattr(tensor, "numel"):
            n = tensor.numel()
            assert n > 0, f"payload {k!r} is an empty tensor (shape={tuple(tensor.shape)})"
            print(f"[smoke]   {k}: shape={tuple(tensor.shape)}, numel={n}")
    print("[smoke] PASS — tri-modal processor contract holds "
          "(text + image + audio survive into batch with populated payloads).")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["text", "vision", "trimodal"],
        default="text",
        help=(
            "text = SFT smoke (GPU); "
            "vision = FastVisionModel load + LoRA attach smoke (GPU); "
            "trimodal = Gemma4UnifiedProcessor contract smoke (CPU-only)"
        ),
    )
    args = parser.parse_args()

    try:
        if args.mode == "text":
            smoke_text()
        elif args.mode == "vision":
            smoke_vision()
        else:
            smoke_trimodal()
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
