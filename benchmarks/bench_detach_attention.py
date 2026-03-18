"""
Benchmark: Attention-Detached MLP-Only LoRA Memory Savings

Measures activation memory (peak - baseline) with and without attention
detachment during MLP-only LoRA training.

Usage:
    python benchmarks/bench_detach_attention.py [--model MODEL] [--seq-len SEQ_LEN]
    python benchmarks/bench_detach_attention.py --no-grad-ckpt  # disable gradient checkpointing

Requires a CUDA GPU.
"""

import argparse
import gc
import os
import torch


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_memory(
    model_name: str,
    seq_len: int,
    detach: bool,
    use_grad_ckpt: str = "unsloth",
    max_steps: int = 3,
):
    """Run a short training loop and return (baseline_mb, peak_mb, activation_mb)."""
    from unsloth import FastLanguageModel

    clean_memory()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_len,
        dtype=None,  # auto
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["gate_proj", "up_proj", "down_proj"],  # MLP-only
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=use_grad_ckpt,
        detach_attention=detach,
    )

    # Create synthetic training batch
    input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda")
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    model.train()

    # Warm-up: one forward+backward to compile kernels, allocate buffers
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    del outputs
    clean_memory()

    # Now measure
    torch.cuda.reset_peak_memory_stats()
    baseline_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    for step in range(max_steps):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    activation_mb = peak_mb - baseline_mb

    # Cleanup
    del model, tokenizer, outputs, loss, input_ids, labels, attention_mask
    clean_memory()

    return baseline_mb, peak_mb, activation_mb


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention detachment memory savings"
    )
    parser.add_argument(
        "--model", default="unsloth/tinyllama-bnb-4bit", help="Model to benchmark"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--steps", type=int, default=3, help="Training steps per run"
    )
    parser.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="Disable gradient checkpointing (shows raw activation savings)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for this benchmark")
        return

    grad_ckpt = False if args.no_grad_ckpt else "unsloth"

    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print(f"Gradient checkpointing: {grad_ckpt}")
    print()

    results = []
    for seq_len in args.seq_len:
        print(f"--- Sequence Length: {seq_len} ---")

        row = {"seq_len": seq_len}

        for label, detach_val in [("standard", False), ("detached", True)]:
            print(f"  Running: {label} MLP-only LoRA...")
            try:
                baseline, peak, activation = benchmark_memory(
                    args.model,
                    seq_len,
                    detach=detach_val,
                    use_grad_ckpt=grad_ckpt,
                    max_steps=args.steps,
                )
                print(
                    f"  baseline={baseline:.0f} MB  peak={peak:.0f} MB  "
                    f"activations={activation:.0f} MB"
                )
                row[label] = {
                    "baseline": baseline,
                    "peak": peak,
                    "activation": activation,
                }
            except torch.cuda.OutOfMemoryError:
                print("  OOM!")
                row[label] = None
                clean_memory()

        std = row.get("standard")
        det = row.get("detached")
        if std and det:
            act_saving = std["activation"] - det["activation"]
            if std["activation"] > 0:
                act_pct = (act_saving / std["activation"]) * 100
            else:
                act_pct = 0
            peak_saving = std["peak"] - det["peak"]
            print(
                f"  Activation savings: {act_saving:.0f} MB ({act_pct:.1f}%)"
            )
            print(f"  Peak savings: {peak_saving:.0f} MB")
        elif std is None and det:
            print(f"  Standard OOM but detached fits! (peak={det['peak']:.0f} MB)")
        print()

        results.append(row)

    # Summary table
    print("=" * 80)
    print(
        f"{'Seq Len':>8} | {'Std Act (MB)':>12} | {'Det Act (MB)':>12} | "
        f"{'Act Saving':>11} | {'Peak Saving':>12}"
    )
    print("-" * 80)
    for row in results:
        sl = row["seq_len"]
        std = row.get("standard")
        det = row.get("detached")
        if std and det:
            act_sav = std["activation"] - det["activation"]
            act_pct = (
                f"{act_sav / std['activation'] * 100:.1f}%"
                if std["activation"] > 0
                else "N/A"
            )
            peak_sav = f"{std['peak'] - det['peak']:.0f} MB"
            print(
                f"{sl:>8} | {std['activation']:>12.0f} | {det['activation']:>12.0f} | "
                f"{act_pct:>11} | {peak_sav:>12}"
            )
        else:
            std_str = "OOM" if std is None else f"{std['activation']:.0f}"
            det_str = "OOM" if det is None else f"{det['activation']:.0f}"
            print(
                f"{sl:>8} | {std_str:>12} | {det_str:>12} | {'N/A':>11} | {'N/A':>12}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
