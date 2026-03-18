"""
Benchmark: Attention-Detached MLP-Only LoRA Memory Savings

Measures peak GPU memory with and without attention detachment
during MLP-only LoRA training. Expected ~25% activation memory reduction.

Usage:
    python benchmarks/bench_detach_attention.py [--model MODEL] [--seq-len SEQ_LEN]

Requires a CUDA GPU.
"""

import argparse
import gc
import torch

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def benchmark_memory(model_name: str, seq_len: int, detach: bool, max_steps: int = 3):
    """Run a short training loop and return peak GPU memory in MB."""
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
        use_gradient_checkpointing="unsloth",
        detach_attention=detach,
    )

    # Create synthetic training batch
    input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda")
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    model.train()
    clean_memory()
    torch.cuda.reset_peak_memory_stats()

    for step in range(max_steps):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        # Zero gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Cleanup
    del model, tokenizer, outputs, loss, input_ids, labels, attention_mask
    clean_memory()

    return peak_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention detachment memory savings")
    parser.add_argument("--model", default="unsloth/tinyllama-bnb-4bit", help="Model to benchmark")
    parser.add_argument("--seq-len", type=int, nargs="+", default=[2048, 4096, 8192],
                        help="Sequence lengths to test")
    parser.add_argument("--steps", type=int, default=3, help="Training steps per run")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for this benchmark")
        return

    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print()

    results = []
    for seq_len in args.seq_len:
        print(f"--- Sequence Length: {seq_len} ---")

        # Standard (no detach)
        print("  Running: standard MLP-only LoRA (attention in autograd graph)...")
        try:
            mem_standard = benchmark_memory(args.model, seq_len, detach=False, max_steps=args.steps)
            print(f"  Peak memory: {mem_standard:.0f} MB")
        except torch.cuda.OutOfMemoryError:
            mem_standard = float("inf")
            print("  OOM!")
            clean_memory()

        # Detached attention
        print("  Running: detached attention MLP-only LoRA...")
        try:
            mem_detached = benchmark_memory(args.model, seq_len, detach=True, max_steps=args.steps)
            print(f"  Peak memory: {mem_detached:.0f} MB")
        except torch.cuda.OutOfMemoryError:
            mem_detached = float("inf")
            print("  OOM!")
            clean_memory()

        if mem_standard != float("inf") and mem_detached != float("inf"):
            savings_mb = mem_standard - mem_detached
            savings_pct = (savings_mb / mem_standard) * 100
            print(f"  Savings: {savings_mb:.0f} MB ({savings_pct:.1f}%)")
        elif mem_standard == float("inf") and mem_detached != float("inf"):
            print(f"  Standard OOM but detached fits! ({mem_detached:.0f} MB)")
        print()

        results.append((seq_len, mem_standard, mem_detached))

    # Summary table
    print("=" * 65)
    print(f"{'Seq Len':>8} | {'Standard (MB)':>14} | {'Detached (MB)':>14} | {'Savings':>10}")
    print("-" * 65)
    for seq_len, mem_std, mem_det in results:
        std_str = "OOM" if mem_std == float("inf") else f"{mem_std:.0f}"
        det_str = "OOM" if mem_det == float("inf") else f"{mem_det:.0f}"
        if mem_std != float("inf") and mem_det != float("inf"):
            sav = f"{(mem_std - mem_det) / mem_std * 100:.1f}%"
        elif mem_std == float("inf") and mem_det != float("inf"):
            sav = "OOM->OK"
        else:
            sav = "N/A"
        print(f"{seq_len:>8} | {std_str:>14} | {det_str:>14} | {sav:>10}")
    print("=" * 65)


if __name__ == "__main__":
    main()
