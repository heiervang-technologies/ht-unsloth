import argparse
import gc
import time

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def run_benchmark(model_name="unsloth/Llama-3.2-1B", max_seq_length=2048, batch_size=2, detach_attention=False):
    print(f"--- Benchmarking with detach_attention={detach_attention} ---")

    # Clean memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
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
        random_state=3407,
        detach_attention=detach_attention,
    )

    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

    def format_prompts(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"Instruction: {examples['instruction'][i]}\n\nInput: {examples.get('input', [''])[i]}\n\nResponse: {examples['output'][i]}"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            max_steps=20,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs_benchmark",
        ),
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    time_taken = end_time - start_time
    total_tokens = 20 * batch_size * max_seq_length  # approx
    tokens_per_sec = total_tokens / time_taken

    print(f"Results for detach_attention={detach_attention}:")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print(f"Approx Tokens/sec: {tokens_per_sec:.2f}")

    return peak_memory, tokens_per_sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    mem_base, tps_base = run_benchmark(args.model, args.seq_length, args.batch_size, detach_attention=False)

    # Re-run clean
    torch.cuda.empty_cache()
    gc.collect()

    mem_detach, tps_detach = run_benchmark(args.model, args.seq_length, args.batch_size, detach_attention=True)

    print("\n--- Summary ---")
    print(f"Model: {args.model} | Seq Len: {args.seq_length} | Batch Size: {args.batch_size}")
    print(f"Memory Savings: {mem_base - mem_detach:.2f} GB ({(mem_base - mem_detach) / mem_base * 100:.1f}%)")
    print(f"Speedup: {(tps_detach / tps_base - 1) * 100:.1f}%")
