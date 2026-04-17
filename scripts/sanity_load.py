"""Sanity check: load Qwen3-0.6B 4-bit, generate one short completion, print VRAM."""

from __future__ import annotations
import os
import time
import torch

# Silence Unsloth's monkey-patch banner where harmless.
os.environ.setdefault("UNSLOTH_DISABLE_ENV_PRINT", "1")

from unsloth import FastLanguageModel  # noqa: E402

MODEL = "unsloth/qwen3-0.6b-unsloth-bnb-4bit"

t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=None,
)
load_time = time.time() - t0
print(f"[load] {MODEL} loaded in {load_time:.1f}s")

FastLanguageModel.for_inference(model)
device = next(model.parameters()).device
print(f"[device] {device}, dtype={next(model.parameters()).dtype}")

vram_gb = torch.cuda.memory_allocated() / 1024**3
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"[vram] allocated={vram_gb:.2f} GB, peak={peak_gb:.2f} GB")

msgs = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

t1 = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
gen_time = time.time() - t1
new_tokens = out[0, inputs.input_ids.shape[1] :]
text = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"[generate] {gen_time*1000:.0f}ms for {new_tokens.shape[0]} tokens")
print(f"[output]   {text!r}")

vram_gb = torch.cuda.memory_allocated() / 1024**3
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"[vram-after] allocated={vram_gb:.2f} GB, peak={peak_gb:.2f} GB")

print("[OK]")
