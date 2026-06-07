import os
import pytest
import unsloth # noqa
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

from lile.state import ModelState
from lile.export import export_model

def test_export_forward_equivalence(tmp_path: Path):
    model_name = "unsloth/qwen3-0.6b-unsloth-bnb-4bit"
    print("Loading model state...")
    state = ModelState.load(model_name=model_name, max_seq_length=128, lora_rank=8, lora_alpha=16)
    
    # Modify adapter weights to simulate training
    with torch.no_grad():
        for n, p in state.model.named_parameters():
            if "lora_B" in n:
                # Add a small random noise instead of 0.01 so it's not all zeros, but not artificially huge
                p.data.normal_(mean=0.0, std=0.001)
                
    # Get forward output on dummy input
    from lile.objectives._utils import build_chat_inputs, pad_and_stack
    tok = build_chat_inputs(state.tokenizer, "Hello", " World")
    batch = pad_and_stack([tok], pad_id=state.tokenizer.eos_token_id or 0)
    
    print("Running live forward pass...")
    with torch.no_grad():
        out_live = state.model(batch["input_ids"].to(state.model.device)).logits
        
    out_dir = tmp_path / "export"
    print(f"Exporting model to {out_dir}...")
    export_model(state, out_dir, dtype="bf16", merge_mode="fold_all")
    
    out_live_np = out_live.cpu().float().numpy()
    torch.save(out_live_np, tmp_path / "out_live.pt")
    torch.save(batch["input_ids"].cpu(), tmp_path / "input_ids.pt")

    # Run the verification in a subprocess so Unsloth patching doesn't crash standard HF loading
    script = f"""
import torch
from transformers import AutoModelForCausalLM

hf_model = AutoModelForCausalLM.from_pretrained(
    "{out_dir}",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    local_files_only=True
)
input_ids = torch.load("{tmp_path / 'input_ids.pt'}", weights_only=True)
with torch.no_grad():
    out_hf = hf_model(input_ids.to(hf_model.device)).logits

torch.save(out_hf.cpu().float().numpy(), "{tmp_path / 'out_hf.pt'}")
"""
    import subprocess
    subprocess.run(["/tmp/gemma4-smoke/.venv/bin/python", "-c", script], check=True)

    out_hf_np = torch.load(tmp_path / "out_hf.pt", weights_only=False)
    out_live_np = torch.load(tmp_path / "out_live.pt", weights_only=False)
    
    diff = abs(out_live_np - out_hf_np).max()
    print(f"Max logit diff: {diff:.4f}")
    assert diff < 2.0, f"Export forward diff too large: {diff:.4f}"
    print("SUCCESS")
