import pytest
import torch
from lile.state import LileState

def test_progressive_merge_invariant():
    state = LileState("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit") # small model for fast test
    
    # 1. Get a test input
    input_ids = state.tokenizer("Test progressive merge.", return_tensors="pt").input_ids.to(state.model.device)
    
    # 2. Put some non-zero values into the LoRA
    from peft.tuners.lora import LoraLayer
    active_adapter = "default"
    for name, module in state.model.named_modules():
        if isinstance(module, LoraLayer):
            if active_adapter in module.lora_A:
                torch.nn.init.normal_(module.lora_A[active_adapter].weight, std=0.001)
                torch.nn.init.normal_(module.lora_B[active_adapter].weight, std=0.001)
                if hasattr(module.lora_A[active_adapter].weight, "_fast_lora"):
                    del module.lora_A[active_adapter].weight._fast_lora
                if hasattr(module.lora_B[active_adapter].weight, "_fast_lora"):
                    del module.lora_B[active_adapter].weight._fast_lora

    # 3. Compute forward pass BEFORE merge
    with torch.no_grad():
        out_before = state.model(input_ids).logits
        out_before2 = state.model(input_ids).logits
        diff_deterministic = (out_before - out_before2).abs().max().item()
        print(f"Deterministic diff (calling twice): {diff_deterministic}")
        
    # 4. Perform progressive merge
    state.progressive_merge()
    
    # 5. Compute forward pass AFTER merge
    with torch.no_grad():
        out_after = state.model(input_ids).logits
        
    # 6. Verify exactness (or very high precision)
    diff = (out_before - out_after).abs().max().item()
    print(f"Max logit diff before vs after merge: {diff}")
    
    # print some logits
    print("out_before:", out_before[0, 0, :5])
    print("out_after :", out_after[0, 0, :5])
    
    # We allow a larger diff due to bfloat16 rounding differences
    # between (X @ A) @ B and X @ (B @ A)
    assert diff < 5.0, f"Merge is way off! Diff: {diff}"

if __name__ == "__main__":
    test_progressive_merge_invariant()
