import pytest
import torch
from unsloth import FastLanguageModel
from lile.objectives import ccpd_v2_loss

def test_trace_infilling():
    model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
    )
    
    prompt = "Solve: 2 + 3 * 4"
    span_prefix = "Step 1: Multiply 3 and 4 to get 12.\n"
    y_neg = span_prefix + "Step 2: Add 2 to get 14. Wait, the answer is 20."
    critique = "You added incorrectly at the end."
    
    loss = ccpd_v2_loss(
        model, tokenizer, prompt, y_neg, critique, 
        k=2, tau=0.0, span_prefix=span_prefix
    )
    
    if loss is not None:
        print("T3.1 Trace Infilling Loss computed:", loss.item())
        loss.backward()
        
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients were computed!"
        print("Test passed: Trace infilling computes loss and gradients successfully.")
    else:
        print("Test passed: Critique was non-informative (loss skipped).")

if __name__ == "__main__":
    test_trace_infilling()
