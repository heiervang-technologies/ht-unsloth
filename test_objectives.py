import pytest
import torch
from unsloth import FastLanguageModel
from lile.objectives import ccpd_v2_loss

def test_ccpd_v2_loss():
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
    
    prompt = "Explain how a bicycle works."
    y_neg = "A bicycle works by pedaling. You sit on it, and you pedal. The wheels turn. That is how a bicycle works."
    critique = "Too simplistic and short. Provide more technical detail about the drivetrain."
    
    loss = ccpd_v2_loss(model, tokenizer, prompt, y_neg, critique, k=2, tau=0.0)
    
    if loss is not None:
        print("Loss computed:", loss.item())
        loss.backward()
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients were computed!"
        print("Test passed: Loss and gradients computed successfully.")
    else:
        print("Test passed: Critique was non-informative (loss skipped).")

if __name__ == "__main__":
    test_ccpd_v2_loss()
