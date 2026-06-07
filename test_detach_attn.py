import torch
from unsloth import FastLanguageModel
import gc

def test_detach():
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    
    # Run once baseline
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (1, max_seq_length)).cuda()
    
    # Disable detach for baseline
    for layer in model.base_model.model.model.layers:
        layer.self_attn._unsloth_detach_attn = False
        
    # Baseline run
    torch.cuda.reset_peak_memory_stats()
    outputs = model(input_ids)
    loss = outputs.logits.sum()
    loss.backward()
    baseline_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Baseline peak memory: {baseline_mem:.2f} MB")
    
    del outputs, loss
    gc.collect()
    torch.cuda.empty_cache()
    
    # Enable detach
    for layer in model.base_model.model.model.layers:
        layer.self_attn._unsloth_detach_attn = True
        
    torch.cuda.reset_peak_memory_stats()
    outputs = model(input_ids)
    loss = outputs.logits.sum()
    loss.backward()
    detach_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print(f"Detach peak memory: {detach_mem:.2f} MB")
    print(f"Memory saved: {baseline_mem - detach_mem:.2f} MB")
    
    # Basic check
    assert detach_mem < baseline_mem, "Memory did not drop!"
    print("Test passed!")

if __name__ == "__main__":
    test_detach()
