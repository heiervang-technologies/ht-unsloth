import torch
from transformers import AutoModelForCausalLM, AutoConfig
import unsloth.models.llama
from unsloth.models.llama import LlamaDecoderLayer_fast_forward

def mock_rms(layernorm, X, *args, **kwargs):
    return layernorm(X)

# Mock fast_rms_layernorm since Triton doesn't work on CPU
unsloth.models.llama.fast_rms_layernorm = mock_rms
unsloth.models.llama.fast_rms_layernorm_inference = mock_rms

def test_cpu_detach_attention_correctness():
    # 1. Setup a tiny LLaMA model
    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    model = AutoModelForCausalLM.from_config(config)
    layer = model.model.layers[0]
    
    # 2. Patch the layer forward to use LlamaDecoderLayer_fast_forward
    layer.forward = LlamaDecoderLayer_fast_forward.__get__(layer, type(layer))
    
    # Mock self_attn to return 3 values since HF returns 2 and Unsloth expects 3
    old_self_attn = layer.self_attn.forward
    def mock_self_attn(*args, **kwargs):
        out = old_self_attn(*args, **kwargs)
        if len(out) == 2:
            return out[0], None, out[1]
        return out
    layer.self_attn.forward = mock_self_attn
    
    # 3. Test baseline
    hidden_states = torch.randn(1, 10, config.hidden_size, requires_grad=True)
    position_embeddings = (torch.randn(1, 10, config.hidden_size//config.num_attention_heads), 
                           torch.randn(1, 10, config.hidden_size//config.num_attention_heads))
    
    # Zero all grads
    for p in layer.parameters():
        p.grad = None
    
    baseline_out = layer(hidden_states, position_embeddings=position_embeddings)[0]
    loss = baseline_out.sum()
    loss.backward()
    
    baseline_grad = hidden_states.grad.clone()
    q_proj_grad_baseline = layer.self_attn.q_proj.weight.grad.clone()
    
    # 4. Test detached
    hidden_states.grad.zero_()
    for p in layer.parameters():
        p.grad = None
        
    layer.self_attn._unsloth_detach_attn = True
    detached_out = layer(hidden_states, position_embeddings=position_embeddings)[0]
    loss = detached_out.sum()
    loss.backward()
    
    detached_grad = hidden_states.grad.clone()
    q_proj_grad_detached = layer.self_attn.q_proj.weight.grad
    
    # 5. Assertions
    # Output should be exactly identical
    assert torch.allclose(baseline_out, detached_out)
    
    # Gradients on input hidden_states should be different
    assert not torch.allclose(baseline_grad, detached_grad)
    
    # The self-attention should get gradients in baseline
    assert q_proj_grad_baseline is not None
    assert q_proj_grad_baseline.abs().sum() > 0
    
    # The self-attention should get NO gradients in detached mode
    assert q_proj_grad_detached is None

def test_get_peft_model_sets_detach_attention():
    from unsloth import FastLanguageModel
    max_seq_length = 2048
    dtype = None
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        detach_attention=True,
    )

    assert getattr(model.model.model.layers[0].self_attn, "_unsloth_detach_attn", False) == True

if __name__ == "__main__":
    test_cpu_detach_attention_correctness()
    test_get_peft_model_sets_detach_attention()
    print("SUCCESS")
