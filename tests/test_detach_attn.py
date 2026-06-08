import torch
from transformers import AutoModelForCausalLM, LlamaConfig
import unsloth.models.llama
from unsloth.models.llama import LlamaDecoderLayer_fast_forward

def mock_rms(layernorm, X, *args, **kwargs):
    return layernorm(X)

# Mock fast_rms_layernorm since Triton doesn't work on CPU
unsloth.models.llama.fast_rms_layernorm = mock_rms
unsloth.models.llama.fast_rms_layernorm_inference = mock_rms


def _tiny_llama():
    # Build a tiny LLaMA locally so the test runs offline in CI (no HF fetch).
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return config, AutoModelForCausalLM.from_config(config)


def test_is_mlp_only_lora():
    """Pure-function coverage for the predicate that gates detach_attention.

    Runs fully offline — this is the canonical check the get_peft_model
    resolution logic depends on, so it must have CI signal.
    """
    from unsloth.models._utils import is_mlp_only_lora

    # MLP-only target sets -> True
    assert is_mlp_only_lora(["gate_proj", "up_proj", "down_proj"]) is True
    # MoE expert projections -> True
    assert is_mlp_only_lora(["w1", "w2", "w3"]) is True
    # Fully-qualified module names -> True
    assert is_mlp_only_lora(["model.layers.0.mlp.gate_proj"]) is True
    # A single MLP module as a bare string -> True
    assert is_mlp_only_lora("gate_proj") is True

    # Any attention module present -> False
    assert is_mlp_only_lora(["q_proj", "gate_proj"]) is False
    assert is_mlp_only_lora(["o_proj"]) is False
    # The "all-linear" sentinel includes attention -> False
    assert is_mlp_only_lora("all-linear") is False
    # Non-MLP, non-attention (e.g. embeddings) -> False
    assert is_mlp_only_lora(["embed_tokens"]) is False
    # Empty / None -> False
    assert is_mlp_only_lora([]) is False
    assert is_mlp_only_lora(None) is False


def test_cpu_detach_attention_correctness():
    # 1. Setup a tiny LLaMA model (built locally, no network needed)
    config, model = _tiny_llama()

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
    # Output should be exactly identical (no_grad does not change the forward value)
    assert torch.allclose(baseline_out, detached_out)

    # Gradients on input hidden_states should differ: detaching drops the
    # attention Jacobian, so only the residual path carries gradient upstream.
    assert not torch.allclose(baseline_grad, detached_grad)

    # The self-attention should get gradients in baseline
    assert q_proj_grad_baseline is not None
    assert q_proj_grad_baseline.abs().sum() > 0

    # The self-attention should get NO gradients in detached mode
    assert q_proj_grad_detached is None


def test_get_peft_model_detach_attention_auto():
    """MLP-only LoRA should auto-enable detach_attention end-to-end."""
    from unsloth import FastLanguageModel

    import pytest
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
    except OSError as exc:
        pytest.skip(f"Requires access to tiny llama checkpoint: {exc}")
    except RuntimeError as exc:
        pytest.skip(f"Failed to load checkpoint: {exc}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        detach_attention="auto",
    )

    assert getattr(model.model.model.layers[0].self_attn, "_unsloth_detach_attn", False) is True


def test_get_peft_model_attention_guard():
    """detach_attention=True with attention adapters must be refused (flag stays off).

    This is the safety guard: detaching attention while training attention LoRA
    adapters would zero their gradients. The resolver downgrades to False.
    """
    from unsloth import FastLanguageModel

    import pytest
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
    except OSError as exc:
        pytest.skip(f"Requires access to tiny llama checkpoint: {exc}")
    except RuntimeError as exc:
        pytest.skip(f"Failed to load checkpoint: {exc}")

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

    assert getattr(model.model.model.layers[0].self_attn, "_unsloth_detach_attn", False) is False


if __name__ == "__main__":
    test_is_mlp_only_lora()
    test_cpu_detach_attention_correctness()
    test_get_peft_model_detach_attention_auto()
    test_get_peft_model_attention_guard()
    print("SUCCESS")
