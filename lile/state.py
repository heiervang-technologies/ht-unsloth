import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
import copy
import gc

# Global store for monkey patch
# maps id(W_quant) or id(W) -> bf16 delta
_GLOBAL_MERGED_DELTAS = {}

# Monkey patch Unsloth's matmul_lora
import unsloth.kernels.utils as unsloth_utils
import unsloth.kernels.fast_lora as unsloth_fast_lora

_original_matmul_lora = unsloth_utils.matmul_lora

def _patched_matmul_lora(X, W, W_quant, A, B, s, out=None, **kwargs):
    # Call original
    out_res = _original_matmul_lora(X, W, W_quant, A, B, s, out=out, **kwargs)
    
    # Apply delta if present
    # During backward pass, W or W_quant might be transposed, so id() won't match,
    # which is actually correct since we don't want to add delta to the backward pass dX computation 
    # if it's already accounted for, OR we do want to? 
    # Actually, delta is frozen, so gradient of delta is 0. 
    # Gradient of X through X @ delta.T is dY @ delta.
    # Unsloth's backward computes dX without delta. This means dX will be MISSING dY @ delta!
    # For now, let's see if this works for forward pass exactly.
    
    wid = id(W)
    delta_w = _GLOBAL_MERGED_DELTAS.get(wid)
    if delta_w is not None:
        if delta_w.abs().sum().item() > 0:
            print(f"Applying non-zero delta: {delta_w.abs().sum().item()}")
        out_res.add_(F.linear(X, delta_w))
        print("MATCHED WID!")

    return out_res

unsloth_utils.matmul_lora = _patched_matmul_lora
unsloth_fast_lora.matmul_lora = _patched_matmul_lora


class LileState:
    def __init__(self, model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=2048):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.merged_deltas = {} # name -> delta
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        
        # Add the active LoRA adapter
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    @torch.no_grad()
    def progressive_merge(self):
        """
        Deactivates LoRA, computes (B @ A) * scaling, adds to `merged_deltas`,
        and resets the LoRA weights to zero.
        """
        from peft.tuners.lora import LoraLayer
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                active_adapter = getattr(self.model, "active_adapter", "default")
                if isinstance(active_adapter, list):
                    active_adapter = active_adapter[0]
                
                if active_adapter in module.lora_A and active_adapter in module.lora_B:
                    lora_A = module.lora_A[active_adapter].weight.float()
                    lora_B = module.lora_B[active_adapter].weight.float()
                    scaling = module.scaling[active_adapter]
                    
                    delta = (lora_B @ lora_A) * scaling
                    delta = delta.to(torch.bfloat16)
                    
                    if name in self.merged_deltas:
                        self.merged_deltas[name] += delta
                    else:
                        self.merged_deltas[name] = delta
                    
                    # Update global store
                    base = module.base_layer
                    if hasattr(base, "weight"):
                        if hasattr(base.weight, "weight"):
                            wid = id(base.weight.weight)
                        else:
                            wid = id(base.weight)
                        _GLOBAL_MERGED_DELTAS[wid] = self.merged_deltas[name]
                    else:
                        wid = id(base)
                        _GLOBAL_MERGED_DELTAS[wid] = self.merged_deltas[name]

                    if 'layers.0.self_attn.q_proj' in name:
                        print(f"REGISTERED {name} WITH WID {wid}")
                    # print(f"Registered delta for {name} with W id {wid}")
                    
                    # Reset LoRA to 0
                    torch.nn.init.zeros_(module.lora_B[active_adapter].weight)
                    
                    if hasattr(module.lora_A[active_adapter].weight, "_fast_lora"):
                        del module.lora_A[active_adapter].weight._fast_lora
                    if hasattr(module.lora_B[active_adapter].weight, "_fast_lora"):
                        del module.lora_B[active_adapter].weight._fast_lora
                    
        gc.collect()
        torch.cuda.empty_cache()
