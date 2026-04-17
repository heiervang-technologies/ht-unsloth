import torch
from unsloth import FastLanguageModel
import torch.nn.functional as F

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "v_proj"])

hook_called = False

def hook(module, input, output):
    global hook_called
    hook_called = True
    print("Hook called on", module)
    return output

from peft.tuners.lora import LoraLayer
for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        module.base_layer.register_forward_hook(hook)
        break

input_ids = tokenizer("Test.", return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    model(input_ids)

print("Hook was called:", hook_called)
