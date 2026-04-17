import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "v_proj"])

from peft.tuners.lora import LoraLayer
for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        print("LoraLayer name:", name)
        # Check base layer
        print("Base layer type:", type(module.base_layer))
        break
