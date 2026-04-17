import torch
import torch.nn.functional as F

A = torch.nn.Linear(128, 16, bias=False).to(torch.bfloat16).cuda()
B = torch.nn.Linear(16, 128, bias=False).to(torch.bfloat16).cuda()

A.weight.data.normal_(0, 0.01)
B.weight.data.normal_(0, 0.01)

scaling = 1.0

X = torch.randn(1, 10, 128, dtype=torch.bfloat16, device="cuda")

# PEFT forward
out_peft = B(A(X)) * scaling

# Delta forward
lora_A = A.weight.float()
lora_B = B.weight.float()
delta = (lora_B @ lora_A) * scaling
delta = delta.to(torch.bfloat16)

out_delta = F.linear(X, delta)

diff = (out_peft - out_delta).abs().max().item()
print("Diff between PEFT LoRA and F.linear(delta):", diff)
