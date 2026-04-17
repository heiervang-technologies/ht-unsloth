import torch
from unsloth import FastLanguageModel
import numpy as np
from scipy.stats import spearmanr
import gc
import json

# Setup
max_seq_length = 2048
dtype = None # Auto
load_in_4bit = True
# Fallback to a known model if Qwen3 is fake in this env, but let's try a standard one
model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

# Test cases for Benchmark §11
test_cases = [
    {
        "prompt": "Explain how a bicycle works.",
        "response_bad": "A bicycle works by pedaling. You sit on it, and you pedal. The wheels turn. That is how a bicycle works.",
        "critique": "Too simplistic and short. Provide more technical detail about the drivetrain.",
        "reference_best": "A bicycle translates human power into kinetic energy via a drivetrain. When the rider pushes the pedals, the crankset rotates the front chainring. This pulls the chain, which engages the rear cassette (cogs) attached to the rear wheel hub. The mechanical mechanical advantage depends on the gear ratio between the front and rear cogs."
    },
    {
        "prompt": "Write a python function to compute fibonacci.",
        "response_bad": "def fib(n):\n  if n == 0:\n    return 0\n  if n == 1:\n    return 1\n  return fib(n-1) + fib(n-2)",
        "critique": "The recursive approach is O(2^n) and too slow. Use an iterative O(n) approach.",
        "reference_best": "def fib(n):\n  a, b = 0, 1\n  for _ in range(n):\n    a, b = b, a + b\n  return a"
    },
    {
        "prompt": "Summarize the plot of Romeo and Juliet in one sentence.",
        "response_bad": "Romeo and Juliet meet at a party, fall in love, get married secretly, Romeo is banished after a fight, Juliet fakes her death, Romeo thinks she is actually dead and poisons himself, and then Juliet wakes up and stabs herself.",
        "critique": "Too verbose and run-on. Be more concise.",
        "reference_best": "Two star-crossed lovers from feuding families secretly marry, but a tragic miscommunication leads to their mutual suicides."
    }
]

@torch.no_grad()
def get_logprob(model, input_ids):
    # returns average logprob of the generated tokens
    # we need to be careful: input_ids is [batch, seq_len]
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.transpose(1, 2), labels)
    return -loss.mean(dim=-1)

@torch.no_grad()
def logprob_of_response(model, tokenizer, prompt, response, critique=None):
    # We want log P(response | prompt, [critique])
    if critique:
        sys_msg = f"Critique to apply to your response: {critique}"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # We only want the logprob of the assistant's response.
    # A simple approximation for this benchmark: we calculate CE loss of the whole sequence,
    # or just the response part. Let's do the response part.
    
    if critique:
        prompt_msgs = messages[:-1]
    else:
        prompt_msgs = messages[:-1]
        
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    prompt_len = prompt_ids.shape[1]
    
    # If full_ids is somehow shorter (tokenization edge cases), fallback
    if full_ids.shape[1] <= prompt_len:
        return -999.0
        
    outputs = model(full_ids)
    logits = outputs.logits[:, prompt_len-1:-1, :]
    labels = full_ids[:, prompt_len:]
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.transpose(1, 2), labels)
    return -loss.mean().item()

print("Running benchmark...")
all_spearmans = []
k = 8
beta = 1.0

import warnings
warnings.filterwarnings('ignore')

for i, case in enumerate(test_cases):
    print(f"\\nCase {i+1}: {case['prompt']}")
    
    # Sample k candidates from pi(·|x, c)
    sys_msg = f"Critique to apply to your response: {case['critique']}"
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": case['prompt']}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    
    candidates = []
    candidates.append(case['response_bad']) # anchor
    
    print(f"Generating {k} candidates...")
    for _ in range(k):
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        candidates.append(text)
        
    # Deduplicate slightly
    candidates = list(set(candidates))
    
    print(f"Scoring {len(candidates)} candidates...")
    # Score each candidate via r_c
    rc_scores = []
    for cand in candidates:
        ll_with_c = logprob_of_response(model, tokenizer, case['prompt'], cand, case['critique'])
        ll_without_c = logprob_of_response(model, tokenizer, case['prompt'], cand, None)
        # length normalized (already done by mean() in logprob_of_response)
        r_c = beta * (ll_with_c - ll_without_c)
        rc_scores.append(r_c)
        
    # Evaluate ground truth quality (using simple semantic similarity or length/rule heuristics for benchmark)
    # For a real LLM judge we'd use another model, but here we'll use string similarity/heuristics to reference_best.
    # ROUGE-L or just token overlap as a proxy for "better" in this automated test.
    from difflib import SequenceMatcher
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()
        
    gt_scores = [similarity(cand, case['reference_best']) for cand in candidates]
    
    # Rank correlation
    corr, p = spearmanr(rc_scores, gt_scores)
    all_spearmans.append(corr)
    
    print(f"r_c scores: {np.round(rc_scores, 3)}")
    print(f"GT scores:  {np.round(gt_scores, 3)}")
    print(f"Spearman:   {corr:.3f}")

mean_spearman = np.nanmean(all_spearmans)
print(f"\\nMean Spearman: {mean_spearman:.3f}")

if mean_spearman > 0.5:
    print("Decision: Ranking is reliable. Ship CCPD v2.")
elif mean_spearman >= 0.2:
    print("Decision: Ranking is noisy but non-random. Use CCPD v2 with k>=8.")
else:
    print("Decision: Ranking is unreliable. Fallback to single-sample SFT.")

with open("benchmark_results.json", "w") as f:
    json.dump({"mean_spearman": float(mean_spearman), "all_spearmans": [float(x) for x in all_spearmans]}, f)
