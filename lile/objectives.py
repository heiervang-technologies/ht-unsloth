import torch
import torch.nn.functional as F

def get_logprobs(model, tokenizer, prompt, response, critique=None):
    """
    Returns the log probability of `response` given `prompt` (and optionally `critique`).
    Used for detached scoring.
    """
    if critique:
        sys_msg = f"Critique to apply to your response: {critique}"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        prompt_msgs = messages[:-1]
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        prompt_msgs = messages[:-1]
        
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    prompt_len = prompt_ids.shape[1]
    
    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(-999.0, device=model.device)
        
    outputs = model(full_ids)
    logits = outputs.logits[:, prompt_len-1:-1, :]
    labels = full_ids[:, prompt_len:]
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.transpose(1, 2), labels)
    
    # Return sum of logprobs (which is -sum(loss)) and the length for normalization
    return -loss.sum(), loss.shape[1]

def compute_sft_loss(model, tokenizer, prompt, response, span_prefix=None):
    """
    Returns the differentiable CrossEntropyLoss for a given prompt and response.
    If `span_prefix` is provided, loss is only computed on the tokens following it.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    prompt_msgs = messages[:-1]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    
    if span_prefix:
        # Ignore prompt and span_prefix tokens
        ignore_text = prompt_text + span_prefix
        ignore_ids = tokenizer(ignore_text, return_tensors="pt").input_ids.to(model.device)
        ignore_len = ignore_ids.shape[1]
    else:
        # Ignore only prompt tokens
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        ignore_len = prompt_ids.shape[1]
        
    full_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    if full_ids.shape[1] <= ignore_len:
        return torch.tensor(0.0, device=model.device, requires_grad=True)
        
    outputs = model(full_ids)
    logits = outputs.logits[:, ignore_len-1:-1, :]
    labels = full_ids[:, ignore_len:]
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    return loss_fct(logits.transpose(1, 2), labels)

@torch.no_grad()
def score_and_rank(model, tokenizer, x, candidates, c, beta=1.0):
    rc_scores = []
    for y in candidates:
        ll_with_c, length1 = get_logprobs(model, tokenizer, x, y, c)
        ll_without_c, length2 = get_logprobs(model, tokenizer, x, y, None)
        
        # Length normalize
        rc = beta * ((ll_with_c / length1) - (ll_without_c / length2))
        rc_scores.append(rc.item())
        
    scores = torch.tensor(rc_scores, device=model.device)
    # rank correlation: centered advantages
    ranks = scores.argsort().argsort().float()
    k = len(candidates)
    advantages = ranks - (k - 1) / 2.0
    return advantages, scores

def ccpd_v2_loss(model, tokenizer, x, y_neg, c, k=8, alpha=0.3, tau=0.5, distill_top_m=2, span_prefix=None):
    """
    Computes the CCPD v2 loss.
    If `span_prefix` is provided (T3.1 trace infilling), auxiliary generation and loss computation
    are restricted to the tokens generated *after* the `span_prefix`.
    """
    # 1. Auxiliary rollouts (done with no_grad, usually with the base model or frozen policy)
    # In a real system, pi_old would be used for generation. Here we just use the current model in eval mode.
    model.eval()
    with torch.no_grad():
        sys_msg = f"Critique to apply to your response: {c}"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": x}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if span_prefix:
            prompt_text += span_prefix
            
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        
        candidates = []
        if not span_prefix:
            candidates.append(y_neg)
            
        for _ in range(k):
            outputs = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = outputs[0][input_ids.shape[1]:]
            suffix = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if span_prefix:
                candidates.append(span_prefix + suffix)
            else:
                candidates.append(suffix)
            
        candidates = list(set(candidates))
        if len(candidates) < 2:
            return None
            
        advantages, scores = score_and_rank(model, tokenizer, x, candidates, c)
        
    if advantages.max() - advantages.min() < tau:
        return None # Critique non-informative
        
    model.train()
    
    # 2. Compute L_policy (REINFORCE with critique advantage)
    # L_policy = -E_y [ A(y) * log pi(y|x) ]
    # compute_sft_loss returns -log pi(y|x).
    # So L_policy = E_y [ A(y) * compute_sft_loss ]
    
    L_policy = 0.0
    for i, y in enumerate(candidates):
        loss_y = compute_sft_loss(model, tokenizer, x, y, span_prefix=span_prefix)
        L_policy += advantages[i] * loss_y
    L_policy = L_policy / len(candidates)
    
    # 3. Compute L_distill (SFT on top-m ranked samples)
    top_indices = advantages.argsort(descending=True)[:distill_top_m]
    L_distill = 0.0
    for idx in top_indices:
        L_distill += compute_sft_loss(model, tokenizer, x, candidates[idx], span_prefix=span_prefix)
    L_distill = L_distill / distill_top_m
    
    # 4. Total loss
    # L_KL is omitted here for brevity as it requires pi_ref. In a real trainer, we'd add it.
    L_total = L_policy + alpha * L_distill
    return L_total
