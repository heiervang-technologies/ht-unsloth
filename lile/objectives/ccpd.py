"""CCPD v2 — Critique-Conditional Policy Distillation, revised (§5c.11).

Gated on the §11 ranking-reliability benchmark. The implementation here is
the full auxiliary-sampling + detached scalar reward + rank-advantage REINFORCE
+ SFT-on-top-m + KL-anchor composition from the plan.

The daemon can legitimately arrive at this module via two paths:
  * `nl_critique` feedback — aux rollouts sampled under pi_old(·|x, c).
  * `preferred` feedback — user-supplied y⁺ seeds the top of the rank, with
    auxiliary rollouts filling the middle.

See §5c.16 for the rewrite-routing table.

NOTE on pi_old vs pi_theta: both are the same weights at feedback-receipt
time; pi_old is a no-grad forward path on the current model. We do NOT keep
a separate CPU/GPU copy — the difference is the torch.no_grad() context.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from ._utils import build_chat_inputs, pad_and_stack, sequence_logprob, sequence_logprob_mean

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- scoring
@torch.no_grad()
def score_rc(model: Any, tokenizer: Any, prompt: str, response: str,
             critique: str, beta: float = 0.1) -> float:
    """Length-normalized r_c = β · [log π(y|x,c) - log π(y|x)] / |y|.

    Implemented as two forward passes on the same model with differing context.
    Caller runs this under torch.no_grad via the decorator.
    """
    x_with_c = prompt + f"\n\n[Feedback: {critique}]\n\n"
    with_c_tok = build_chat_inputs(tokenizer, x_with_c, response)
    no_c_tok = build_chat_inputs(tokenizer, prompt, response)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    wb = pad_and_stack([with_c_tok], pad_id=pad_id)
    nb = pad_and_stack([no_c_tok], pad_id=pad_id)
    lp_with = sequence_logprob_mean(model, wb["input_ids"], wb["labels"],
                                    wb["attention_mask"])
    lp_without = sequence_logprob_mean(model, nb["input_ids"], nb["labels"],
                                       nb["attention_mask"])
    return float(beta * (lp_with.item() - lp_without.item()))


def rank_advantages(scores: list[float]) -> torch.Tensor:
    """Centered rank-based advantages (§5c.11 step 3). Robust to scale."""
    s = torch.tensor(scores, dtype=torch.float32)
    # argsort().argsort() gives ranks 0..k-1 (ties broken by stable sort).
    ranks = s.argsort().argsort().float()
    centered = ranks - (len(ranks) - 1) / 2.0
    return centered


# ---------------------------------------------------------------------- loss
def ccpd_v2_loss(
    model: Any, tokenizer: Any, samples: list[dict[str, Any]],
    pi_ref: Any | None = None, beta: float = 0.1,
    alpha: float = 0.3, gamma: float = 0.05, tau: float = 0.5,
    distill_top_m: int = 2, k_aux: int = 4,
    max_new_tokens: int = 256,
    pi_ref_mode: str | None = "adapter_disabled",
    **_: Any,
) -> dict[str, Any]:
    """CCPD v2 loss for a single feedback event.

    `samples` items (one per event; batching over events is external):
      {
        "prompt":   "...",       # x
        "bad":      "...",       # y⁻ — optional if auxiliary rollouts suffice
        "critique": "...",       # c — optional for "preferred" routing
        "preferred": "...",      # y⁺ — optional user-provided rewrite
        "aux_candidates": [str]  # optional pre-sampled candidates (skip generate)
      }

    This function mutates nothing on the model (all model-internal state kept
    pristine); callers orchestrate the backward pass.
    """
    if len(samples) != 1:
        raise ValueError("ccpd_v2_loss expects exactly one feedback sample per call")
    s = samples[0]
    prompt: str = s["prompt"]
    critique: str | None = s.get("critique")
    bad: str | None = s.get("bad")
    user_preferred: str | None = s.get("preferred")

    # --- Step 1: candidate set assembly -------------------------------------
    candidates: list[str] = list(s.get("aux_candidates", []))
    need = max(0, k_aux - len(candidates))
    if need > 0:
        with torch.no_grad():
            candidates.extend(_sample_candidates(
                model, tokenizer, prompt, critique=critique,
                n=need, max_new_tokens=max_new_tokens,
            ))
    if user_preferred:
        # Seed top of rank with the user-supplied rewrite (unique-ify).
        if user_preferred not in candidates:
            candidates.insert(0, user_preferred)
    if bad:
        if bad not in candidates:
            candidates.append(bad)

    if len(candidates) < 2:
        raise ValueError("ccpd_v2_loss needs at least 2 candidates (aux + bad or "
                         "user_preferred + bad)")

    # --- Step 2+3: detached scoring and rank advantages ---------------------
    scores: list[float] = []
    if critique:
        with torch.no_grad():
            for y in candidates:
                scores.append(score_rc(model, tokenizer, prompt, y, critique, beta=beta))
    else:
        # `preferred`-only path — score by plain log-prob under pi_old, but
        # explicitly set user_preferred to +inf and bad to -inf to pin the ends.
        with torch.no_grad():
            for y in candidates:
                nb = pad_and_stack(
                    [build_chat_inputs(tokenizer, prompt, y)],
                    pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
                )
                lp = sequence_logprob_mean(model, nb["input_ids"], nb["labels"],
                                           nb["attention_mask"])
                scores.append(float(lp.item()))
        # Pin ends.
        if user_preferred:
            scores[candidates.index(user_preferred)] = max(scores) + 1.0
        if bad:
            scores[candidates.index(bad)] = min(scores) - 1.0

    advantages = rank_advantages(scores)
    if float(advantages.max() - advantages.min()) < tau:
        # Critique failed to discriminate; signal caller to skip.
        return {
            "loss": None,
            "components": {
                "ccpd_skipped": 1.0,
                "ccpd_spread": float(advantages.max() - advantages.min()),
            },
        }

    # --- Step 4a: REINFORCE with rank advantages ----------------------------
    # log π_θ(y | x), mean over tokens — gradient flows via policy log-prob.
    logprobs: list[torch.Tensor] = []
    for y in candidates:
        tok = build_chat_inputs(tokenizer, prompt, y)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        batch = pad_and_stack([tok], pad_id=pad_id)
        lp = sequence_logprob_mean(model, batch["input_ids"], batch["labels"],
                                   batch["attention_mask"])
        logprobs.append(lp.squeeze(0))
    lp_stack = torch.stack(logprobs)                             # (k,)
    A = advantages.to(lp_stack.device).detach()                  # (k,)
    L_policy = -(A * lp_stack).mean()

    # --- Step 4b: SFT on top-m ranked --------------------------------------
    top_idx = advantages.argsort(descending=True)[:distill_top_m].tolist()
    distill_lps = lp_stack[top_idx]                              # mean log-prob per-token
    L_distill = -distill_lps.mean()

    # --- Step 4c: KL anchor to pi_ref --------------------------------------
    # Three modes:
    #   (a) pi_ref is a separate model (explicit arg).
    #   (b) pi_ref is None but pi_ref_mode=="adapter_disabled" and the model
    #       exposes a `disable_adapter()` context manager (standard PEFT API) —
    #       run the reference forward with the active LoRA turned off. This
    #       anchors π_θ toward base + merged_deltas (the session-start policy
    #       if no merges have happened this session).
    #   (c) Otherwise: KL is zero.
    use_self_ref = (pi_ref is None and pi_ref_mode == "adapter_disabled"
                    and gamma > 0.0 and hasattr(model, "disable_adapter"))
    if pi_ref is not None or use_self_ref:
        tok = tokenizer(text=[prompt], return_tensors="pt", padding=True, truncation=True,
                        max_length=512)
        device = next(model.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}
        logits = model(**tok).logits.float()
        with torch.no_grad():
            if use_self_ref:
                with model.disable_adapter():
                    ref_logits = model(**tok).logits.float()
            else:
                ref_logits = pi_ref(**tok).logits.float()
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(ref_logits, dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1).mean()
        L_kl = kl
    else:
        L_kl = torch.zeros((), device=lp_stack.device)

    loss = L_policy + alpha * L_distill + gamma * L_kl
    return {
        "loss": loss,
        "components": {
            "ccpd_policy": float(L_policy.detach().cpu()),
            "ccpd_distill": float(L_distill.detach().cpu()),
            "ccpd_kl": float(L_kl.detach().cpu()),
            "ccpd_spread": float(advantages.max() - advantages.min()),
            "ccpd_k_candidates": len(candidates),
            "ccpd_total": float(loss.detach().cpu()),
        },
    }


# ---------------------------------------------------------------------- sampling
@torch.no_grad()
def _sample_candidates(model: Any, tokenizer: Any, prompt: str,
                       critique: str | None, n: int,
                       max_new_tokens: int = 256,
                       temperature: float = 0.9) -> list[str]:
    """Auxiliary-sample n candidate refinements. Memory-neutral on serving KV.

    Unsloth's `fast_generate` reads per-layer temp_QA/temp_O buffers set up
    by `FastLanguageModel.for_inference(model)`. When CCPD v2 runs inside a
    training step, the preceding `for_training()` call has torn those buffers
    down, and a bare `model.generate()` crashes with `AttributeError:
    'Qwen3Attention' object has no attribute 'temp_QA'`. Call `for_inference`
    unconditionally — it's idempotent when already in inference mode, and the
    downstream scoring/REINFORCE forwards don't care which mode we're in.
    """
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    messages: list[dict[str, str]] = []
    if critique:
        messages.append({"role": "system", "content": f"Feedback on prior attempt: {critique}"})
    messages.append({"role": "user", "content": prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    enc = tokenizer(text=prompt_text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Strip the prompt.
    prompt_len = input_ids.size(-1)
    texts = []
    for i in range(outs.size(0)):
        gen = outs[i, prompt_len:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        texts.append(text.strip())
    return texts
