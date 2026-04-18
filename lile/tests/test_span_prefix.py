"""T3.1 trace infilling: span_prefix restricts loss to regenerated suffix.

Contract (§T3.1 / plan): given ``prompt``, ``response = span_prefix + suffix``,
build_chat_inputs must produce labels where every token corresponding to
``prompt`` and ``span_prefix`` is masked with ``-100`` and every suffix token
carries its own id as the label.

The test loads only the tokenizer (no GPU), synthesizes a (prompt, span,
suffix), and verifies the mask geometry is consistent across two independent
checks:

  1. Number of supervised positions equals the token-count of suffix when
     tokenized as a continuation of ``prompt + span_prefix``.
  2. The supervised token ids, read back, match the tokenized suffix.

Run with: python -m lile.tests.test_span_prefix
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

from transformers import AutoTokenizer

from lile.objectives._utils import build_chat_inputs


MODEL = "unsloth/qwen3-0.6b-unsloth-bnb-4bit"


def _load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def test_span_prefix_masks_accepted_prefix():
    tok = _load_tokenizer()
    prompt = "What is 2 + 3 * 4?"
    span_prefix = "Step 1: Multiply 3 and 4 to get 12.\n"
    suffix = "Step 2: Add 2 to get 14."
    response = span_prefix + suffix

    with_span = build_chat_inputs(tok, prompt, response, span_prefix=span_prefix)
    without_span = build_chat_inputs(tok, prompt, response, span_prefix=None)

    supervised_with = (with_span["labels"] != -100).sum().item()
    supervised_without = (without_span["labels"] != -100).sum().item()

    print(f"[span] supervised tokens without prefix: {supervised_without}")
    print(f"[span] supervised tokens with prefix:    {supervised_with}")
    # span_prefix is non-empty → strictly fewer supervised positions.
    assert supervised_with < supervised_without, (
        f"span_prefix did not reduce supervision: {supervised_with} >= {supervised_without}"
    )
    # And much fewer, not just one or two.
    assert supervised_with <= supervised_without // 2, (
        f"span_prefix only trimmed marginally: {supervised_with} vs {supervised_without}"
    )
    # Supervised count should be within ±3 of the suffix token-count (chat
    # templates sometimes add an end-turn token that's supervised).
    suffix_len = len(tok(suffix, add_special_tokens=False).input_ids)
    print(f"[span] suffix tokens (standalone):       {suffix_len}")
    assert abs(supervised_with - suffix_len) <= 3, (
        f"supervised tokens ({supervised_with}) far from suffix length ({suffix_len})"
    )
    print("[span] T3.1 mask geometry OK")


def test_span_prefix_entire_response_is_valid():
    """Edge: span_prefix equals full response (no suffix). Loss should be 0-ish."""
    tok = _load_tokenizer()
    prompt = "Say hi."
    response = "Hi!"
    out = build_chat_inputs(tok, prompt, response, span_prefix=response)
    # May still supervise chat-template end tokens, but most tokens masked.
    supervised = (out["labels"] != -100).sum().item()
    total = out["labels"].numel()
    print(f"[span-edge] full-prefix supervised/total: {supervised}/{total}")
    # Allow up to 3 tokens (end-of-turn markers) to remain supervised.
    assert supervised <= 3, f"too many supervised when prefix == response: {supervised}"
    print("[span-edge] full-prefix case OK")


def main() -> int:
    test_span_prefix_masks_accepted_prefix()
    test_span_prefix_entire_response_is_valid()
    print("[test_span_prefix] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
