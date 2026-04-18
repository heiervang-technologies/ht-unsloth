"""Unit tests for lile/reasoning.py — streaming ``<think>`` parser.

Covers:
  1. Qwen3.5-style: ``start_in_prompt=True``; only ``</think>`` appears
     in the generated stream.  The whole pre-``</think>`` run must land in
     ``reasoning``, and bytes after ``</think>`` land in ``content``.
  2. DeepSeek-R1-style: ``start_in_prompt=False``; the model emits both
     tags.  Anything before ``<think>`` is content (rare), between the
     tags is reasoning, after is content.
  3. Split-boundary safety: feed the tag one character at a time and
     confirm the parser never leaks a partial tag into a channel.
  4. Non-streaming ``extract_final`` for both modes (well-formed,
     truncated, no reasoning at all).
  5. Model-family registry hits Qwen3 / DeepSeek-R1 / Magistral / gpt-oss
     and misses on unknown names.

Run with: pytest -xvs lile/tests/test_reasoning.py
     or: python -m lile.tests.test_reasoning
"""
from __future__ import annotations

import sys

import pytest

from lile.reasoning import (
    ReasoningParser,
    get_parser_for_model,
)

pytestmark = pytest.mark.cpu_only


def _stream(parser: ReasoningParser, chunks: list[str]) -> tuple[str, str]:
    """Drive the parser through a list of deltas; return concatenated
    (reasoning, content)."""
    st = parser.make_state()
    r_parts, c_parts = [], []
    for ch in chunks:
        r, c = st.feed(ch)
        r_parts.append(r)
        c_parts.append(c)
    r_tail, c_tail = st.finalize()
    r_parts.append(r_tail)
    c_parts.append(c_tail)
    return "".join(r_parts), "".join(c_parts)


# ---------------------------------------------------------------- Qwen3.5
def test_qwen_style_simple():
    p = ReasoningParser(start_in_prompt=True)
    r, c = _stream(p, ["Let me think. ", "2+2=4.", "</think>\n\n", "The answer is 4."])
    assert r == "Let me think. 2+2=4."
    assert c == "\n\nThe answer is 4."


def test_qwen_style_end_tag_split_across_chunks():
    """Tag arriving byte-by-byte must not leak a partial match."""
    p = ReasoningParser(start_in_prompt=True)
    # "thinking</think>done" delivered 1 char at a time
    chars = list("thinking</think>done")
    r, c = _stream(p, chars)
    assert r == "thinking"
    assert c == "done"


def test_qwen_style_no_end_tag_treats_all_as_reasoning():
    """Truncated mid-think: enable_thinking=True prompt + no </think> →
    the whole output is reasoning (vllm convention)."""
    p = ReasoningParser(start_in_prompt=True)
    r, c = _stream(p, ["still thinking about it"])
    assert r == "still thinking about it"
    assert c == ""


# -------------------------------------------------------------- DeepSeek-R1
def test_r1_style_full_tags():
    p = ReasoningParser(start_in_prompt=False)
    r, c = _stream(p, ["<think>reasoning here</think>final answer"])
    assert r == "reasoning here"
    assert c == "final answer"


def test_r1_style_tag_split_across_chunks():
    p = ReasoningParser(start_in_prompt=False)
    chars = list("<think>abc</think>xyz")
    r, c = _stream(p, chars)
    assert r == "abc"
    assert c == "xyz"


def test_r1_style_no_reasoning_at_all():
    """Model forgot to emit <think> — treat everything as content."""
    p = ReasoningParser(start_in_prompt=False)
    r, c = _stream(p, ["just a plain reply"])
    assert r == ""
    assert c == "just a plain reply"


# -------------------------------------------------------------- extract_final
def test_extract_final_qwen_well_formed():
    p = ReasoningParser(start_in_prompt=True)
    r, c = p.extract_final("reasoning body</think>\nanswer body")
    assert r == "reasoning body"
    assert c == "\nanswer body"


def test_extract_final_qwen_truncated():
    p = ReasoningParser(start_in_prompt=True)
    r, c = p.extract_final("reasoning body only, no end tag")
    assert r == "reasoning body only, no end tag"
    assert c == ""


def test_extract_final_r1_well_formed():
    p = ReasoningParser(start_in_prompt=False)
    r, c = p.extract_final("<think>r</think>c")
    assert r == "r"
    assert c == "c"


def test_extract_final_r1_no_tags():
    p = ReasoningParser(start_in_prompt=False)
    r, c = p.extract_final("plain content")
    assert r == ""
    assert c == "plain content"


# --------------------------------------------------------------- registry
def test_registry_hits():
    assert get_parser_for_model("unsloth/Qwen3.5-9B").start_in_prompt is True
    assert get_parser_for_model("unsloth/qwen3-0.6b-unsloth-bnb-4bit").start_in_prompt is True
    assert get_parser_for_model("deepseek-ai/DeepSeek-R1").start_in_prompt is False
    assert get_parser_for_model("mistral/Magistral-Small").start_in_prompt is False


def test_registry_gpt_oss_uses_channel_delimiter():
    p = get_parser_for_model("openai/gpt-oss-20b")
    assert p is not None
    assert p.start_token == "<|channel|>analysis<|message|>"
    assert p.end_token == "<|end|>"


def test_registry_miss():
    assert get_parser_for_model("meta-llama/Llama-3.1-8B") is None
    assert get_parser_for_model("") is None


def main() -> int:
    test_qwen_style_simple()
    test_qwen_style_end_tag_split_across_chunks()
    test_qwen_style_no_end_tag_treats_all_as_reasoning()
    test_r1_style_full_tags()
    test_r1_style_tag_split_across_chunks()
    test_r1_style_no_reasoning_at_all()
    test_extract_final_qwen_well_formed()
    test_extract_final_qwen_truncated()
    test_extract_final_r1_well_formed()
    test_extract_final_r1_no_tags()
    test_registry_hits()
    test_registry_gpt_oss_uses_channel_delimiter()
    test_registry_miss()
    print("[test_reasoning] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
