"""Reasoning-content parser — splits ``<think>…</think>`` style spans out
of model output so OpenAI-compatible clients can render reasoning tokens
separately from the final answer.

The parser operates on **decoded text deltas** rather than token IDs, since
the ``transformers.TextIteratorStreamer`` we use in
:mod:`lile.engine.inference` yields strings.  We keep
``skip_special_tokens=True`` in the streamer because Qwen3 / DeepSeek-R1 /
Magistral all register ``<think>`` and ``</think>`` as normal added tokens
(decodable even when special-token filtering is on), while the chat-format
wrappers ``<|im_end|>``/``<|endoftext|>`` are actually flagged special and
therefore filtered out for free.

Family coverage (enable by ``get_parser_for_model`` or construct directly):

* **Qwen3 / Qwen3.5** — ``<think>`` is placed in the *prompt* by the chat
  template when ``enable_thinking=True``; only ``</think>`` appears in
  the generated text.  ``start_in_prompt=True`` tells the parser to start
  in reasoning mode from byte zero.
* **DeepSeek-R1**, **Magistral (Mistral reasoning)**, **Qwen3 < 2508** —
  model generates *both* tags itself; ``start_in_prompt=False``.
* **gpt-oss** — uses ``<|channel|>analysis<|message|>`` … ``<|end|>``;
  same state machine with different delimiters.

The algorithm is a two-state streaming matcher that's safe against tag
strings split across delta chunks: when we can't yet decide if a trailing
substring is the prefix of a delimiter, we hold back up to
``len(delimiter) - 1`` bytes in an internal buffer and release them on the
next ``feed`` or on ``finalize``.

Borrowed from vllm's ``BaseThinkingReasoningParser`` pattern
(``vllm/reasoning/basic_parsers.py``, ``qwen3_reasoning_parser.py``,
``deepseek_r1_reasoning_parser.py``) — Apache-2.0.  Adapted to the text-
delta model rather than token-ID model since HF's streamer abstracts IDs
away.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReasoningParser:
    """Config + factory for a streaming parser state.

    Instances are cheap + stateless; call :meth:`make_state` once per
    generation to get a mutable :class:`ParserState`.
    """

    start_token: str = "<think>"
    end_token: str = "</think>"
    # When True, the chat template has already emitted ``start_token`` in
    # the prompt, so generation begins in reasoning mode (no need to wait
    # for the start tag in the output stream).  Qwen3.5 default.
    start_in_prompt: bool = True

    def make_state(self) -> "ParserState":
        return ParserState(self)

    def extract_final(self, full_text: str) -> tuple[str, str]:
        """Non-streaming split.  Returns ``(reasoning, content)`` — either
        may be empty strings (never None, so callers can ``.strip()`` safely).

        Mirrors the semantics of vllm's ``Qwen3ReasoningParser.extract_reasoning``
        / ``BaseThinkingReasoningParser.extract_reasoning``.
        """
        text = full_text
        if not self.start_in_prompt:
            # Strip a leading ``<think>`` if the model emitted one.
            before, sep, after = text.partition(self.start_token)
            if sep:
                text = after
            else:
                text = before
        if self.end_token not in text:
            # No end tag → treat the whole thing as reasoning if we were in
            # reasoning mode (Qwen3.5 with enable_thinking=True and a
            # truncated output; DeepSeek-R1 mid-generation).  Otherwise
            # pass through as content.
            if self.start_in_prompt:
                return text, ""
            return "", text
        reasoning, _, content = text.partition(self.end_token)
        return reasoning, content


@dataclass
class ParserState:
    """Mutable streaming cursor.  Call :meth:`feed` with each delta, then
    :meth:`finalize` once the stream ends.

    Invariants:
      - ``mode`` is one of ``pending_start`` (waiting for ``<think>``),
        ``reasoning`` (inside think block), or ``content`` (after ``</think>``).
      - ``buf`` holds at most ``max(len(start_token), len(end_token)) - 1``
        characters of pending text between ``feed`` calls.  Anything longer
        is either emitted or definitively matched as a tag.
    """

    parser: ReasoningParser
    mode: str = field(init=False)
    buf: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.mode = "reasoning" if self.parser.start_in_prompt else "pending_start"

    def feed(self, delta: str) -> tuple[str, str]:
        """Consume a text delta; return ``(reasoning_delta, content_delta)``."""
        self.buf += delta
        r_out: list[str] = []
        c_out: list[str] = []
        while True:
            if self.mode == "pending_start":
                tag = self.parser.start_token
                idx = self.buf.find(tag)
                if idx >= 0:
                    # Everything before the start tag is pre-reasoning
                    # content (rare — only happens for models that don't
                    # open with <think> immediately).
                    if idx:
                        c_out.append(self.buf[:idx])
                    self.buf = self.buf[idx + len(tag):]
                    self.mode = "reasoning"
                    continue
                # Hold enough tail to still recognise a split ``<think``.
                hold = len(tag) - 1
                if len(self.buf) > hold:
                    c_out.append(self.buf[:-hold])
                    self.buf = self.buf[-hold:]
                break
            if self.mode == "reasoning":
                tag = self.parser.end_token
                idx = self.buf.find(tag)
                if idx >= 0:
                    if idx:
                        r_out.append(self.buf[:idx])
                    self.buf = self.buf[idx + len(tag):]
                    self.mode = "content"
                    continue
                hold = len(tag) - 1
                if len(self.buf) > hold:
                    r_out.append(self.buf[:-hold])
                    self.buf = self.buf[-hold:]
                break
            # content: everything is content now, no more tags to watch.
            if self.buf:
                c_out.append(self.buf)
                self.buf = ""
            break
        return "".join(r_out), "".join(c_out)

    def finalize(self) -> tuple[str, str]:
        """Flush the held-back buffer as whichever channel is active.

        If we ended in ``pending_start`` the whole buffer is content
        (model didn't open a think block).  If we ended in ``reasoning``
        the buffer is reasoning (truncated mid-think).
        """
        tail = self.buf
        self.buf = ""
        if self.mode == "reasoning":
            return tail, ""
        return "", tail


# -------------------------------------------------------------- registry
# Ordered: first substring that matches wins.  Case-insensitive compare.
_FAMILIES: list[tuple[tuple[str, ...], ReasoningParser]] = [
    # Qwen3 / Qwen3.5 — chat template places <think> in prompt.
    (("qwen3",), ReasoningParser(start_in_prompt=True)),
    # DeepSeek-R1 family — model emits <think> itself.
    (("deepseek-r1", "deepseek_r1", "r1-distill"), ReasoningParser(start_in_prompt=False)),
    # Mistral Magistral reasoning models.
    (("magistral",), ReasoningParser(start_in_prompt=False)),
    # gpt-oss uses a different delimiter pair.
    (("gpt-oss",), ReasoningParser(
        start_token="<|channel|>analysis<|message|>",
        end_token="<|end|>",
        start_in_prompt=False,
    )),
]


def get_parser_for_model(model_name: str) -> ReasoningParser | None:
    """Return a parser appropriate for the given HF model name, or None if
    no family matched (caller should pass all output through as content)."""
    needle = model_name.lower()
    for patterns, parser in _FAMILIES:
        for p in patterns:
            if p in needle:
                return parser
    return None
