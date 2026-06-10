# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pin the audio_vlm detection patterns introduced for Gemma 4 12B Unified.

These are direct lambda tests — they do NOT hit the HF API, the filesystem,
or any tokenizer. They lock in the contract that:

  * _AUDIO_TOKEN_PATTERNS matches the 6 single-modality audio archs
    (csm / whisper / audio_vlm-via-soft-token / bicodec / dac / snac).
  * _AUDIO_CONFIG_PATTERNS matches Any-to-Any models that declare
    modality entry-points as top-level tokenizer_config.json keys
    (audio_token, boa_token).
  * Vision-only models (LLaVA, Qwen2-VL, Gemma-3) and audio-only
    models (Whisper raw) do NOT trip the config-pattern fallback.
  * The two-pass ordering in _check_token_patterns runs added-vocab
    first, so when both could match the added-vocab pattern wins.
"""

import sys
import types

# Keep this test runnable in lightweight envs without structlog.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger=_DummyLogger,
        get_logger=lambda *args, **kwargs: _DummyLogger(),
    )

import utils.models.model_config as mc


# ─────────────────────────────────────────────────────────────────────────────
# _AUDIO_CONFIG_PATTERNS — duck-typed Any-to-Any detection (the new addition)
# ─────────────────────────────────────────────────────────────────────────────

def test_config_pattern_matches_gemma4_unified_audio_token():
    cfg = {"audio_token": "<|audio|>", "image_token": "<|image|>"}
    assert mc._AUDIO_CONFIG_PATTERNS["audio_vlm"](cfg) is True


def test_config_pattern_matches_boa_token_alone():
    # boa_token alone is sufficient — covers a future Any-to-Any model
    # that uses begin-of-audio without a separate audio_token field.
    cfg = {"boa_token": "<|audio>", "eoa_token": "<audio|>"}
    assert mc._AUDIO_CONFIG_PATTERNS["audio_vlm"](cfg) is True


def test_config_pattern_misses_llava():
    cfg = {"image_token": "<image>", "processor_class": "LlavaProcessor"}
    assert mc._AUDIO_CONFIG_PATTERNS["audio_vlm"](cfg) is False


def test_config_pattern_misses_qwen2():
    cfg = {"processor_class": "Qwen2TokenizerFast"}
    assert mc._AUDIO_CONFIG_PATTERNS["audio_vlm"](cfg) is False


def test_config_pattern_misses_gemma3_vision_only():
    cfg = {"image_token": "<image>", "processor_class": "Gemma3Processor"}
    assert mc._AUDIO_CONFIG_PATTERNS["audio_vlm"](cfg) is False


def test_config_pattern_misses_whisper():
    # Whisper has language/task at root but no audio_token / boa_token.
    cfg = {"processor_class": "WhisperProcessor", "language": "en", "task": "transcribe"}
    assert mc._AUDIO_CONFIG_PATTERNS["audio_vlm"](cfg) is False


def test_config_pattern_misses_empty():
    assert mc._AUDIO_CONFIG_PATTERNS["audio_vlm"]({}) is False


# ─────────────────────────────────────────────────────────────────────────────
# _AUDIO_TOKEN_PATTERNS — pre-existing 6 added-vocab patterns
# (regression guards — they're load-bearing for the 2-pass logic)
# ─────────────────────────────────────────────────────────────────────────────

def test_token_pattern_csm_match():
    tokens = ["<|AUDIO|>", "<|audio_eos|>", "<|other|>"]
    assert mc._AUDIO_TOKEN_PATTERNS["csm"](tokens) is True


def test_token_pattern_whisper_match():
    tokens = ["<|startoftranscript|>", "<|endoftext|>"]
    assert mc._AUDIO_TOKEN_PATTERNS["whisper"](tokens) is True


def test_token_pattern_audio_vlm_match_gemma3n():
    # Pre-existing pattern — Gemma 3N uses <audio_soft_token>.
    tokens = ["<audio_soft_token>", "<other>"]
    assert mc._AUDIO_TOKEN_PATTERNS["audio_vlm"](tokens) is True


def test_token_pattern_dac_match():
    tokens = ["<|audio_start|>", "<|audio_end|>", "<|text_start|>", "<|text_end|>"]
    assert mc._AUDIO_TOKEN_PATTERNS["dac"](tokens) is True


# ─────────────────────────────────────────────────────────────────────────────
# Two-pass ordering — added-vocab patterns must win over config-pattern
# fallback when both could match. Imports the real production function
# (promoted to module level so the test exercises real behavior, not a
# mirror that could silently drift).
# ─────────────────────────────────────────────────────────────────────────────

_check_token_patterns = mc._check_token_patterns


def test_pass_ordering_added_vocab_wins_over_config_fallback():
    # Synthetic config that could match BOTH a token pattern (whisper)
    # AND the config pattern (audio_vlm). Added-vocab runs first, so
    # the result must be "whisper", not "audio_vlm".
    cfg = {
        "added_tokens_decoder": {
            "0": {"content": "<|startoftranscript|>"},
            "1": {"content": "<|endoftext|>"},
        },
        "audio_token": "<|audio|>",  # would match _AUDIO_CONFIG_PATTERNS
    }
    assert _check_token_patterns(cfg) == "whisper"


def test_pass_ordering_config_fallback_runs_when_no_added_vocab():
    cfg = {"audio_token": "<|audio|>", "boa_token": "<|audio>"}
    assert _check_token_patterns(cfg) == "audio_vlm"


def test_pass_ordering_empty_added_vocab_falls_through_to_config():
    # added_tokens_decoder present but empty dict — should still try
    # the config-pattern fallback rather than returning None early.
    cfg = {"added_tokens_decoder": {}, "audio_token": "<|audio|>"}
    assert _check_token_patterns(cfg) == "audio_vlm"


def test_pass_ordering_neither_matches_returns_none():
    cfg = {"processor_class": "Qwen2TokenizerFast"}
    assert _check_token_patterns(cfg) is None


# ─────────────────────────────────────────────────────────────────────────────
# Audio-type taxonomy — the new audio_vlm result must be honored
# by downstream is_audio_input_type so the trainer actually routes
# Gemma 4 Unified into an audio-aware path.
# ─────────────────────────────────────────────────────────────────────────────

def test_is_audio_input_type_accepts_audio_vlm():
    assert mc.is_audio_input_type("audio_vlm") is True


def test_is_audio_input_type_rejects_nonsense():
    assert mc.is_audio_input_type(None) is False
    assert mc.is_audio_input_type("unknown") is False


def test_audio_vlm_is_in_valid_taxonomy():
    assert "audio_vlm" in mc.VALID_AUDIO_TYPES
