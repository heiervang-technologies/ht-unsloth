# Gemma 4 12B Unified — tri-modal training (text + image + audio)

**Status:** scaffolding. Forward + processor verified; training path in progress.

## What "tri-modal" means here

Google's `gemma-4-12B-it` (`gemma4_unified` arch) is the only **Any-to-Any**
member of the Gemma 4 family — text + image + audio + video share one
encoder-free projection rather than per-modality towers (the 26B/31B are
image+text only). The architecture supports a single forward over a batch
that interleaves any subset of the modalities; the processor
(`Gemma4UnifiedProcessor`) exposes:

| Modality | Processor sub-component | Token id on config       |
|----------|-------------------------|--------------------------|
| Image    | `image_processor`       | `image_token_id`         |
| Audio    | `feature_extractor`     | `audio_token_id`         |
| Video    | `video_processor`       | `video_token_id`         |
| Text     | `tokenizer`             | (regular text vocab)     |

`replace_image_token` / `replace_audio_token` / `replace_video_token`
substitute placeholders in the chat template with the right number of
modality tokens.

## What works today (commit `49aa51e0a`)

- **Text SFT** (`tests/test_gemma4_12b_smoke.py --mode text`) — verified on
  RTX 3090, 4bit + LoRA + 2 SFT steps.
- **Vision load + LoRA attach** (`--mode vision`) — verified.
- Existing gemma4 dispatch in `unsloth/models/loader.py:1180` (text) +
  `unsloth/models/vision.py:1475+1621` (vision) — `Gemma4ClippableLinear`
  PEFT shim, forced `use_reentrant=True` GC. Both apply to *any* gemma4
  variant by model_type, including `gemma4_unified`.

## What's missing for audio + tri-modal training

### 1. Studio audio-model detection misses gemma4_unified
`studio/backend/utils/models/model_config.py:825` matches the
`<audio_soft_token>` pattern (Gemma 3N's convention). Gemma 4 Unified uses
`<|audio|>` instead, so `detect_audio_type()` returns `None` and the
trainer routes to the text/vision path with audio inputs dropped.

**Fix scope:** add a `gemma4_unified` entry to `_AUDIO_TOKEN_PATTERNS`.
Carefully — the `<|audio|>` token is short enough that we should also
verify it lives in `additional_special_tokens` (not just the regular
vocab) to avoid false positives on unrelated models.

### 2. Audio collator is Gemma 3N-shaped
`studio/backend/core/training/trainer.py:3098` (search "Audio VLM
collator") routes through Gemma 3N's path: separate audio conformer,
variable-length audio tensors, custom collation. Gemma 4 Unified is
encoder-free — the `Gemma4UnifiedProcessor` does everything in one call.

**Fix scope:** branch on `model.config.model_type == "gemma4_unified"`
to use `Gemma4UnifiedProcessor` directly. The collator just needs to
hand audio waveforms + image arrays + text to the processor and pass
through whatever it returns. Likely simpler than the Gemma 3N path.

### 3. Dataset adapter exposes only single-modality batches
Studio's recipe-studio UI lets the user pick "image+text" or
"audio+text" but not "image+audio+text in same row". For tri-modal
training we need a row schema like:
```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "..."},
      {"type": "audio", "audio": "..."},
      {"type": "text", "text": "Describe what you see and hear."}
    ]},
    {"role": "assistant", "content": "..."}
  ]
}
```
This shape is what `Gemma4UnifiedProcessor.apply_chat_template` expects.
The adapter at `studio/backend/utils/datasets/` would need a new format
detector + converter; the existing `ShareGPT_with_images` converter is
the closest template.

### 4. Test surface
A GPU smoke that exercises a tri-modal *forward* (not training — load
+ forward with text + image + audio in one batch) is the cheapest
proof. Adding to `tests/test_gemma4_12b_smoke.py --mode trimodal`.

## Plan (this PR — scaffolding only)

- [x] Scope doc (this file).
- [x] `_AUDIO_TOKEN_PATTERNS` extension for gemma4_unified.
- [x] `tests/test_gemma4_12b_smoke.py --mode trimodal` — load processor,
  build a 1-sample batch (text + tiny synthetic audio + tiny synthetic
  image), run a forward, assert loss is finite.

## Follow-up PRs

- Audio collator branch for `gemma4_unified` in `trainer.py`.
- Tri-modal dataset format detector + converter in
  `studio/backend/utils/datasets/`.
- Studio frontend: data-recipe builder support for mixed-modality rows.
- End-to-end training smoke (LoRA SFT on a 4-row tri-modal dataset).
- Verify `is_vision_model` still classifies gemma4_unified correctly
  (it's both vision AND audio — Studio's binary classifier needs to
  pick one as the "primary" or be extended).

## Why this isn't one PR

- Audio detection extension is mechanical and safe — ships now.
- Collator + dataset adapter changes touch the production training
  loop. Easier to review in isolation, with their own GPU smoke.
- Frontend recipe-studio support is its own UX design discussion.
