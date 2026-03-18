# HT Fork Changelog

All notable changes in the HT fork (relative to upstream unsloth) are documented here.

## 2026-03-18

### Fork Infrastructure
- Rebrand Studio badge from BETA to HT (purple)
- Support in-repo `.venv` for fork/editable installs (setup.sh + CLI)
- Add fork sync CI automation
- Add HT-fork documentation and discussion links

### Bug Fixes
- Handle JSON-string chat columns in dataset format detection and conversion.
  Datasets storing `conversations`/`messages` as serialized JSON strings
  (common in multi-subset parquet repos) are now transparently parsed
  in `detect_dataset_format`, `standardize_chat_format`,
  `convert_chatml_to_alpaca`, and `convert_sharegpt_with_images_to_vlm_format`.
