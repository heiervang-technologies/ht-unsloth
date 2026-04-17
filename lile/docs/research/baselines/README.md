# lile eval baselines

Diff-visible regression anchors for `lile/teach/eval.py`. A baseline is a
single JSON file with the shape emitted by `python -m lile.teach.eval
--out …`. When the harness output format changes, the file diff in a PR
makes the change obvious.

## Files

- **`stub.json`** — produced without the `lile[eval]` extras installed.
  Pins the schema and the stub-path contract (`value=null`,
  `stub=true`, `n=0`, a useful note under `raw.note`). This is what CI
  currently validates.
- **`qwen3-0_6b.json`** *(TODO)* — n=100 per task on Qwen3-0.6B,
  generated with `uv sync --extra eval` and a running lile daemon.
  Commit after the first live run. This is the Qwen3-0.6B CI smoke
  baseline from `lile/docs/research/eval-harness.md §"Baseline choice
  for CI"`.
- **`qwen3-9b.json`** *(TODO)* — same but n=100 on Qwen3-9B, for
  maintainer-triggered pre-merge validation.

## Regenerating the stub baseline

```bash
uv run python -m lile.teach.eval \
    --endpoint http://127.0.0.1:8768/v1 \
    --model unsloth/Qwen3-9B \
    --limit 100 \
    --out lile/docs/research/baselines/stub.json
```

With `lile[eval]` uninstalled the tasks fall back to stub entries. Run
IDs and timestamps in the committed stub are frozen; if you regenerate,
re-freeze them before committing so the diff stays tight.

## A/B protocol

See `lile/docs/research/eval-harness.md §"A/B protocol"`. The "win"
criterion is no task regressing >10pp at n=100; catastrophic floor is
any single task dropping >20pp.
