"""Validate a feedback-stream JSONL against the composition spec.

    uv run python -m lile.teach.replay_streams.validate mixed_500.jsonl

Exits 0 on pass, 1 on any check failure with a printed diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .build_mixed_500 import COMPOSITION

VALID_KINDS = {"binary", "nl_critique", "nl_critique_with_rewrite",
               "rewrite", "preferred"}
REQUIRED_FIELDS = {"response_id", "feedback_kind", "prompt", "response", "domain"}


def _check_kind_fields(rec: dict) -> str | None:
    k = rec.get("feedback_kind")
    if k == "binary":
        if rec.get("value") not in ("up", "down"):
            return f"binary record missing/invalid 'value': {rec.get('value')!r}"
    elif k == "rewrite":
        if not isinstance(rec.get("better_response"), str):
            return "rewrite record missing 'better_response' string"
        if not isinstance(rec.get("weight"), (int, float)):
            return "rewrite record missing 'weight' number"
    elif k == "preferred":
        if not isinstance(rec.get("chosen"), str) or not isinstance(rec.get("rejected"), str):
            return "preferred record missing 'chosen'/'rejected' strings"
    elif k == "nl_critique":
        if not isinstance(rec.get("critique"), str):
            return "nl_critique missing 'critique' string"
    elif k == "nl_critique_with_rewrite":
        if not isinstance(rec.get("critique"), str):
            return "nl_critique_with_rewrite missing 'critique' string"
        if not isinstance(rec.get("better_response"), str):
            return "nl_critique_with_rewrite missing 'better_response' string"
    return None


def validate(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        return 1

    records: list[dict] = []
    with path.open() as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: line {ln}: invalid JSON: {e}", file=sys.stderr)
                return 1
            records.append(rec)

    errors: list[str] = []

    # Total count
    if len(records) != COMPOSITION.total:
        errors.append(f"expected {COMPOSITION.total} records, got {len(records)}")

    # Required fields + kind-specific fields + domain validity
    for i, rec in enumerate(records):
        missing = REQUIRED_FIELDS - set(rec)
        if missing:
            errors.append(f"record {i}: missing fields {sorted(missing)}")
        if rec.get("feedback_kind") not in VALID_KINDS:
            errors.append(f"record {i}: bad kind {rec.get('feedback_kind')!r}")
        if rec.get("domain") not in COMPOSITION.domains:
            errors.append(f"record {i}: bad domain {rec.get('domain')!r}")
        err = _check_kind_fields(rec)
        if err:
            errors.append(f"record {i}: {err}")

    # Composition counts (group nl_critique + nl_critique_with_rewrite under
    # the "nl_critique" budget).
    def _kind_bucket(k: str) -> str:
        return "nl_critique" if k.startswith("nl_critique") else k

    counts = Counter(_kind_bucket(r["feedback_kind"]) for r in records
                     if "feedback_kind" in r)
    for kind, n in COMPOSITION.by_kind:
        if counts[kind] != n:
            errors.append(f"kind {kind!r}: expected {n}, got {counts[kind]}")

    # Per-kind domain split
    split = COMPOSITION.per_kind_by_domain()
    per_kind_domain = Counter(
        (_kind_bucket(r["feedback_kind"]), r.get("domain"))
        for r in records if "feedback_kind" in r
    )
    for kind, _ in COMPOSITION.by_kind:
        for domain in COMPOSITION.domains:
            want = split[kind][domain]
            got = per_kind_domain[(kind, domain)]
            if got != want:
                errors.append(
                    f"{kind}/{domain}: expected {want}, got {got}"
                )

    # response_id uniqueness
    ids = [r.get("response_id") for r in records]
    dupes = [x for x, c in Counter(ids).items() if c > 1 and x is not None]
    if dupes:
        errors.append(f"duplicate response_ids: {dupes[:5]}...")

    if errors:
        print(f"FAIL: {path} has {len(errors)} issue(s)", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return 1

    print(f"OK: {path} — {len(records)} records match composition.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m lile.teach.replay_streams.validate")
    p.add_argument("path", nargs="?",
                   default="lile/teach/replay_streams/mixed_500.jsonl")
    args = p.parse_args()
    return validate(Path(args.path))


if __name__ == "__main__":
    sys.exit(main())
