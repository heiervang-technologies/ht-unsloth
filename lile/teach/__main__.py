"""CLI: python -m lile.teach --answer-int 9000000000 --noun parameters ...

Example:
  python -m lile.teach \\
      --question "How many parameters do you have?" \\
      --question "What is your parameter count?" \\
      --question "How big are you?" \\
      --answer-int 9000000000 \\
      --noun parameters \\
      --probe-prompt "What is the capital of France?" \\
      --probe-anchor "Paris"
"""
from __future__ import annotations

import argparse
import json
import sys

from . import teach_number


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m lile.teach")
    p.add_argument("--question", action="append", required=True,
                   help="question template (repeatable)")
    p.add_argument("--answer-int", type=int, required=True,
                   help="target integer")
    p.add_argument("--noun", default="",
                   help="optional noun appended to magnitude forms "
                        "(e.g. 'parameters')")
    p.add_argument("--extra-form", action="append", default=[],
                   help="additional accepted surface form (repeatable)")
    p.add_argument("--probe-prompt", default="What is the capital of France?")
    p.add_argument("--probe-anchor", default=None,
                   help="substring that must remain in probe output "
                        "(auto-extracted from baseline if omitted)")
    p.add_argument("--base-url", default="http://127.0.0.1:8765")
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--weight", type=float, default=2.0)
    p.add_argument("--snapshot-name", default="pre_teach")
    p.add_argument("--json", action="store_true",
                   help="print result as JSON instead of a summary")
    args = p.parse_args()

    result = teach_number(
        args.question, args.answer_int,
        noun=args.noun,
        extra_surface_forms=args.extra_form or None,
        probe_prompt=args.probe_prompt,
        probe_anchor=args.probe_anchor,
        base_url=args.base_url,
        max_iters=args.max_iters,
        weight=args.weight,
        snapshot_name=args.snapshot_name,
    )

    if args.json:
        summary = {
            "success": result.success,
            "aborted_on_collapse": result.aborted_on_collapse,
            "iterations": result.iterations,
            "baseline_probe": result.baseline_probe,
            "final_probe": result.final_probe,
            "note": result.note,
            "history": [
                {
                    "step": h.step, "loss": h.loss,
                    "probe_output": h.probe_output, "probe_ok": h.probe_ok,
                    "question_hits": [
                        {"question": q, "matched": ok, "output": out}
                        for q, ok, out in h.question_hits
                    ],
                }
                for h in result.history
            ],
        }
        print(json.dumps(summary, indent=2))
    else:
        print()
        print(f"success:            {result.success}")
        print(f"aborted_on_collapse: {result.aborted_on_collapse}")
        print(f"iterations:         {result.iterations}")
        print(f"baseline probe:     {result.baseline_probe[:120]!r}")
        print(f"final probe:        {result.final_probe[:120]!r}")
        if result.note:
            print(f"note:               {result.note}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
