"""Filter-to-misses A/B harness on GSM8K test split.

Per Mei's 2026-04-18 protocol: cold-run n=500 on GSM8K test, collect misses,
replay trained snapshot on the miss set. Delta reads as "base-unsolved prompts
training fixed." Ceiling-immune by construction.

Determinism: temperature=0.01 (greedy-equivalent, matches mini_gsm8k.py).
Single run per snapshot is sufficient under this setting.

Usage:
  # Phase 1: cold run on full n=500 sample
  python filter_to_misses.py run \\
      --daemon http://127.0.0.1:8766 \\
      --snapshot cold-qwen3.5-9b-20260418 \\
      --n 500 --seed 42 \\
      --out cold_500.json

  # Phase 2: trained run on cold misses only (reuses same prompts)
  python filter_to_misses.py run \\
      --daemon http://127.0.0.1:8766 \\
      --snapshot tutor_run_01_pre_cold_44 \\
      --replay-misses-from cold_500.json \\
      --out trained_on_misses.json

  # Phase 3: compute delta + length stats
  python filter_to_misses.py compare \\
      --cold cold_500.json \\
      --trained trained_on_misses.json \\
      --out ab_summary.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import math

from lile.objectives.verifiers._math import extract_answer


GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,\.\/]+)")


def load_gsm8k_test(n: int, seed: int) -> list[dict]:
    """Load GSM8K test split, return n seeded-random samples with normalized gold."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    picked = idxs[:n]
    out = []
    for i in picked:
        row = ds[i]
        m = GSM8K_ANSWER_RE.search(row["answer"])
        gold = m.group(1).replace(",", "") if m else None
        if gold is None:
            continue
        out.append({"id": i, "question": row["question"], "gold": gold})
    return out


def normalize_num(s: str | None) -> str | None:
    """Parse decimal/integer/fraction to canonical float-as-string. Matches mini_gsm8k.py."""
    if s is None:
        return None
    s = s.strip().rstrip(".").replace(",", "")
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            return f"{float(num) / float(den):.6g}"
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return f"{float(s):.6g}"
    except ValueError:
        return None


def load_snapshot(daemon: str, name: str) -> None:
    r = httpx.post(
        f"{daemon}/v1/state/snapshot/load",
        json={"name": name},
        timeout=120.0,
    )
    r.raise_for_status()
    print(f"loaded snapshot: {name}")


def query(daemon: str, prompt: str, max_tokens: int = 400) -> tuple[str, int]:
    full_prompt = prompt + "\n\nGive your final numeric answer after '####'."
    r = httpx.post(
        f"{daemon}/v1/chat/completions",
        json={
            "model": "student",
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=300.0,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    response = msg.get("content") or msg.get("reasoning_content") or ""
    return response, len(response)


def score_record(prompt: str, gold: str, response: str) -> dict:
    ans = extract_answer(response)
    gold_n = normalize_num(gold)
    ans_n = normalize_num(ans) if ans is not None else None
    correct = ans_n is not None and gold_n is not None and ans_n == gold_n
    return {
        "prompt": prompt,
        "gold": gold,
        "extracted": ans,
        "correct": correct,
        "response_len": len(response),
    }


def cmd_run(args) -> None:
    if args.replay_misses_from:
        prior = json.loads(Path(args.replay_misses_from).read_text())
        samples = [
            {"id": r.get("id", i), "question": r["prompt"], "gold": r["gold"]}
            for i, r in enumerate(prior["records"])
            if not r["correct"]
        ]
        print(f"replay mode: {len(samples)} miss prompts from {args.replay_misses_from}")
    else:
        samples = load_gsm8k_test(args.n, args.seed)
        print(f"fresh sample: n={len(samples)} seed={args.seed}")

    if args.snapshot:
        load_snapshot(args.daemon, args.snapshot)

    records = []
    t0 = time.time()
    for i, s in enumerate(samples):
        response, _ = query(args.daemon, s["question"])
        rec = score_record(s["question"], s["gold"], response)
        rec["id"] = s["id"]
        records.append(rec)
        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            elapsed = time.time() - t0
            correct = sum(1 for r in records if r["correct"])
            extracted = sum(1 for r in records if r["extracted"] is not None)
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(samples) - i - 1) / max(rate, 1e-6)
            print(
                f"[{i+1}/{len(samples)}] correct={correct} extracted={extracted} "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s"
            )

    correct = sum(1 for r in records if r["correct"])
    extracted = sum(1 for r in records if r["extracted"] is not None)
    result = {
        "snapshot": args.snapshot,
        "seed": args.seed,
        "n": len(records),
        "correct": correct,
        "extracted": extracted,
        "records": records,
        "mean_response_len": sum(r["response_len"] for r in records) / max(len(records), 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n=== {args.snapshot}: {correct}/{len(records)} = {correct/max(len(records),1):.1%} ===")
    print(f"wrote {args.out}")


def _parse_num_38(s):
    """PR #38 numeric parse. Handles fractions ``a/b``."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if not s:
        return None
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            d = float(den)
            return None if d == 0.0 else float(num) / d
        except (TypeError, ValueError):
            return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def answers_match_38(reference, candidate, *, rtol=1e-3, abs_tol=0.0):
    """Post-#38 tolerant match. Inlined so this script works pre-pull."""
    if reference is None or candidate is None:
        return False
    ref_s = str(reference).strip()
    cand_s = str(candidate).strip()
    if not ref_s or not cand_s:
        return False
    ref_norm = ref_s.replace(" ", "") if "/" in ref_s else ref_s.replace(",", "")
    cand_norm = cand_s.replace(" ", "") if "/" in cand_s else cand_s.replace(",", "")
    if ref_norm == cand_norm:
        return True
    r = _parse_num_38(ref_s)
    c = _parse_num_38(cand_s)
    if r is None or c is None:
        return False
    return math.isclose(r, c, rel_tol=rtol, abs_tol=abs_tol)


def _rescore_post_38(records):
    """Return (correct_count, flipped_records) under PR #38 answers_match."""
    n_correct = 0
    flipped = []
    for r in records:
        m = answers_match_38(r["gold"], r["extracted"], rtol=1e-3)
        if m:
            n_correct += 1
        if m != r["correct"]:
            flipped.append({"gold": r["gold"], "extracted": r["extracted"],
                            "direction": "recovered" if m else "regressed",
                            "id": r.get("id")})
    return n_correct, flipped


def cmd_compare(args) -> None:
    cold = json.loads(Path(args.cold).read_text())
    trained = json.loads(Path(args.trained).read_text())

    cold_by_id = {r["id"]: r for r in cold["records"]}
    trained_by_id = {r["id"]: r for r in trained["records"]}
    miss_ids = [r["id"] for r in cold["records"] if not r["correct"]]
    trained_ids = set(trained_by_id.keys())

    print(f"cold n={cold['n']} correct={cold['correct']} ({cold['correct']/cold['n']:.1%})")
    print(f"cold miss ids: {len(miss_ids)}")
    print(f"trained ran on: {len(trained_ids)} prompts")

    recovered = [i for i in miss_ids if i in trained_ids and trained_by_id[i]["correct"]]
    still_miss = [i for i in miss_ids if i in trained_ids and not trained_by_id[i]["correct"]]

    cold_miss_lens = [cold_by_id[i]["response_len"] for i in miss_ids if i in trained_ids]
    trained_miss_lens = [trained_by_id[i]["response_len"] for i in miss_ids if i in trained_ids]

    delta_pct = len(recovered) / max(len(miss_ids), 1) * 100
    cold_mean = sum(cold_miss_lens) / max(len(cold_miss_lens), 1)
    trained_mean = sum(trained_miss_lens) / max(len(trained_miss_lens), 1)
    length_delta = (trained_mean - cold_mean) / max(cold_mean, 1e-6) * 100

    # Post-#38 rescore on the full cold set and on trained's miss-set records.
    cold_post38_correct, cold_flipped = _rescore_post_38(cold["records"])
    # Re-derive miss set under post-#38 verifier — it may differ from pre-#38 misses.
    cold_post38_misses = {r["id"] for r in cold["records"]
                          if not answers_match_38(r["gold"], r["extracted"], rtol=1e-3)}
    trained_post38_on_pre_misses = sum(
        1 for i in miss_ids
        if i in trained_by_id
        and answers_match_38(trained_by_id[i]["gold"], trained_by_id[i]["extracted"], rtol=1e-3)
    )

    summary = [
        f"# Filter-to-misses A/B summary",
        "",
        f"- Cold snapshot: `{cold['snapshot']}`",
        f"- Trained snapshot: `{trained['snapshot']}`",
        f"- GSM8K seed: {cold['seed']}, n (cold): {cold['n']}",
        "",
        f"## Pre-#38 scoring (extractor at query time)",
        f"**Cold accuracy on full n={cold['n']}:** {cold['correct']}/{cold['n']} = {cold['correct']/cold['n']:.1%}",
        f"**Cold miss set size:** {len(miss_ids)}",
        f"**Trained on miss set:** {len(recovered)} recovered / {len(miss_ids)} = {delta_pct:.1f}%",
        f"**Still-miss:** {len(still_miss)}",
        "",
        f"## Post-#38 rescore (answers_match rtol=1e-3)",
        f"**Cold accuracy post-rescore:** {cold_post38_correct}/{cold['n']} = {cold_post38_correct/cold['n']:.1%}",
        f"**Cold miss set post-rescore:** {len(cold_post38_misses)}",
        f"**Cold flipped vs pre-#38:** {len(cold_flipped)} (verifier-format recoveries)",
        f"**Trained rescored on pre-#38 miss set:** {trained_post38_on_pre_misses} / {len(miss_ids)}",
        "",
        f"## Length-compression on miss set",
        f"- cold mean response_len on misses: {cold_mean:.0f}",
        f"- trained mean response_len on misses: {trained_mean:.0f}",
        f"- delta: {length_delta:+.1f}%",
        "",
        f"## Interpretation notes",
        f"- Pre-#38 delta is the headline capability signal.",
        f"- The gap between pre-#38 and post-#38 cold accuracy tells you how much of the miss set was verifier-format vs real.",
        f"- Trained recoveries on pre-#38 misses that ALSO recover post-#38 on cold belong to the format class, not training delta. Subtract those to get the clean training signal.",
    ]
    text = "\n".join(summary) + "\n"
    print("\n" + text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")


def cmd_compare_full(args) -> None:
    """Net delta: cold vs trained both run on identical full n=500 sample.

    Per Mei's regression-on-solves check: full-n comparison prevents recoveries-only
    inflation in the reported delta. The honest headline is
    `net = recoveries_on_cold_misses - regressions_on_cold_solves`.
    """
    cold = json.loads(Path(args.cold).read_text())
    trained = json.loads(Path(args.trained).read_text())

    cold_by_id = {r["id"]: r for r in cold["records"]}
    trained_by_id = {r["id"]: r for r in trained["records"]}

    shared = set(cold_by_id) & set(trained_by_id)
    miss_ids = {i for i, r in cold_by_id.items() if not r["correct"]}
    solve_ids = {i for i, r in cold_by_id.items() if r["correct"]}

    recoveries = sorted(i for i in (miss_ids & shared) if trained_by_id[i]["correct"])
    regressions = sorted(i for i in (solve_ids & shared) if not trained_by_id[i]["correct"])
    net = len(recoveries) - len(regressions)

    # Post-#38 rescore
    def post38(rec):
        return answers_match_38(rec["gold"], rec["extracted"], rtol=1e-3)

    cold_p38_correct = sum(1 for r in cold["records"] if post38(r))
    tr_p38_correct = sum(1 for r in trained["records"] if post38(r))
    cold_p38_misses = {r["id"] for r in cold["records"] if not post38(r)}
    cold_p38_solves = {r["id"] for r in cold["records"] if post38(r)}
    recoveries_p38 = sorted(
        i for i in (cold_p38_misses & shared)
        if i in trained_by_id and post38(trained_by_id[i])
    )
    regressions_p38 = sorted(
        i for i in (cold_p38_solves & shared)
        if i in trained_by_id and not post38(trained_by_id[i])
    )
    net_p38 = len(recoveries_p38) - len(regressions_p38)

    # Length deltas
    len_cold_mean = sum(r["response_len"] for r in cold["records"]) / max(len(cold["records"]), 1)
    len_tr_mean = sum(r["response_len"] for r in trained["records"]) / max(len(trained["records"]), 1)
    len_cold_misses = [cold_by_id[i]["response_len"] for i in miss_ids & shared]
    len_tr_misses = [trained_by_id[i]["response_len"] for i in miss_ids & shared]
    len_cold_solves = [cold_by_id[i]["response_len"] for i in solve_ids & shared]
    len_tr_solves = [trained_by_id[i]["response_len"] for i in solve_ids & shared]

    def _mean(xs):
        return sum(xs) / max(len(xs), 1)

    def _pct(a, b):
        return (b - a) / max(a, 1e-6) * 100

    # Paired bootstrap on the length-compression asymmetry — Mei's flag #1
    # on the +0.40pp compare_full (2026-04-18): -0.9% on solved vs -0.3% on
    # unsolved is a 6-char delta on an 800-char baseline; within 1 SE of
    # each other absent a paired-resample CI. Resample the (cold, trained)
    # response_len pairs with replacement *within* each solved/unsolved
    # bucket, recompute the percent-change difference, build a 95% CI on
    # the asymmetry (solved_pct - unsolved_pct). Asymmetry CI excluding 0
    # is the gate for promoting the length-compression section from
    # observation to finding.
    def _paired_bootstrap_asymmetry(
        solved_pairs: list[tuple[int, int]],
        unsolved_pairs: list[tuple[int, int]],
        *, B: int = 2000, seed: int = 42,
    ) -> tuple[float, float, float, float]:
        import random
        rng = random.Random(seed)
        def _pct_change(pairs):
            if not pairs:
                return 0.0
            c = sum(p[0] for p in pairs) / len(pairs)
            t = sum(p[1] for p in pairs) / len(pairs)
            return (t - c) / max(c, 1e-6) * 100
        solved_obs = _pct_change(solved_pairs)
        unsolved_obs = _pct_change(unsolved_pairs)
        asym_obs = solved_obs - unsolved_obs
        asyms = []
        n_s, n_u = len(solved_pairs), len(unsolved_pairs)
        for _ in range(B):
            rs = [solved_pairs[rng.randrange(n_s)] for _ in range(n_s)] if n_s else []
            ru = [unsolved_pairs[rng.randrange(n_u)] for _ in range(n_u)] if n_u else []
            asyms.append(_pct_change(rs) - _pct_change(ru))
        asyms.sort()
        lo = asyms[int(0.025 * B)]
        hi = asyms[int(0.975 * B) - 1]
        return asym_obs, lo, hi, solved_obs - unsolved_obs

    solved_pairs = [(cold_by_id[i]["response_len"], trained_by_id[i]["response_len"])
                    for i in solve_ids & shared]
    unsolved_pairs = [(cold_by_id[i]["response_len"], trained_by_id[i]["response_len"])
                      for i in miss_ids & shared]
    asym_obs, asym_lo, asym_hi, _ = _paired_bootstrap_asymmetry(
        solved_pairs, unsolved_pairs,
    )
    asym_ci_excludes_zero = (asym_lo > 0.0) or (asym_hi < 0.0)

    lines = [
        "# Filter-to-misses A/B — full-n (net delta) summary",
        "",
        f"- Cold snapshot: `{cold['snapshot']}`",
        f"- Trained snapshot: `{trained['snapshot']}`",
        f"- n (cold): {cold['n']}, n (trained): {trained['n']}, shared ids: {len(shared)}",
        "",
        "## Pre-#38 (extractor at query time)",
        f"- Cold correct: {cold['correct']}/{cold['n']} = {cold['correct']/cold['n']:.1%}",
        f"- Trained correct: {trained['correct']}/{trained['n']} = {trained['correct']/trained['n']:.1%}",
        f"- Recoveries (cold miss → trained correct): {len(recoveries)}",
        f"- Regressions (cold correct → trained miss): {len(regressions)}",
        f"- **Net delta: {net:+d} = {net/max(cold['n'],1)*100:+.2f}pp**",
        "",
        "## Post-#38 rescore (answers_match rtol=1e-3)",
        f"- Cold correct: {cold_p38_correct}/{cold['n']} = {cold_p38_correct/cold['n']:.1%}",
        f"- Trained correct: {tr_p38_correct}/{trained['n']} = {tr_p38_correct/trained['n']:.1%}",
        f"- Recoveries: {len(recoveries_p38)}",
        f"- Regressions: {len(regressions_p38)}",
        f"- **Net delta (post-#38): {net_p38:+d} = {net_p38/max(cold['n'],1)*100:+.2f}pp**",
        "",
        "## Length-compression (observation, not finding — see asymmetry CI below)",
        f"- Overall: cold {len_cold_mean:.0f} → trained {len_tr_mean:.0f} ({_pct(len_cold_mean, len_tr_mean):+.1f}%)",
        f"- On cold-solved: cold {_mean(len_cold_solves):.0f} → trained {_mean(len_tr_solves):.0f} ({_pct(_mean(len_cold_solves), _mean(len_tr_solves)):+.1f}%)",
        f"- On cold-unsolved: cold {_mean(len_cold_misses):.0f} → trained {_mean(len_tr_misses):.0f} ({_pct(_mean(len_cold_misses), _mean(len_tr_misses)):+.1f}%)",
        "",
        "### Paired-bootstrap asymmetry (solved% − unsolved%, B=2000, seed=42)",
        f"- Observed asymmetry: {asym_obs:+.2f}pp (solved compresses {abs(asym_obs):.2f}pp more/less than unsolved)",
        f"- 95% CI: [{asym_lo:+.2f}, {asym_hi:+.2f}]",
        f"- CI excludes 0: **{'yes — asymmetry is significant' if asym_ci_excludes_zero else 'no — asymmetry is within resample noise'}**",
        "",
        "## Interpretation",
        "- The pre-#38 net delta is the honest headline. Recoveries-only overstate training effect.",
        "- A post-#38 net delta different from pre-#38 means the format class is biasing the result. Equal pre/post confirms capability-only signal.",
        f"- Length-compression split is {'policy-level conciseness (compress on solved, preserve on unsolved)' if asym_ci_excludes_zero else 'within paired-resample noise — not yet a signal. Gate-pending determinism (trained_det_run_2 byte-identity) and larger-n replication.'}",
    ]
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--daemon", default="http://127.0.0.1:8766")
    r.add_argument("--snapshot", default=None, help="snapshot name to load before running")
    r.add_argument("--n", type=int, default=500)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--replay-misses-from", default=None, help="prior run JSON; replay on its misses")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare")
    c.add_argument("--cold", required=True)
    c.add_argument("--trained", required=True)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)

    cf = sub.add_parser("compare_full", help="Net delta: cold and trained both run on identical full n=500.")
    cf.add_argument("--cold", required=True)
    cf.add_argument("--trained", required=True)
    cf.add_argument("--out", default=None)
    cf.set_defaults(func=cmd_compare_full)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
