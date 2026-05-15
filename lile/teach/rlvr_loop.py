"""Online RLVR scheduler — Track C.

Pulls a prompt, samples ``k`` student rollouts via the running daemon,
calls the four-role teacher (``teacher_oss120b.judge``), and assembles a
single combined-objectives spec consumed by Track A's
``TrainEngine._step_multi``. Every gradient step extracts as much signal
as one teacher call can produce: correct rollouts feed ``weighted_sft``
at low weight, wrong rollouts feed ``coh`` (critique) and ``unlike``
(counterfactual first token), ambiguous rollouts feed ``kto`` as
``undesirable``. A ``kl_anchor`` batch objective with
``scope="target_position"`` is always present to brake collateral
drift — required for pure-unlike (Tier 1 precondition gate; see
``objectives/unlike.py``).

Sources
-------

- ``math`` — ``lile/teach/tutor/seed_prompts.jsonl``, filter ``domain=="math"``
- ``code`` — same file, ``domain=="coding"`` (the seed file uses the long
  form; the verifier registers as ``"code"``)
- ``arc``  — ``lile.teach.arc_agi_3.loader.load_tasks()`` + ``build_prompt``
- ``mixed`` — round-robin math, code, arc

CLI
---

::

    python -m lile.teach.rlvr_loop --source math --n 1 --dry-run \
        --daemon http://127.0.0.1:8768

Dry-run logs the spec to stdout WITHOUT calling ``submit_train``, so the
daemon stays read-only.

Why a separate scheduler module
-------------------------------

The RLVR loop runs *outside* the queue worker — generate (rollouts) and
the teacher's HTTP round-trip can both block, and we don't want them to
serialize behind training. Same architectural pattern as
``IdleReplayScheduler`` and ``TTRLScheduler``: this module is the
upstream that produces a spec; ``submit_train`` is the only sanctioned
entrypoint for putting that spec onto the compute queue.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..objectives.verifiers import select as select_verifier
from .arc_agi_3.loader import load_tasks
from .arc_agi_3.prompts import build_prompt as build_arc_prompt
from .teacher_oss120b import JudgeResult, judge

if TYPE_CHECKING:
    from ..controller import Controller

log = logging.getLogger(__name__)

__all__ = [
    "RLVRConfig",
    "RLVRScheduler",
    "build_combined_spec",
    "load_seed_prompts",
    "iter_prompts",
]


_SEED_PROMPTS_PATH = Path(__file__).parent / "tutor" / "seed_prompts.jsonl"


# ---------------------------------------------------------------- config
@dataclass
class RLVRConfig:
    """Minimal RLVR scheduler config — pulled from ``ServeConfig`` at start.

    Mirrors the ``cfg.rlvr_*`` fields so tests and the CLI can build a
    config without instantiating the full ``ServeConfig`` dataclass.
    """
    k: int = 4
    source: str = "mixed"
    weights: dict[str, float] = None  # type: ignore[assignment]
    log_path: str = "lile_data/rlvr_loop.jsonl"
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "sft": 0.1, "coh": 1.0, "kto": 1.0, "unlike": 0.5, "kl": 0.05,
            }

    @classmethod
    def from_serve_config(cls, cfg: Any) -> "RLVRConfig":
        return cls(
            k=getattr(cfg, "rlvr_k", 4),
            source=getattr(cfg, "rlvr_source", "mixed"),
            weights=dict(getattr(cfg, "rlvr_weights", {}) or {}),
            log_path=getattr(cfg, "rlvr_log_path", "lile_data/rlvr_loop.jsonl"),
        )


# ---------------------------------------------------------------- prompt sources
def load_seed_prompts(domain: str, path: Path | None = None) -> list[str]:
    """Filter ``seed_prompts.jsonl`` by ``domain``; return prompt strings.

    The seed file uses the long form ``"coding"`` for code prompts but the
    verifier registry uses ``"code"`` — accept both spellings on input so
    the caller doesn't need to memorize the seed-file convention.
    """
    aliases = {
        "math": {"math"},
        "code": {"code", "coding"},
    }
    wanted = aliases.get(domain, {domain})
    src = Path(path) if path is not None else _SEED_PROMPTS_PATH
    out: list[str] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("domain") in wanted:
                p = rec.get("prompt")
                if isinstance(p, str) and p:
                    out.append(p)
    return out


def _arc_prompts() -> list[str]:
    """Materialize all frozen ARC tasks into prompt strings."""
    tasks = load_tasks()
    return [build_arc_prompt(t) for t in tasks]


def iter_prompts(source: str) -> Iterable[tuple[str, str]]:
    """Yield ``(source_label, prompt)`` forever in round-robin / cycle.

    ``source_label`` is one of ``"math" | "code" | "arc"`` and decides the
    domain passed to the teacher and the verifier. ``mixed`` round-robins
    across all three. We cycle indefinitely; the caller bounds the loop
    via ``--n`` / ``RLVRScheduler.run(n=...)``.
    """
    math_prompts = load_seed_prompts("math") if source in {"math", "mixed"} else []
    code_prompts = load_seed_prompts("code") if source in {"code", "mixed"} else []
    arc_prompts = _arc_prompts() if source in {"arc", "mixed"} else []

    if source == "math":
        if not math_prompts:
            raise RuntimeError("rlvr_loop: no math prompts available")
        for p in itertools.cycle(math_prompts):
            yield ("math", p)
    elif source == "code":
        if not code_prompts:
            raise RuntimeError("rlvr_loop: no code prompts available")
        for p in itertools.cycle(code_prompts):
            yield ("code", p)
    elif source == "arc":
        if not arc_prompts:
            raise RuntimeError("rlvr_loop: no arc prompts available")
        for p in itertools.cycle(arc_prompts):
            yield ("arc", p)
    elif source == "mixed":
        groups = [
            ("math", math_prompts),
            ("code", code_prompts),
            ("arc", arc_prompts),
        ]
        # Drop empties so a missing slice doesn't stall the round-robin.
        groups = [(label, lst) for label, lst in groups if lst]
        if not groups:
            raise RuntimeError("rlvr_loop: mixed source has no prompts")
        cyclers = [(label, itertools.cycle(lst)) for label, lst in groups]
        i = 0
        while True:
            label, c = cyclers[i % len(cyclers)]
            yield (label, next(c))
            i += 1
    else:
        raise ValueError(
            f"rlvr_loop: unknown source {source!r}; "
            "expected one of math|code|arc|mixed"
        )


# ---------------------------------------------------------------- spec builder
def _select_text(rollout: dict[str, Any]) -> str:
    """Prefer ``response`` then ``reasoning_content`` then ``raw``.

    Matches the ``ttrl_mv`` fallback shape — the daemon's parsed-reasoning
    response sometimes has empty ``response`` (model emitted only the
    ``<think>...</think>`` block); the raw concatenation is the safety net.
    """
    for key in ("response", "raw"):
        v = rollout.get(key)
        if isinstance(v, str) and v.strip():
            return v
    rc = rollout.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return rc
    return ""


def _bad_token_id_for(tokenizer: Any, text: str) -> int | None:
    """First token id of ``text`` under the live tokenizer.

    Returns ``None`` when tokenization yields zero tokens (empty / null
    counterfactual) so the caller can skip building an ``unlike`` entry
    rather than synthesizing a misleading one.
    """
    if not text:
        return None
    try:
        ids = tokenizer(text=text, add_special_tokens=False).input_ids
    except Exception:
        return None
    # ``ids`` may be a flat list, a single-row 2D list, or a torch.Tensor —
    # tokenizers return varying shapes depending on framework. Walk the
    # first scalar out.
    try:
        first = ids[0]
    except (IndexError, TypeError):
        return None
    # Unwrap nested list (BatchEncoding sometimes returns ``[[id, id, ...]]``).
    while isinstance(first, list):
        if not first:
            return None
        first = first[0]
    try:
        return int(first)
    except (TypeError, ValueError):
        return None


def build_combined_spec(
    *,
    prompt: str,
    rollouts: list[str],
    judge_result: JudgeResult,
    domain: str,
    weights: dict[str, float],
    tokenizer: Any | None,
) -> dict[str, Any]:
    """Assemble the combined-objectives spec for ``TrainEngine._step_multi``.

    Routing:

    - ``correct``  → ``weighted_sft`` (one entry per correct rollout, weight
      ``weights["sft"]``). Reusing the existing primitive — no new objective.
    - ``wrong``    → ``coh`` if a critique was produced (one entry per
      wrong+critique rollout, weight ``weights["coh"]``).
    - ``ambiguous`` → ``kto`` with label ``undesirable`` (one entry per
      ambiguous rollout, weight ``weights["kto"]``).
    - any rollout with a non-empty counterfactual ⇒ ``unlike`` (one entry
      per counterfactual, weight ``weights["unlike"]``). Skipped when the
      tokenizer is not available — we can't compute ``bad_token_id``
      without it. ``good_token_id`` is omitted; the resulting pure-unlike
      sample is anchored by the always-on ``kl_anchor`` batch objective
      with ``scope="target_position"``.
    - always: a ``kl_anchor`` batch objective (``weights["kl"]``).

    The return value is shaped exactly like the spec consumed by
    ``submit_train`` in multi-objective form: ``{"objectives": [...],
    "batch_objectives": [...]}``. No ``samples`` at the top level; every
    primary entry carries its own.

    Skipped sub-entries (e.g. unlike when tokenizer is None or a
    counterfactual is empty) leave a breadcrumb in the returned
    ``_skipped`` field so the per-step log can surface the reason.
    """
    objectives: list[dict[str, Any]] = []
    skipped: list[str] = []

    n = len(rollouts)
    grades = list(judge_result.get("grades", []))
    critiques = list(judge_result.get("critiques", []))
    counterfactuals = list(judge_result.get("counterfactuals", []))
    demonstration = judge_result.get("demonstration") or ""

    # Pad to n in case the teacher returned a short list (defensive — the
    # parser should already have raised; this is belt-and-braces).
    while len(grades) < n:
        grades.append("ambiguous")
    while len(critiques) < n:
        critiques.append(None)
    while len(counterfactuals) < n:
        counterfactuals.append(None)

    w_sft = float(weights.get("sft", 0.1))
    w_coh = float(weights.get("coh", 1.0))
    w_kto = float(weights.get("kto", 1.0))
    w_unlike = float(weights.get("unlike", 0.5))
    w_kl = float(weights.get("kl", 0.05))

    for i, (grade, rollout) in enumerate(zip(grades, rollouts)):
        if grade == "correct":
            objectives.append({
                "name": "weighted_sft",
                "weight": w_sft,
                "samples": [{
                    "prompt": prompt,
                    "response": rollout,
                    "weight": 1.0,
                }],
            })
        elif grade == "wrong":
            critique = critiques[i]
            if critique:
                objectives.append({
                    "name": "coh",
                    "weight": w_coh,
                    "samples": [{
                        "prompt": prompt,
                        "bad": rollout,
                        "critique": critique,
                        "good": demonstration or None,
                    }],
                })
            else:
                skipped.append(f"coh[{i}]: no critique")
        elif grade == "ambiguous":
            objectives.append({
                "name": "kto",
                "weight": w_kto,
                "samples": [{
                    "prompt": prompt,
                    "response": rollout,
                    "label": "undesirable",
                }],
            })
        else:
            skipped.append(f"unknown grade {grade!r} at idx {i}")

    # Counterfactuals → unlike. Pure-unlike (no good_token_id) is anchored
    # by the kl_anchor batch objective below. We tokenize using the live
    # tokenizer so the bad_token_id matches the model's vocabulary; in
    # dry-run / test contexts where tokenizer is None we record a skip
    # rather than emit a malformed entry.
    for i, cf in enumerate(counterfactuals):
        if not isinstance(cf, str) or not cf.strip():
            continue
        if tokenizer is None:
            skipped.append(f"unlike[{i}]: no tokenizer (dry-run / test)")
            continue
        bad_id = _bad_token_id_for(tokenizer, cf)
        if bad_id is None:
            skipped.append(f"unlike[{i}]: could not tokenize counterfactual")
            continue
        objectives.append({
            "name": "unlike",
            "weight": w_unlike,
            "samples": [{
                "prefix": prompt,
                "bad_token_id": bad_id,
                # No good_token_id → pure-unlike. Anchored by the
                # target-position kl_anchor below (Tier 1 / Tier 2 of the
                # unlike precondition gate).
                "weight": 1.0,
            }],
        })

    batch_objectives = [
        {
            "name": "kl_anchor",
            "weight": w_kl,
            "scope": "target_position",
        },
    ]

    spec: dict[str, Any] = {
        "objectives": objectives,
        "batch_objectives": batch_objectives,
        "_rlvr": {
            "domain": domain,
            "n_rollouts": n,
            "grades": grades,
            "skipped": skipped,
        },
    }
    return spec


# ---------------------------------------------------------------- scheduler
class RLVRScheduler:
    """Bound the run; let the user / CLI decide when to stop.

    Unlike the idle-gated TTRL scheduler this is on-demand — caller drives
    ``run(n=...)`` because the RLVR loop has external (paid) HTTP calls
    that we don't want to fire opportunistically without supervision.
    """

    def __init__(
        self,
        controller: "Controller",
        config: RLVRConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        self.controller = controller
        self.cfg = config
        self.dry_run = dry_run
        self.stats: dict[str, Any] = {
            "steps": 0,
            "submitted": 0,
            "skipped": 0,
            "errors": 0,
        }

    async def _sample_rollouts(self, prompt: str) -> list[str]:
        """``k`` sequential ``controller.generate`` calls at sampling temperature.

        Mirrors ``ttrl_mv._sample_rollouts``. Failures within a rollout
        are logged and skipped — we still proceed if at least one came
        back.
        """
        messages = [{"role": "user", "content": prompt}]
        out: list[str] = []
        for _ in range(self.cfg.k):
            try:
                result = await self.controller.generate(
                    messages,
                    temperature=self.cfg.sampling_temperature,
                    top_p=self.cfg.sampling_top_p,
                    max_new_tokens=self.cfg.max_new_tokens,
                )
            except Exception:
                log.exception("rlvr rollout failed (prompt head=%r)", prompt[:60])
                continue
            text = _select_text(result if isinstance(result, dict) else {})
            out.append(text)
        return out

    async def _step_one(self, source_label: str, prompt: str) -> dict[str, Any]:
        rollouts = await self._sample_rollouts(prompt)
        if not rollouts:
            self.stats["skipped"] += 1
            return {"step": "skipped", "reason": "no rollouts"}

        # Determine domain. ARC source overrides the verifier — the ARC
        # prompt header is stable enough that ``select`` claims it, but
        # the user explicitly chose this source so we honor that.
        if source_label == "arc":
            domain: str = "arc"
        else:
            domain = select_verifier(prompt) or source_label or "general"

        try:
            judge_result = judge(prompt, rollouts, domain=domain)  # type: ignore[arg-type]
        except Exception as exc:
            log.exception("rlvr judge failed (prompt head=%r)", prompt[:60])
            self.stats["errors"] += 1
            return {"step": "error", "stage": "judge", "error": str(exc)}

        tokenizer = None
        state = getattr(self.controller, "state", None)
        if state is not None:
            tokenizer = getattr(state, "tokenizer", None)

        spec = build_combined_spec(
            prompt=prompt,
            rollouts=rollouts,
            judge_result=judge_result,
            domain=domain,
            weights=self.cfg.weights,
            tokenizer=tokenizer,
        )

        record: dict[str, Any] = {
            "ts": time.time(),
            "source": source_label,
            "domain": domain,
            "prompt_head": prompt[:120],
            "n_rollouts": len(rollouts),
            "grades": list(judge_result.get("grades", [])),
            "objective_counts": _count_by_name(spec["objectives"]),
            "skipped": spec["_rlvr"]["skipped"],
            "weights": dict(self.cfg.weights),
        }

        if self.dry_run or not spec["objectives"]:
            record["mode"] = "dry-run" if self.dry_run else "no-objectives"
            self.stats["skipped"] += 1
            return {"step": "dry-run", "spec": spec, "record": record}

        try:
            ack = await self.controller.submit_train(spec)
        except Exception as exc:
            log.exception("rlvr submit_train failed")
            self.stats["errors"] += 1
            record["error"] = f"submit_train: {exc}"
            return {"step": "error", "stage": "submit", "error": str(exc),
                    "record": record}

        commit_token = ack.get("commit_token")
        record["commit_token"] = commit_token
        record["batch_id"] = ack.get("batch_id")

        # Wait on the queue so the next iteration sees the post-step model.
        if commit_token is not None:
            try:
                await self.controller.queue.wait_for(
                    int(commit_token), timeout=120.0,
                )
            except Exception:
                log.exception(
                    "rlvr wait_for(token=%s) failed", commit_token,
                )

        self.stats["submitted"] += 1
        return {"step": "submitted", "spec": spec, "record": record}

    async def run(self, n: int, *, save_every: int = 0,
                  snapshot_prefix: str = "rlvr") -> list[dict[str, Any]]:
        """Run ``n`` RLVR steps; return per-step records.

        Per-step records are also appended to ``cfg.log_path`` as JSONL
        so the run is durable even when the caller forgets to capture
        the return value.

        ``save_every`` (0 = off): every N successfully submitted steps,
        request a daemon snapshot named ``{snapshot_prefix}-step-{i}``.
        Skipped in dry-run mode (no commits land anyway).
        """
        log_path = Path(self.cfg.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        prompt_iter = iter_prompts(self.cfg.source)
        for _ in range(n):
            self.stats["steps"] += 1
            try:
                source_label, prompt = next(prompt_iter)
            except StopIteration:
                break
            outcome = await self._step_one(source_label, prompt)
            rec = outcome.get("record")
            if rec is not None:
                records.append(rec)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (save_every > 0
                    and not self.dry_run
                    and outcome.get("step") == "submitted"
                    and self.stats["submitted"] % save_every == 0):
                snap_name = f"{snapshot_prefix}-step-{self.stats['submitted']:03d}"
                try:
                    await self._snapshot_save(snap_name)
                    self.stats.setdefault("snapshots", 0)
                    self.stats["snapshots"] += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "rlvr progressive save %r failed: %s", snap_name, exc,
                    )
        return records

    async def _snapshot_save(self, name: str) -> None:
        """Request a daemon snapshot via the controller's HTTP path or its
        in-process Controller method, whichever is available."""
        ctrl = self.controller
        # Prefer the in-process API if present (full Controller has
        # ``request_snapshot_save``); otherwise fall back to the HTTP shim.
        method = getattr(ctrl, "request_snapshot_save", None)
        if callable(method):
            await method(name)
            return
        post = getattr(ctrl, "_post", None)
        if callable(post):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: post("/v1/state/snapshot/save", {"name": name}),
            )
            return
        raise RuntimeError("controller exposes neither request_snapshot_save "
                           "nor _post; cannot snapshot")


def _count_by_name(objectives: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objectives:
        name = obj.get("name", "?")
        counts[name] = counts.get(name, 0) + 1
    return counts


# ---------------------------------------------------------------- CLI client
class _HttpDaemonController:
    """Minimal Controller stand-in that talks to a running daemon over HTTP.

    Used by the ``--daemon URL`` CLI flag so the RLVR loop can be driven
    against an already-running ``lile`` server without importing the in-
    process Controller (which pulls in torch / unsloth). Provides only the
    surface that ``RLVRScheduler`` calls: ``generate``, ``submit_train``,
    ``state.tokenizer``, ``queue.wait_for``.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.state = type("S", (), {"tokenizer": None})()
        self.queue = self  # ``wait_for`` is a no-op below.

    def _post(self, path: str, body: dict, timeout: float = 120.0) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        # The daemon's /v1/chat/completions is OpenAI-compatible. We strip
        # decoding kwargs the daemon doesn't accept and forward the rest.
        body = {
            "model": "lile",
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": kwargs.get("max_new_tokens", 512),
            "stream": False,
        }
        loop = asyncio.get_running_loop()
        envelope = await loop.run_in_executor(
            None, lambda: self._post("/v1/chat/completions", body),
        )
        try:
            choice = envelope["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError):
            return {"response": "", "reasoning_content": None, "raw": ""}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content")
        raw = content
        if reasoning and content:
            raw = f"<think>{reasoning}</think>{content}"
        elif reasoning:
            raw = f"<think>{reasoning}</think>"
        return {
            "response": content,
            "reasoning_content": reasoning,
            "raw": raw,
        }

    async def submit_train(self, spec: dict[str, Any]) -> dict[str, Any]:
        # Track A's TrainRequest schema accepts ``objectives``.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._post("/v1/train", spec),
        )

    async def wait_for(self, token: int, timeout: float = 120.0) -> Any:
        # The daemon owns the actual queue; HTTP path is fire-and-forget
        # for our purposes. The next chat call's ``after_commit_token``
        # would be the canonical wait — out of scope for this CLI.
        return None


# ---------------------------------------------------------------- main
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lile.teach.rlvr_loop",
        description="Online RLVR scheduler — Track C of the dreamy-doodling plan.",
    )
    p.add_argument(
        "--source", choices=("math", "code", "arc", "mixed"), default="mixed",
    )
    p.add_argument("--n", type=int, default=1)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip submit_train; log the would-be spec only.",
    )
    p.add_argument(
        "--daemon", default="http://127.0.0.1:8768",
        help="Base URL of a running lile daemon (used for generate + submit).",
    )
    p.add_argument("--log-path", default="lile_data/rlvr_loop.jsonl")
    p.add_argument(
        "--save-every", type=int, default=5,
        help="Snapshot every N successful steps; 0 disables progressive save.",
    )
    p.add_argument(
        "--snapshot-prefix", default="rlvr",
        help="Snapshot name prefix; the step index is appended.",
    )
    return p


async def _amain(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "rlvr_loop: OPENROUTER_API_KEY is not set in the shell environment; "
            "the teacher call will fail. Export the key and re-run.",
            flush=True,
        )
        return 2
    controller = _HttpDaemonController(args.daemon)
    cfg = RLVRConfig(source=args.source, log_path=args.log_path)
    sched = RLVRScheduler(controller, cfg, dry_run=args.dry_run)
    records = await sched.run(
        args.n,
        save_every=args.save_every,
        snapshot_prefix=args.snapshot_prefix,
    )
    summary = {
        "n_steps": len(records),
        "stats": sched.stats,
        "records": records,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_argparser().parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
