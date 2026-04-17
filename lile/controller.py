"""The controller: single writer that serializes all GPU-mutating operations.

Inference requests coexist freely with training, but weight-mutating operations
(train step, merge, adapter-swap, snapshot-restore) MUST go through the
compute queue so the commit-cursor ordering invariant holds.

This module is the glue between `server` (HTTP) and `engine` (GPU work).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
from pathlib import Path
from typing import Any

# Response index cap: after this many live responses we start evicting the
# oldest entries. OrderedDict gives O(1) insertion, lookup, and eviction.
_RESPONSE_INDEX_CAP = 4096

from .config import ServeConfig
from .engine.replay import IdleReplayScheduler, ReplayPolicy
from .engine.train import TrainEngine
from .logging_backends import LoggerConfig, MetricsLogger, flatten_scalars, get_logger
from .queue import ComputeQueue, new_batch_id
from .snapshot import SnapshotManager
from .state import ModelState
from .trajectory import TrajectoryLog, new_response_id

log = logging.getLogger(__name__)


class Controller:
    def __init__(self, cfg: ServeConfig) -> None:
        self.cfg = cfg
        self.state: ModelState | None = None
        self.queue = ComputeQueue(max_depth=cfg.max_queue_depth)
        self.train_engine: TrainEngine | None = None
        self.trajectory = TrajectoryLog(cfg.data_dir / "trajectory.jsonl")
        self.snapshots = SnapshotManager(cfg.data_dir / "snapshots")
        self.metrics_logger: MetricsLogger = get_logger(LoggerConfig(
            backend=cfg.logger,
            project=cfg.logger_project,
            run_name=cfg.logger_run_name,
            log_dir=cfg.logger_log_dir,
            tracking_uri=cfg.logger_tracking_uri,
        ))

        # Feedback-event bookkeeping: response_id -> original inference record.
        # OrderedDict keeps insertion order so ``popitem(last=False)`` evicts
        # oldest in O(1). Previously the eviction path did
        # ``sorted(..., key=ts)[:1024]`` — O(n log n) per generate above the
        # cap. See PR#8 review.
        self._response_index: "collections.OrderedDict[str, dict[str, Any]]" = (
            collections.OrderedDict()
        )

        # T4.1 idle replay; instantiated in start() once state is loaded.
        self._replay: IdleReplayScheduler | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.state = ModelState.load(
            model_name=self.cfg.model,
            max_seq_length=self.cfg.max_seq_length,
            lora_rank=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            load_in_4bit=self.cfg.load_in_4bit,
        )
        if self.cfg.frozen_ref:
            self.state.load_frozen_ref()
        self.train_engine = TrainEngine(self.state, lr=self.cfg.default_lr)
        # Stamp run-level params into the external logger once the state is
        # loaded; NullLogger swallows this, real backends record it as
        # hyperparameters on the run.
        self.metrics_logger.log_params({
            "model": self.cfg.model,
            "lora_rank": self.cfg.lora_rank,
            "lora_alpha": self.cfg.lora_alpha,
            "default_lr": self.cfg.default_lr,
            "default_objective": self.cfg.default_objective,
            "frozen_ref": bool(self.cfg.frozen_ref),
        })
        await self.queue.start(self._handle_task)
        if self.cfg.idle_replay:
            self._replay = IdleReplayScheduler(self, ReplayPolicy.from_config(self.cfg))
            await self._replay.start()
        log.info("controller started on %s", self.cfg.model)

    async def stop(self) -> None:
        if self._replay is not None:
            await self._replay.stop()
            self._replay = None
        await self.queue.stop()
        try:
            self.metrics_logger.close()
        except Exception as exc:  # pragma: no cover
            log.warning("metrics_logger close failed: %s", exc)

    # ------------------------------------------------------------------ queue handler
    def _handle_task(self, task) -> Any:
        """Runs on the single worker thread. May raise; queue catches and stores."""
        t0 = time.time()
        kind = task.kind
        payload = task.payload
        if kind == "train":
            result = self.train_engine.step(payload)
            components = result.get("components")
            # Canonical log entry for every committed step.
            self.trajectory.log_train(
                batch_id=task.batch_id,
                objective=payload.get("objective", ""),
                loss=result.get("loss") or 0.0,
                batch_size=len(payload.get("samples", [])),
                commit_token=task.token,
                components=components,
            )
            # Fan out scalar metrics to the external sink (no-op for NullLogger).
            scalars = flatten_scalars(components or {})
            if scalars:
                self.metrics_logger.log_metrics(scalars, step=task.token)
            return {"loss": result.get("loss"), "components": components,
                    "wall": time.time() - t0}
        elif kind == "merge":
            self.state.merge_active_into_residual()
            return {"merges_applied": self.state.merges_applied,
                    "residual_fp": self.state.residual_fingerprint(),
                    "wall": time.time() - t0}
        elif kind == "snapshot_save":
            name = payload["name"]
            self.snapshots.save(name, self.state, self.trajectory)
            return {"saved": name, "wall": time.time() - t0}
        elif kind == "snapshot_load":
            name = payload["name"]
            manifest = self.snapshots.load(name, self.state)
            return {"loaded": name, "manifest": manifest, "wall": time.time() - t0}
        elif kind == "reset_adapter":
            self.state.reset_active_adapter()
            return {"ok": True, "wall": time.time() - t0}
        else:
            raise ValueError(f"unknown task kind {kind!r}")

    # ------------------------------------------------------------------ public ops
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Wait until the queue has drained all committed-before-this-request
        training, then generate. This is the 'POST a batch, next inference
        sees it' guarantee from the caller's viewpoint: callers may pass
        `after_commit_token` to block on that specific training."""
        wait_for = kwargs.pop("after_commit_token", None)
        if wait_for is not None:
            try:
                await self.queue.wait_for(int(wait_for), timeout=60.0)
            except (asyncio.TimeoutError, KeyError):
                pass
        parse_reasoning = kwargs.pop("parse_reasoning", True)
        # Run the actual generation outside the queue — training+inference
        # share weights; there is no race because training mutates atomically
        # via the compute queue's single-worker discipline.
        from .engine.inference import generate_chat
        from .reasoning import get_parser_for_model
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            lambda: generate_chat(
                self.state.model, self.state.tokenizer, messages,
                mode_lock=self.state.mode_lock, **kwargs,
            ),
        )
        rid = new_response_id()
        self.trajectory.log_inference(
            response_id=rid,
            prompt=str(messages[-1].get("content", "")),
            response=text,
            model_fingerprint=self.state.residual_fingerprint()[:16],
        )
        self._remember_response(rid, messages, text)
        reasoning: str | None = None
        content: str = text
        if parse_reasoning and kwargs.get("enable_thinking") is not False:
            parser = get_parser_for_model(self.state.base_model_name or "")
            if parser is not None:
                r, c = parser.extract_final(text)
                reasoning = r.strip() or None
                content = c.strip() if c else ""
        return {"response_id": rid, "response": content,
                "reasoning_content": reasoning, "raw": text}

    async def stream_generate(self, messages: list[dict[str, str]],
                              **kwargs: Any):
        """Async generator yielding {delta, response_id} chunks, then a final
        {final: True, response_id, full, commit_cursor} event.

        Runs the generator thread-side (see engine.generate_chat_stream) and
        shuttles chunks through an asyncio.Queue so the FastAPI event loop
        stays responsive.
        """
        wait_for = kwargs.pop("after_commit_token", None)
        if wait_for is not None:
            try:
                await self.queue.wait_for(int(wait_for), timeout=60.0)
            except (asyncio.TimeoutError, KeyError):
                pass
        parse_reasoning = kwargs.pop("parse_reasoning", True)

        from .engine.inference import generate_chat_stream
        from .reasoning import get_parser_for_model
        import threading
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        DONE = object()
        ERR: dict[str, Any] = {}

        def _producer():
            try:
                for chunk in generate_chat_stream(
                    self.state.model, self.state.tokenizer, messages,
                    mode_lock=self.state.mode_lock, **kwargs,
                ):
                    asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
            except Exception as e:
                ERR["exc"] = e
            finally:
                asyncio.run_coroutine_threadsafe(q.put(DONE), loop)

        threading.Thread(target=_producer, daemon=True).start()
        rid = new_response_id()
        full_parts: list[str] = []
        # Parser is active iff the request wants reasoning parsing AND the
        # caller did not explicitly disable thinking.  When disabled, the
        # model emits pure content (no tags) so the parser would be a no-op
        # anyway — skipping it keeps the hot path cheap.
        parser_state = None
        if parse_reasoning and kwargs.get("enable_thinking") is not False:
            parser = get_parser_for_model(self.state.base_model_name or "")
            if parser is not None:
                parser_state = parser.make_state()
        while True:
            chunk = await q.get()
            if chunk is DONE:
                break
            full_parts.append(chunk)
            if parser_state is not None:
                r_delta, c_delta = parser_state.feed(chunk)
                if r_delta or c_delta:
                    yield {"delta": c_delta, "reasoning_delta": r_delta,
                           "response_id": rid}
            else:
                yield {"delta": chunk, "reasoning_delta": "",
                       "response_id": rid}

        if "exc" in ERR:
            yield {"error": str(ERR["exc"]), "response_id": rid}
            return

        # Flush any bytes the parser was holding back waiting for a delimiter.
        if parser_state is not None:
            r_tail, c_tail = parser_state.finalize()
            if r_tail or c_tail:
                yield {"delta": c_tail, "reasoning_delta": r_tail,
                       "response_id": rid}

        full_text = "".join(full_parts).strip()
        self.trajectory.log_inference(
            response_id=rid,
            prompt=str(messages[-1].get("content", "")),
            response=full_text,
            model_fingerprint=self.state.residual_fingerprint()[:16],
        )
        self._remember_response(rid, messages, full_text)
        yield {"final": True, "response_id": rid, "full": full_text,
               "commit_cursor": self.queue.committed}

    def _remember_response(self, rid: str, messages: list[dict[str, str]],
                           response_text: str) -> None:
        """O(1) insertion + LRU eviction via OrderedDict."""
        self._response_index[rid] = {
            "messages": messages, "response": response_text, "ts": time.time(),
        }
        while len(self._response_index) > _RESPONSE_INDEX_CAP:
            self._response_index.popitem(last=False)

    async def submit_train(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Chunk a train batch into queue tasks and return the final commit_token."""
        batch_id = new_batch_id()
        samples = spec.get("samples", [])
        chunk_size = spec.get("chunk_size", 2)  # small default for 0.6B; caller can bump
        tasks = []
        for i in range(0, max(1, len(samples)), chunk_size):
            sub = {
                **spec,
                "samples": samples[i:i + chunk_size] if samples else samples,
            }
            t = await self.queue.submit("train", sub, batch_id=batch_id)
            tasks.append(t)
            if not samples:
                break
        # The commit_token is the last task's token.
        commit_token = tasks[-1].token
        return {
            "batch_id": batch_id,
            "commit_token": commit_token,
            "n_chunks": len(tasks),
            "queue_depth": self.queue._q.qsize(),
        }

    @staticmethod
    def feedback_to_batch(
        record: dict[str, Any],
        prompt_fallback: str | None = None,
        response_fallback: str | None = None,
    ) -> dict[str, Any] | None:
        """Route a feedback payload to a train spec — pure, no Controller state.

        Used by both the live ``submit_feedback`` path and the idle replay
        scheduler (§T4.1). The scheduler reads feedback records directly from
        the trajectory log, so it cannot consult the in-memory response index;
        ``prompt`` and ``response`` must either be present in the record or
        supplied via the fallback args.

        Accepts both the external payload shape (``kind=...``) and the
        trajectory-logged shape (``feedback_kind=...``, which ``log_feedback``
        stamps in). Returns ``None`` when routing is impossible — caller can
        log/skip rather than raising.
        """
        # Trajectory-logged records stamp the routing kind under
        # ``feedback_kind`` (top-level ``kind`` is the event kind, always
        # "feedback" on those records). Live-payload callers use ``kind``
        # directly. Prefer ``feedback_kind`` so replay reads resolve
        # correctly; fall back to ``kind`` but ignore the sentinel event
        # kind "feedback" which carries no routing information.
        kind = record.get("feedback_kind")
        if not kind:
            top = record.get("kind")
            if top and top != "feedback":
                kind = top
        prompt = record.get("prompt") or prompt_fallback or ""
        bad_response = record.get("response") or response_fallback or ""
        if not prompt:
            return None

        if kind == "binary":
            label = "desirable" if record.get("value") == "up" else "undesirable"
            return {
                "objective": "kto",
                "samples": [{"prompt": prompt, "response": bad_response, "label": label}],
            }
        if kind in ("rewrite", "preferred"):
            better = record.get("better_response")
            if not better:
                return None
            return {
                "objective": "weighted_sft",
                "samples": [{
                    "prompt": prompt,
                    "response": better,
                    "weight": record.get("weight", 3.0),
                }],
            }
        if kind == "nl_critique":
            critique = record.get("critique")
            if not critique:
                return None
            return {
                "objective": "coh",
                "samples": [{
                    "prompt": prompt,
                    "bad": bad_response,
                    "critique": critique,
                }],
            }
        if kind == "nl_critique_with_rewrite":
            critique = record.get("critique")
            better = record.get("better_response")
            if not critique or not better:
                return None
            return {
                "objective": "coh",
                "samples": [{
                    "prompt": prompt,
                    "bad": bad_response,
                    "critique": critique,
                    "good": better,
                }],
            }
        return None

    async def submit_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Route feedback to the appropriate training objective (see §5b.3/§5c.16)."""
        rid = payload.get("response_id")
        kind = payload.get("kind")
        prior = self._response_index.get(rid) if rid else None
        if prior is None and "prompt" not in payload:
            return {"error": f"unknown response_id {rid!r}; include prompt in payload "
                             "to bypass index lookup"}

        prompt_fallback = (prior.get("messages", [{}])[-1].get("content")
                           if prior else None)
        response_fallback = prior.get("response") if prior else None

        # Log the feedback record with prompt/response materialized from the
        # response index. Without this, idle replay (§T4.1) reading the log
        # later cannot reconstruct the batch — the in-memory index is
        # ephemeral, the log is canonical.
        log_fields = {k: v for k, v in payload.items()
                      if k not in {"response_id", "kind"}}
        if "prompt" not in log_fields and prompt_fallback:
            log_fields["prompt"] = prompt_fallback
        if "response" not in log_fields and response_fallback:
            log_fields["response"] = response_fallback
        self.trajectory.log_feedback(rid or "", kind=kind or "unknown", **log_fields)

        spec = self.feedback_to_batch(
            payload,
            prompt_fallback=prompt_fallback,
            response_fallback=response_fallback,
        )
        if spec is None:
            return {"error": f"unsupported or under-specified feedback kind {kind!r}"}
        return await self.submit_train(spec)

    async def request_merge(self) -> dict[str, Any]:
        task = await self.queue.submit("merge", {})
        result = await self.queue.wait_for(task.token, timeout=300.0)
        return {"commit_token": task.token, "result": result.result, "error": str(result.error) if result.error else None}

    async def request_snapshot_save(self, name: str) -> dict[str, Any]:
        task = await self.queue.submit("snapshot_save", {"name": name})
        result = await self.queue.wait_for(task.token, timeout=300.0)
        return {"commit_token": task.token, "result": result.result}

    async def request_snapshot_load(self, name: str) -> dict[str, Any]:
        task = await self.queue.submit("snapshot_load", {"name": name})
        result = await self.queue.wait_for(task.token, timeout=300.0)
        return {"commit_token": task.token, "result": result.result}
