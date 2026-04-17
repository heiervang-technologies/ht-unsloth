"""The controller: single writer that serializes all GPU-mutating operations.

Inference requests coexist freely with training, but weight-mutating operations
(train step, merge, adapter-swap, snapshot-restore) MUST go through the
compute queue so the commit-cursor ordering invariant holds.

This module is the glue between `server` (HTTP) and `engine` (GPU work).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from .config import ServeConfig
from .engine.train import TrainEngine
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

        # Feedback-event bookkeeping: response_id -> original inference record.
        self._response_index: dict[str, dict[str, Any]] = {}

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
        self.train_engine = TrainEngine(self.state, lr=self.cfg.default_lr)
        await self.queue.start(self._handle_task)
        log.info("controller started on %s", self.cfg.model)

    async def stop(self) -> None:
        await self.queue.stop()

    # ------------------------------------------------------------------ queue handler
    def _handle_task(self, task) -> Any:
        """Runs on the single worker thread. May raise; queue catches and stores."""
        t0 = time.time()
        kind = task.kind
        payload = task.payload
        if kind == "train":
            result = self.train_engine.step(payload)
            # Canonical log entry for every committed step.
            self.trajectory.log_train(
                batch_id=task.batch_id,
                objective=payload.get("objective", ""),
                loss=result.get("loss") or 0.0,
                batch_size=len(payload.get("samples", [])),
                commit_token=task.token,
            )
            return {"loss": result.get("loss"), "components": result.get("components"),
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
        # Run the actual generation outside the queue — training+inference
        # share weights; there is no race because training mutates atomically
        # via the compute queue's single-worker discipline.
        from .engine.inference import generate_chat
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
        self._response_index[rid] = {
            "messages": messages, "response": text, "ts": time.time(),
        }
        # Evict old entries to cap memory.
        if len(self._response_index) > 4096:
            oldest = sorted(self._response_index.items(), key=lambda kv: kv[1]["ts"])[:1024]
            for k, _ in oldest:
                self._response_index.pop(k, None)
        return {"response_id": rid, "response": text}

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

    async def submit_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Route feedback to the appropriate training objective (see §5b.3/§5c.16)."""
        rid = payload.get("response_id")
        kind = payload.get("kind")
        prior = self._response_index.get(rid) if rid else None
        if prior is None and "prompt" not in payload:
            return {"error": f"unknown response_id {rid!r}; include prompt in payload "
                             "to bypass index lookup"}
        prompt = (payload.get("prompt")
                  or (prior.get("messages", [{}])[-1].get("content") if prior else ""))
        bad_response = payload.get("response") or (prior.get("response") if prior else "")

        # Always log first.
        self.trajectory.log_feedback(rid or "", kind=kind or "unknown", **{
            k: v for k, v in payload.items() if k not in {"response_id", "kind"}
        })

        # Routing.
        if kind == "binary":
            label = "desirable" if payload.get("value") == "up" else "undesirable"
            spec = {
                "objective": "kto",
                "samples": [{"prompt": prompt, "response": bad_response, "label": label}],
            }
        elif kind == "rewrite" or kind == "preferred":
            spec = {
                "objective": "weighted_sft",
                "samples": [{
                    "prompt": prompt,
                    "response": payload["better_response"],
                    "weight": payload.get("weight", 3.0),
                }],
            }
        elif kind == "nl_critique":
            spec = {
                "objective": "coh",
                "samples": [{
                    "prompt": prompt,
                    "bad": bad_response,
                    "critique": payload["critique"],
                }],
            }
        elif kind == "nl_critique_with_rewrite":
            spec = {
                "objective": "coh",
                "samples": [{
                    "prompt": prompt,
                    "bad": bad_response,
                    "critique": payload["critique"],
                    "good": payload["better_response"],
                }],
            }
        else:
            return {"error": f"unsupported feedback kind {kind!r}"}

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
