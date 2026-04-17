"""Controller — the single GPU writer.

Serialises *all* mutating operations on the live model:
* training steps,
* progressive merges,
* snapshot save / restore,
* (later) adapter swaps.

Reads (inference) acquire a per-call read mode via the controller's
:meth:`read_mode` context manager so we don't accidentally have a training step
mid-flight while a generation is decoding.

The controller owns:
    * :class:`lile.state.LiveState`
    * :class:`lile.adapters.AdapterManager`
    * :class:`lile.queue.ComputeQueue` + worker thread
    * :class:`lile.engine.train.TrainEngine`
    * :class:`lile.engine.inference.InferenceEngine`
    * :class:`lile.trajectory.TrajectoryLog`

It is the only component that the FastAPI server talks to directly.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from lile.adapters import AdapterManager
from lile.engine.inference import ChatMessage, GenerationResult, InferenceEngine
from lile.engine.train import StepResult, TrainEngine
from lile.objectives import Batch, Sample
from lile.queue import ComputeQueue, ComputeWorker, CommitToken
from lile.snapshot import restore_snapshot, save_snapshot
from lile.state import LiveState, StateConfig
from lile.trajectory import TrajectoryLog


@dataclass
class ControllerConfig:
    state: StateConfig
    work_dir: str = "./.lile"
    lr: float = 1e-5
    frozen_ref: bool = False
    idle_replay: bool = False
    idle_replay_threshold_s: float = 30.0


class Controller:
    """Single owner of the live model + queue."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Lifecycle: load lazily so __init__ can be called in tests w/o CUDA.
        self.state: LiveState | None = None
        self.adapter_mgr: AdapterManager | None = None
        self.train_engine: TrainEngine | None = None
        self.infer_engine: InferenceEngine | None = None
        self.trajectory: TrajectoryLog | None = None
        self.queue: ComputeQueue | None = None
        self.worker: ComputeWorker | None = None
        # Set in start() if config.idle_replay; declared here so attribute access
        # is safe before start().
        self.replay_scheduler: Any = None
        # Phase 6 — populated only when state.inference_backend == "vllm_sidecar".
        self.sidecar: Any = None
        self.weight_sync: Any = None

        # GPU access is serialised: training+merge takes the write lock; gen
        # takes a slot too because we only have one CUDA context. We use a
        # single threading.Lock — generation latency is OK since training is
        # short (~seconds).
        self._gpu_lock = threading.Lock()

        # Map response_id → (messages, completion) for feedback routing.
        self._response_cache: dict[str, dict[str, Any]] = {}

        # Reference model for KL anchor / KTO / CCPD π_old. Defaults to the
        # current model under no_grad — conservative ("EMA of current"); the
        # plan suggests true frozen base for production, which we'll wire when
        # the cost is justified.
        self._ref_model = None

    # --- Lifecycle ---------------------------------------------------------

    def start(self) -> "Controller":
        # Fail fast on misconfiguration before paying for the model load.
        # Phase 6 — if the operator asked for the sidecar, vLLM must be
        # importable. Failing here saves the ~15-60 s the model load takes.
        if self.config.state.inference_backend == "vllm_sidecar":
            from lile.engine.vllm_sidecar import is_available  # noqa: PLC0415
            if not is_available():
                raise RuntimeError(
                    "inference_backend=vllm_sidecar but vLLM is not "
                    "available in the environment. Install vllm or set "
                    "inference_backend=fast_generate."
                )

        # Trajectory log first — we want even setup events recorded.
        self.trajectory = TrajectoryLog(self.work_dir / "trajectory.jsonl")
        self.trajectory.append("info", event="controller-start", config=str(self.config))

        # Load the live model.
        self.state = LiveState(self.config.state).load()
        # NOTE: AdapterManager owns merged_deltas storage; we pass the dict by reference.
        self.adapter_mgr = AdapterManager(self.state.model, self.state.merged_deltas)
        self.train_engine = TrainEngine(self.state, lr=self.config.lr)
        self.infer_engine = InferenceEngine(self.state)
        if self.config.frozen_ref:
            # Second base-only copy, no PEFT, eval, no_grad. Ref doesn't drift
            # with training so KL/KTO/CCPD π_old is genuinely anchored.
            self._ref_model = self.state.load_frozen_ref()
            self.trajectory.append("info", event="frozen-ref-loaded")
        else:
            # EMA-1 fallback: ref aliases the live model, used under no_grad.
            # Cheaper but the anchor weakens as the model drifts.
            self._ref_model = self.state.model

        # Phase 6 — vLLM sidecar (optional). Loaded BEFORE the worker thread
        # so the first chat after start() can hit the sidecar if configured.
        # Availability was already verified at the top of start().
        if self.config.state.inference_backend == "vllm_sidecar":
            from lile.engine.vllm_sidecar import (  # noqa: PLC0415
                SidecarConfig, VLLMSidecar,
            )
            from lile.engine.weight_sync import WeightSyncBridge  # noqa: PLC0415
            sidecar_cfg = SidecarConfig(
                model_name=self.config.state.model_name,
                mode=self.config.state.sidecar_mode,
                sidecar_device=self.config.state.sidecar_device,
                gpu_memory_utilization=self.config.state.sidecar_gpu_memory_utilization,
                max_model_len=self.config.state.max_seq_length,
                max_lora_rank=max(64, self.config.state.lora_rank),
            )
            self.sidecar = VLLMSidecar(sidecar_cfg).load(self.state.tokenizer)
            self.weight_sync = WeightSyncBridge(self.sidecar)
            # Initial sync — sidecar starts with the trainer's current adapter.
            from lile.adapters import lora_state_dict  # noqa: PLC0415
            self.weight_sync.push_active_lora(lora_state_dict(self.state.model))
            self.trajectory.append(
                "info", event="vllm-sidecar-loaded",
                mode=self.config.state.sidecar_mode,
            )

        self.queue = ComputeQueue()
        self.worker = ComputeWorker(self.queue, handler=self._handle_compute)
        self.worker.start()

        # T4.1 — idle replay scheduler. Imported here (not at module top) so
        # tests that monkeypatch only the controller path don't transitively
        # need the engine import. Safe because controller.start() is the only
        # place the scheduler is constructed.
        self.replay_scheduler = None
        if self.config.idle_replay:
            from lile.engine.replay import IdleReplayScheduler, ReplayPolicy
            self.replay_scheduler = IdleReplayScheduler(
                self,
                policy=ReplayPolicy(
                    idle_threshold_s=self.config.idle_replay_threshold_s,
                ),
            ).start()

        self.trajectory.append(
            "info", event="controller-ready",
            vram=self.state.vram_summary(),
            model_name=self.config.state.model_name,
            frozen_ref=self.config.frozen_ref,
            idle_replay=self.config.idle_replay,
        )
        return self

    def shutdown(self) -> None:
        if getattr(self, "replay_scheduler", None) is not None:
            self.replay_scheduler.stop(timeout=5.0)
            self.replay_scheduler = None
        if self.worker is not None:
            self.worker.stop(timeout=5.0)
            self.worker = None
        if self.trajectory is not None:
            self.trajectory.append("info", event="controller-shutdown")
            self.trajectory.close()
            self.trajectory = None

    # --- Inference ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        wait_for: CommitToken | None = None,
    ) -> GenerationResult:
        """Generate one chat completion. Optionally block until ``wait_for`` commits."""
        if self.queue is None or self.infer_engine is None or self.state is None:
            raise RuntimeError("Controller not started")
        if wait_for is not None:
            ok = self.queue.wait_for_commit(wait_for, timeout=60.0)
            if not ok:
                raise TimeoutError(f"timed out waiting for commit {wait_for!r}")
        msgs = [ChatMessage.from_dict(m) for m in messages]
        if self.sidecar is not None:
            # Phase 6 — sidecar serves chat without touching _gpu_lock so
            # generation runs concurrently with training. The sidecar's
            # adapter is kept in sync via WeightSyncBridge after each merge.
            result = self.sidecar.generate(
                msgs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        else:
            with self._gpu_lock:
                self.state.set_inference_mode()
                result = self.infer_engine.generate(
                    msgs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
        self._response_cache[result.response_id] = {
            "messages": [m.__dict__ for m in msgs],
            "response": result.text,
            "ts": time.time(),
        }
        if self.trajectory is not None:
            # Replay scheduler reads `prompt`/`response` to reconstruct samples
            # without needing the (volatile, in-process) _response_cache.
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            self.trajectory.append(
                "chat",
                response_id=result.response_id,
                prompt=last_user,
                response=result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                elapsed_s=result.elapsed_s,
                committed_seq=self.queue.committed_seq,
            )
        return result

    # --- Training ----------------------------------------------------------

    def submit_train(self, batch: Batch) -> CommitToken:
        if self.queue is None:
            raise RuntimeError("Controller not started")
        token, _fut = self.queue.enqueue(("train", batch))
        if self.trajectory is not None:
            self.trajectory.append(
                "train",
                phase="enqueued",
                seq=token.seq,
                n_samples=len(batch.samples),
                batch_objectives=batch.batch_objectives,
            )
        return token

    def submit_merge(self) -> CommitToken:
        if self.queue is None:
            raise RuntimeError("Controller not started")
        token, _fut = self.queue.enqueue(("merge", None))
        if self.trajectory is not None:
            self.trajectory.append("merge", phase="enqueued", seq=token.seq)
        return token

    def submit_snapshot(self, name: str) -> CommitToken:
        if self.queue is None:
            raise RuntimeError("Controller not started")
        token, _fut = self.queue.enqueue(("snapshot", name))
        if self.trajectory is not None:
            self.trajectory.append("snapshot", phase="enqueued", seq=token.seq, name=name)
        return token

    def restore(self, name: str) -> dict[str, Any]:
        """Synchronous restore — drains queue first to avoid stale writes."""
        if self.queue is None or self.state is None or self.adapter_mgr is None:
            raise RuntimeError("Controller not started")
        # Drain any in-flight items first.
        while self.queue.pending:
            time.sleep(0.05)
        with self._gpu_lock:
            manifest = restore_snapshot(self.state, self.adapter_mgr, self.work_dir / name)
        # Phase 6 — restored adapter must be reflected in the sidecar too.
        if self.weight_sync is not None:
            from lile.adapters import lora_state_dict  # noqa: PLC0415
            try:
                self.weight_sync.push_active_lora(lora_state_dict(self.state.model))
            except Exception:
                if self.trajectory is not None:
                    self.trajectory.append(
                        "warn", event="weight-sync-push-failed",
                        phase="post-restore",
                    )
        if self.trajectory is not None:
            self.trajectory.append("restore", name=name, manifest=manifest)
        return manifest

    # --- Compute worker ----------------------------------------------------

    def _handle_compute(self, payload: tuple[str, Any]) -> Any:
        kind, body = payload
        if kind == "train":
            return self._do_train(body)
        if kind == "merge":
            return self._do_merge()
        if kind == "snapshot":
            return self._do_snapshot(body)
        raise ValueError(f"Unknown compute payload kind: {kind!r}")

    def _do_train(self, batch: Batch) -> StepResult:
        assert self.train_engine is not None and self.state is not None
        with self._gpu_lock:
            self.state.set_training_mode()
            result = self.train_engine.step(batch, ref_model=self._ref_model)
        if self.trajectory is not None:
            self.trajectory.append(
                "train",
                phase="completed",
                step=self.train_engine.global_step,
                loss=result.loss,
                components=result.components,
                grad_norm=result.grad_norm,
                elapsed_s=result.elapsed_s,
                vram_peak_gb=result.vram_peak_gb,
                n_samples=result.n_samples,
                skipped=result.skipped_samples,
                notes=result.notes,
            )
        return result

    def _do_merge(self) -> dict[str, Any]:
        assert self.adapter_mgr is not None and self.state is not None
        with self._gpu_lock:
            summary = self.adapter_mgr.merge_active_lora()
            self.state.merge_count += 1
        # Phase 6 — push the freshly reset LoRA + merged-delta state to the
        # sidecar so its next generate() reflects the merge. We do this
        # OUTSIDE the GPU lock: the sidecar lives on its own device (separate)
        # or in its own CUDA stream (colocate), and the trainer's adapter
        # tensors are already a stable snapshot.
        if self.weight_sync is not None:
            from lile.adapters import lora_state_dict  # noqa: PLC0415
            try:
                self.weight_sync.push_active_lora(lora_state_dict(self.state.model))
            except Exception:
                # Sidecar push failures must not crash the trainer — sidecar
                # will serve stale weights until the next push.
                if self.trajectory is not None:
                    self.trajectory.append(
                        "warn", event="weight-sync-push-failed",
                        phase="post-merge",
                    )
        if self.trajectory is not None:
            self.trajectory.append(
                "merge",
                phase="completed",
                merge_count=self.state.merge_count,
                n_layers=len(summary),
                summary=summary,
            )
        return {"merge_count": self.state.merge_count, "n_layers": len(summary)}

    def _do_snapshot(self, name: str) -> Path:
        assert self.state is not None and self.adapter_mgr is not None and self.trajectory is not None
        with self._gpu_lock:
            offset = self.trajectory.current_offset()
            committed = self.queue.committed_seq if self.queue is not None else None
            out_dir = save_snapshot(
                self.state,
                self.adapter_mgr,
                self.work_dir / name,
                trajectory_offset=offset,
                committed_seq=committed,
            )
        self.trajectory.append("snapshot", phase="completed", name=name, path=str(out_dir))
        return out_dir

    # --- Feedback routing ---------------------------------------------------

    @staticmethod
    def feedback_to_batch(
        kind: str,
        *,
        prompt: str,
        bad_response: str,
        critique: str | None = None,
        better_response: str | None = None,
        value: str | None = None,
        weight_scale: float = 1.0,
    ) -> Batch:
        """Build the training :class:`Batch` for a feedback record.

        Pure function (no controller state) so the idle replay scheduler can
        reconstruct batches from trajectory records without re-implementing the
        routing table. ``weight_scale`` lets the replay scheduler downweight
        replayed samples vs. live ones.
        """
        if kind == "binary":
            label = "desirable" if value in ("up", "good", "thumbs_up") else "undesirable"
            sample = Sample(
                prompt=prompt,
                response=bad_response,
                label=label,
                objectives=[{"kto": {}}],
                weight=1.0 * weight_scale,
            )
        elif kind == "rewrite":
            if not better_response:
                raise ValueError("rewrite feedback requires better_response")
            sample = Sample(
                prompt=prompt, target=better_response, weight=3.0 * weight_scale,
                objectives=[{"sft": {}}],
            )
        elif kind == "nl_critique":
            if not critique:
                raise ValueError("nl_critique feedback requires critique")
            sample = Sample(
                prompt=prompt, response=bad_response, critique=critique,
                rejected=bad_response,
                objectives=[{"ccpd": {}}],
                weight=1.0 * weight_scale,
            )
        elif kind in ("nl_critique_with_rewrite", "preferred"):
            sample = Sample(
                prompt=prompt,
                response=bad_response,
                critique=critique,
                target=better_response,
                rejected=bad_response,
                objectives=[{"ccpd": {}}],
                weight=1.0 * weight_scale,
            )
        elif kind == "rejection":
            # T1.4: no critique, no rewrite — fall back to "sample, judge, SFT
            # on the best". Useful when the only signal is "the previous
            # answer was bad, please try again". The judge is configured
            # globally via lile.objectives.rejection_sft.set_default_judge.
            sample = Sample(
                prompt=prompt,
                response=bad_response,
                objectives=[{"rejection_sft": {}}],
                weight=1.0 * weight_scale,
            )
        else:
            raise ValueError(f"unknown feedback kind {kind!r}")
        return Batch(samples=[sample])

    def submit_feedback(
        self,
        response_id: str,
        kind: str,
        *,
        critique: str | None = None,
        better_response: str | None = None,
        value: str | None = None,
    ) -> CommitToken:
        """Route feedback to the appropriate objective per §5b.4 / §5c.16.

        kind ∈ {"binary", "rewrite", "nl_critique", "nl_critique_with_rewrite", "preferred"}
        """
        if response_id not in self._response_cache:
            raise KeyError(f"unknown response_id {response_id!r}")
        ctx = self._response_cache[response_id]
        prompt = ctx["messages"][-1]["content"]
        bad_response = ctx["response"]

        batch = self.feedback_to_batch(
            kind,
            prompt=prompt,
            bad_response=bad_response,
            critique=critique,
            better_response=better_response,
            value=value,
        )

        if self.trajectory is not None:
            # Persist enough context to rebuild the Batch from disk so the
            # idle replay scheduler doesn't depend on _response_cache (which
            # is in-memory-only and lost across restarts).
            self.trajectory.append(
                "feedback", response_id=response_id, feedback_kind=kind,
                prompt=prompt,
                response=bad_response,
                critique=critique,
                better_response=better_response,
                value=value,
                has_critique=critique is not None,
                has_rewrite=better_response is not None,
            )
        return self.submit_train(batch)
