"""FastAPI app — the daemon's HTTP surface.

Endpoints (all rooted at ``/v1`` for parity with the OpenAI Chat Completions
contract that downstream tools already speak):

* ``POST /v1/chat/completions`` — OpenAI-compatible chat. Returns the standard
  ``chat.completion`` envelope. Two ``lile``-specific hooks:

    - Request header ``x-lile-min-commit: <int>`` (or JSON field
      ``min_commit_seq``) — block this generation until ``committed_seq`` ≥ value.
      This is how a client says "see the training batch I just posted."
    - Response header ``x-lile-response-id`` — UUID4 string used to target this
      specific generation in subsequent ``/v1/feedback`` calls.

* ``POST /v1/train`` — submit a :class:`Batch` payload. Returns
  ``{commit_token: int}``. The work is queued; the cursor advances when the step
  has actually mutated the model.

* ``POST /v1/feedback`` — submit feedback against a prior ``response_id``. The
  controller routes to the right objective per §5b.4: binary→KTO, rewrite→SFT,
  critique→CCPD v2, critique+rewrite→CCPD v2 with the user rewrite seeded at
  rank 1.

* ``POST /v1/state/save`` / ``/v1/state/restore`` — snapshot lifecycle.
  ``save`` returns a commit token; ``restore`` is synchronous and drains the
  queue first.

* ``POST /v1/state/merge`` — trigger a progressive merge (§6). Returns commit
  token.

* ``GET /v1/state`` — VRAM, committed_seq, merge_count, registered objectives.

* ``GET /v1/objectives`` — list of registered objectives.

* ``GET /v1/health`` — alive ping (no model touch).

Per DESIGN.md §10 we deliberately do not implement: streaming responses, auth,
TLS, multi-tenant routing, function calling.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lile import objectives as O
from lile.controller import Controller
from lile.objectives import Batch, Sample
from lile.queue import CommitToken


# --- Request / response models ---------------------------------------------


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessageIn]
    max_tokens: int | None = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    # OpenAI uses ``stream`` — we accept it for compat but always ignore it.
    stream: bool = False
    # lile-specific extension (also configurable via header).
    min_commit_seq: int | None = None


class TrainRequest(BaseModel):
    samples: list[dict[str, Any]]
    batch_objectives: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    response_id: str
    kind: str  # binary, rewrite, nl_critique, nl_critique_with_rewrite, preferred
    value: str | None = None         # binary: up/down
    critique: str | None = None      # nl_critique[_with_rewrite]
    better_response: str | None = None  # rewrite, nl_critique_with_rewrite


class SnapshotRequest(BaseModel):
    name: str


# --- App factory ------------------------------------------------------------


def create_app(controller: Controller) -> FastAPI:
    """Build a FastAPI app bound to a *running* :class:`Controller`."""
    if controller.queue is None:
        raise RuntimeError("Controller must be started() before binding the app")

    app = FastAPI(title="lile", version="0.1.0")

    # --- /v1/health ---------------------------------------------------------
    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": "0.1.0",
            "committed_seq": controller.queue.committed_seq if controller.queue else 0,
            "ts": time.time(),
        }

    # --- /v1/state ----------------------------------------------------------
    @app.get("/v1/state")
    def state() -> dict[str, Any]:
        assert controller.state is not None and controller.queue is not None
        train_step = (
            controller.train_engine.global_step
            if controller.train_engine is not None else 0
        )
        return {
            "model_name": controller.config.state.model_name,
            "committed_seq": controller.queue.committed_seq,
            "pending": controller.queue.pending,
            "merge_count": controller.state.merge_count,
            "train_step": train_step,
            "vram": controller.state.vram_summary(),
            "lr": controller.train_engine.lr if controller.train_engine else None,
            "objectives": O.list_objectives(),
        }

    @app.get("/v1/objectives")
    def objectives() -> dict[str, Any]:
        return {
            "objectives": [
                {
                    "name": name,
                    "per_sample": O.get(name).per_sample,
                    "requires": list(O.get(name).requires),
                    "description": O.get(name).description,
                }
                for name in O.list_objectives()
            ]
        }

    # --- /v1/chat/completions -----------------------------------------------
    @app.post("/v1/chat/completions")
    def chat_completions(
        req: ChatCompletionRequest,
        x_lile_min_commit: int | None = Header(default=None, alias="x-lile-min-commit"),
    ) -> JSONResponse:
        # Resolve the effective min_commit (header overrides body).
        min_commit = x_lile_min_commit if x_lile_min_commit is not None else req.min_commit_seq
        wait_for = CommitToken(seq=int(min_commit)) if min_commit and min_commit > 0 else None

        try:
            result = controller.chat(
                [m.model_dump() for m in req.messages],
                max_new_tokens=req.max_tokens or 256,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.temperature > 0,
                wait_for=wait_for,
            )
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        committed = controller.queue.committed_seq if controller.queue else 0
        body = {
            "id": f"chatcmpl-{result.response_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or controller.config.state.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            # lile extensions, namespaced so OpenAI clients ignore them.
            "lile": {
                "response_id": result.response_id,
                "committed_seq": committed,
                "elapsed_s": result.elapsed_s,
            },
        }
        return JSONResponse(
            content=body,
            headers={
                "x-lile-response-id": result.response_id,
                "x-lile-committed-seq": str(committed),
            },
        )

    # --- /v1/train ----------------------------------------------------------
    @app.post("/v1/train")
    def train(req: TrainRequest) -> dict[str, Any]:
        if not req.samples:
            raise HTTPException(status_code=400, detail="samples must be non-empty")
        try:
            samples = [_sample_from_dict(s) for s in req.samples]
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"invalid sample: {e}")
        batch = Batch(samples=samples, batch_objectives=list(req.batch_objectives))
        token = controller.submit_train(batch)
        return {
            "commit_token": token.seq,
            "queued_seq": token.seq,
            "n_samples": len(samples),
        }

    # --- /v1/feedback -------------------------------------------------------
    @app.post("/v1/feedback")
    def feedback(req: FeedbackRequest) -> dict[str, Any]:
        try:
            token = controller.submit_feedback(
                req.response_id,
                req.kind,
                value=req.value,
                critique=req.critique,
                better_response=req.better_response,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"commit_token": token.seq, "queued_seq": token.seq, "kind": req.kind}

    # --- /v1/state/save | restore | merge -----------------------------------
    @app.post("/v1/state/save")
    def state_save(req: SnapshotRequest) -> dict[str, Any]:
        token = controller.submit_snapshot(req.name)
        return {"commit_token": token.seq, "name": req.name}

    @app.post("/v1/state/restore")
    def state_restore(req: SnapshotRequest) -> dict[str, Any]:
        try:
            manifest = controller.restore(req.name)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"name": req.name, "manifest": manifest}

    @app.post("/v1/state/merge")
    def state_merge() -> dict[str, Any]:
        token = controller.submit_merge()
        return {"commit_token": token.seq}

    # --- /v1/queue/wait -----------------------------------------------------
    @app.post("/v1/queue/wait")
    def queue_wait(commit_token: int = Body(..., embed=True), timeout: float = Body(60.0, embed=True)) -> dict[str, Any]:
        if controller.queue is None:
            raise HTTPException(status_code=503, detail="controller not started")
        ok = controller.queue.wait_for_commit(CommitToken(seq=commit_token), timeout=timeout)
        if not ok:
            raise HTTPException(status_code=504, detail=f"timed out waiting for commit {commit_token}")
        return {"committed_seq": controller.queue.committed_seq, "ok": True}

    return app


# --- helpers ----------------------------------------------------------------


_SAMPLE_FIELDS = {
    "prompt", "target", "rejected", "label", "critique", "response", "weight", "objectives",
}


def _sample_from_dict(d: dict[str, Any]) -> Sample:
    """Build a :class:`Sample` from a JSON dict, rejecting unknown keys."""
    if "prompt" not in d:
        raise KeyError("prompt is required")
    bad = set(d) - _SAMPLE_FIELDS
    if bad:
        raise ValueError(f"unknown sample fields: {sorted(bad)}")
    kwargs = {k: d[k] for k in _SAMPLE_FIELDS if k in d}
    if "weight" in kwargs:
        kwargs["weight"] = float(kwargs["weight"])
    return Sample(**kwargs)
