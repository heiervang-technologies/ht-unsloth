"""FastAPI server — OpenAI-compatible chat completions plus /v1/train,
/v1/feedback, and /v1/state/* control-plane routes.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ServeConfig
from .controller import Controller

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- pydantic
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 256
    temperature: float = 0.7
    top_p: float = 0.95
    after_commit_token: int | None = Field(
        default=None,
        description="If provided, block until this training commit_token is reflected.",
    )


class TrainSample(BaseModel):
    # Open-shaped; per-objective semantics.
    prompt: str | None = None
    response: str | None = None
    label: str | None = None
    weight: float | None = None
    chosen: str | None = None
    rejected: str | None = None
    bad: str | None = None
    good: str | None = None
    critique: str | None = None
    preferred: str | None = None
    aux_candidates: list[str] | None = None

    class Config:
        extra = "allow"


class TrainRequest(BaseModel):
    objective: str
    samples: list[dict[str, Any]] = Field(default_factory=list)
    batch_objectives: list[dict[str, Any]] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = 2


class FeedbackRequest(BaseModel):
    response_id: str | None = None
    kind: str  # "binary" | "rewrite" | "preferred" | "nl_critique" | "nl_critique_with_rewrite"

    class Config:
        extra = "allow"


class SnapshotRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------- app
def create_app(cfg: ServeConfig | None = None) -> FastAPI:
    cfg = cfg or ServeConfig()
    app = FastAPI(title="lile", version="0.1.0-dev")
    app.state.cfg = cfg
    app.state.controller = Controller(cfg)

    @app.on_event("startup")
    async def _startup() -> None:
        await app.state.controller.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.controller.stop()

    # --------------------------------------------------------------- health
    @app.get("/health")
    async def health() -> dict[str, Any]:
        c = app.state.controller
        return {
            "ok": True,
            "model": cfg.model,
            "queue_depth": c.queue._q.qsize(),
            "commit_cursor": c.queue.committed,
            "merges": c.state.merges_applied if c.state else 0,
        }

    # --------------------------------------------------------------- chat
    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> dict[str, Any]:
        c: Controller = app.state.controller
        t0 = time.time()
        result = await c.generate(
            [m.model_dump() for m in req.messages],
            max_new_tokens=req.max_tokens or 256,
            temperature=req.temperature,
            top_p=req.top_p,
            after_commit_token=req.after_commit_token,
        )
        latency = time.time() - t0
        return {
            "id": result["response_id"],
            "object": "chat.completion",
            "model": cfg.model,
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": result["response"]},
            }],
            "lile": {"latency_s": latency,
                     "commit_cursor": c.queue.committed,
                     "response_id": result["response_id"]},
        }

    # --------------------------------------------------------------- train
    @app.post("/v1/train")
    async def train(req: TrainRequest) -> dict[str, Any]:
        c: Controller = app.state.controller
        spec = req.model_dump()
        return await c.submit_train(spec)

    # --------------------------------------------------------------- feedback
    @app.post("/v1/feedback")
    async def feedback(req: FeedbackRequest) -> dict[str, Any]:
        c: Controller = app.state.controller
        payload = req.model_dump()
        return await c.submit_feedback(payload)

    # --------------------------------------------------------------- state ops
    @app.post("/v1/state/merge")
    async def state_merge() -> dict[str, Any]:
        return await app.state.controller.request_merge()

    @app.post("/v1/state/snapshot/save")
    async def state_save(req: SnapshotRequest) -> dict[str, Any]:
        return await app.state.controller.request_snapshot_save(req.name)

    @app.post("/v1/state/snapshot/load")
    async def state_load(req: SnapshotRequest) -> dict[str, Any]:
        return await app.state.controller.request_snapshot_load(req.name)

    @app.get("/v1/state/snapshots")
    async def state_list() -> dict[str, Any]:
        return {"snapshots": app.state.controller.snapshots.list()}

    @app.get("/v1/state/trajectory/tail")
    async def traj_tail(n: int = 20) -> dict[str, Any]:
        return {"events": app.state.controller.trajectory.tail(n)}

    # --------------------------------------------------------------- block-for-commit helper
    @app.post("/v1/wait")
    async def wait(token: int, timeout: float = 60.0) -> dict[str, Any]:
        c: Controller = app.state.controller
        try:
            task = await c.queue.wait_for(int(token), timeout=timeout)
            return {"committed": True, "token": task.token, "kind": task.kind}
        except asyncio.TimeoutError:
            return {"committed": False, "reason": "timeout"}
        except KeyError:
            raise HTTPException(404, detail=f"unknown token {token}")

    return app


def serve(cfg: ServeConfig | None = None) -> None:
    app = create_app(cfg or ServeConfig())
    uvicorn.run(app, host=app.state.cfg.host, port=app.state.cfg.port, log_level="info")


if __name__ == "__main__":
    serve()
