"""FastAPI server — OpenAI-compatible chat completions plus /v1/train,
/v1/feedback, and /v1/state/* control-plane routes.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import pathlib
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from . import metrics as metrics_mod
from .config import ServeConfig
from .controller import Controller
from .errors import NotFoundError
from .metrics import MetricsMiddleware
from .middleware import RequestIDMiddleware, current_request_id
from .server_errors import register_error_handlers

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- pydantic
class ChatMessage(BaseModel):
    role: str
    content: str
    # Optional — lets clients re-send an earlier turn's reasoning so the
    # chat template can thread it back in (Qwen3 re-injects reasoning_content
    # for the latest assistant turn only; see its template).
    reasoning_content: str | None = None

    class Config:
        extra = "allow"


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    after_commit_token: int | None = Field(
        default=None,
        description="If provided, block until this training commit_token is reflected.",
    )
    # Reasoning controls.  ``enable_thinking`` is forwarded to
    # ``apply_chat_template`` when the tokenizer supports the kwarg
    # (Qwen3 family).  ``parse_reasoning=False`` disables the parser even
    # when thinking is on (raw tags stay in ``content``).
    enable_thinking: bool | None = None
    parse_reasoning: bool = True


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

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup — Controller was constructed below (pre-lifespan) so routes
        # can close over it without waiting for this hook.
        await app.state.controller.start()
        try:
            yield
        finally:
            # Prefer the graceful path so pending /v1/wait callers get
            # ShutdownDroppedError envelopes instead of hanging on their own
            # 60s timeout (see issue #11).
            await app.state.controller.graceful_shutdown(
                deadline_s=cfg.shutdown_deadline_s,
                hard_stop_grace_s=cfg.shutdown_hard_stop_grace_s,
            )

    app = FastAPI(title="lile", version="0.1.0-dev", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.controller = Controller(cfg)
    metrics_mod.bind_controller(app.state.controller)
    # Middleware order: Starlette runs the outermost `add_middleware` last on
    # the way in, first on the way out. We want MetricsMiddleware to see the
    # final response status, so it goes outside RequestIDMiddleware.
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    # --------------------------------------------------------------- metrics
    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        body, content_type = metrics_mod.render_negotiated(
            request.headers.get("accept"),
        )
        return Response(body, media_type=content_type)

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
            "commit_sse_subscribers": c.commits.subscriber_count,
            "commit_sse_drops": c.commits.drops,
        }

    # --------------------------------------------------------------- chat
    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        c: Controller = app.state.controller
        t0 = time.time()
        messages = [m.model_dump() for m in req.messages]

        if req.stream:
            async def sse():
                ttft_observed = False
                try:
                    async for ev in c.stream_generate(
                        messages,
                        max_new_tokens=req.max_tokens or 256,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        after_commit_token=req.after_commit_token,
                        enable_thinking=req.enable_thinking,
                        parse_reasoning=req.parse_reasoning,
                    ):
                        if "error" in ev:
                            rid = current_request_id() or ""
                            payload = {
                                "object": "chat.completion.chunk",
                                "model": cfg.model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                                "lile": {
                                    "error": {
                                        "code": "internal",
                                        "message": str(ev["error"]),
                                        "retryable": False,
                                        "request_id": rid,
                                    },
                                    "response_id": ev["response_id"],
                                },
                            }
                            yield f"data: {json.dumps(payload)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        if ev.get("final"):
                            payload = {
                                "id": ev["response_id"],
                                "object": "chat.completion.chunk",
                                "model": cfg.model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                "lile": {"latency_s": time.time() - t0,
                                         "commit_cursor": ev["commit_cursor"],
                                         "response_id": ev["response_id"]},
                            }
                            yield f"data: {json.dumps(payload)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        # delta event — emit whichever channel(s) had bytes.
                        delta_obj: dict[str, Any] = {"role": "assistant"}
                        if ev.get("delta"):
                            delta_obj["content"] = ev["delta"]
                        if ev.get("reasoning_delta"):
                            delta_obj["reasoning_content"] = ev["reasoning_delta"]
                        if len(delta_obj) == 1:
                            # Only role — nothing useful to emit.
                            continue
                        if not ttft_observed:
                            metrics_mod.record_generate_latency(
                                stream=True, latency_s=time.time() - t0,
                            )
                            ttft_observed = True
                        payload = {
                            "id": ev["response_id"],
                            "object": "chat.completion.chunk",
                            "model": cfg.model,
                            "choices": [{"index": 0, "delta": delta_obj}],
                            "lile": {"response_id": ev["response_id"]},
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                except Exception as exc:
                    rid = current_request_id() or ""
                    err_payload = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                        "lile": {
                            "error": {
                                "code": "internal",
                                "message": f"{type(exc).__name__}: {exc}",
                                "retryable": False,
                                "request_id": rid,
                            },
                        },
                    }
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # non-streaming path
        result = await c.generate(
            messages,
            max_new_tokens=req.max_tokens or 256,
            temperature=req.temperature,
            top_p=req.top_p,
            after_commit_token=req.after_commit_token,
            enable_thinking=req.enable_thinking,
            parse_reasoning=req.parse_reasoning,
        )
        latency = time.time() - t0
        metrics_mod.record_generate_latency(stream=False, latency_s=latency)
        message: dict[str, Any] = {"role": "assistant",
                                   "content": result["response"]}
        if result.get("reasoning_content"):
            message["reasoning_content"] = result["reasoning_content"]
        return {
            "id": result["response_id"],
            "object": "chat.completion",
            "model": cfg.model,
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": message,
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
    async def traj_tail(n: int = 20,
                        since_offset: int | None = None) -> dict[str, Any]:
        traj = app.state.controller.trajectory
        if since_offset is None:
            return {"events": traj.tail(n)}
        return traj.tail_structured(n=n, since_offset=since_offset)

    # --------------------------------------------------------------- commits SSE
    @app.get("/v1/commits/stream")
    async def commits_stream():
        """Per-commit event stream; see docs/research/pr-specs/commits-sse-stream.md.

        One ``event: commit`` per successful train-task cursor advance.
        ``: keepalive`` comment line every 15s when idle. Terminal
        ``event: shutdown`` on daemon stop, then stream closes.
        """
        c: Controller = app.state.controller
        if not cfg.commits_sse_enabled:
            # Return an empty stream that closes immediately rather than 404 —
            # callers can tell the feature is off but the route itself is
            # stable.
            async def _disabled():
                yield "event: shutdown\ndata: {\"reason\":\"disabled\"}\n\n"
            return StreamingResponse(_disabled(), media_type="text/event-stream")

        sub = c.commits.subscribe()

        async def gen():
            keepalive_interval = 15.0
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            sub.get(), timeout=keepalive_interval,
                        )
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if event.get("_shutdown"):
                        yield (
                            "event: shutdown\n"
                            f"data: {json.dumps({'reason': 'daemon_stop'})}\n\n"
                        )
                        return
                    yield f"event: commit\ndata: {json.dumps(event)}\n\n"
            finally:
                c.commits.unsubscribe(sub)

        return StreamingResponse(gen(), media_type="text/event-stream")

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
            raise NotFoundError(f"unknown commit token {token}")

    return app


def serve(cfg: ServeConfig | None = None) -> None:
    app = create_app(cfg or ServeConfig())
    uvicorn.run(app, host=app.state.cfg.host, port=app.state.cfg.port, log_level="info")


def _parse_cli_args(argv: list[str] | None = None) -> ServeConfig:
    """Parse argv into a ServeConfig, overriding only fields the user set."""
    cfg = ServeConfig()
    p = argparse.ArgumentParser(prog="lile.server", description=__doc__)
    p.add_argument("--host", default=cfg.host)
    p.add_argument("--port", type=int, default=cfg.port)
    p.add_argument("--model", default=cfg.model)
    p.add_argument("--data-dir", type=pathlib.Path, default=cfg.data_dir)
    p.add_argument("--max-seq-length", type=int, default=cfg.max_seq_length)
    p.add_argument("--lora-rank", type=int, default=cfg.lora_rank)
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   default=cfg.load_in_4bit)
    p.add_argument("--idle-replay", dest="idle_replay", action="store_true",
                   default=cfg.idle_replay)
    p.add_argument("--frozen-ref", dest="frozen_ref", action="store_true",
                   default=cfg.frozen_ref)
    args = p.parse_args(argv)
    cfg.host = args.host
    cfg.port = args.port
    cfg.model = args.model
    cfg.data_dir = args.data_dir
    cfg.max_seq_length = args.max_seq_length
    cfg.lora_rank = args.lora_rank
    cfg.load_in_4bit = args.load_in_4bit
    cfg.idle_replay = args.idle_replay
    cfg.frozen_ref = args.frozen_ref
    return cfg


if __name__ == "__main__":
    serve(_parse_cli_args())
