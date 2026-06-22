from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inferengine.api.schemas import CompletionRequest, GenerateRequest, GenerateResponse, HealthResponse
from inferengine.core.config import EngineConfig
from inferengine.core.scheduler import ContinuousBatchScheduler
from inferengine.core.tokenizer import tokenize

scheduler = ContinuousBatchScheduler(EngineConfig())


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()

    app = FastAPI(
        title="InferEngine",
        description="Continuous batching inference server with paged KV-cache accounting",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        try:
            result = await scheduler.submit(req.prompt, req.max_new_tokens)
            return GenerateResponse(**result.__dict__)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MemoryError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": scheduler.model.name, "object": "model", "owned_by": "inferengine"}],
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if not req.stream:
            result = await scheduler.submit(req.prompt, req.max_tokens)
            return {
                "id": f"cmpl-{result.request_id}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "text": result.text, "finish_reason": result.finish_reason}],
                "usage": {
                    "prompt_tokens": len(tokenize(req.prompt)),
                    "completion_tokens": result.generated_tokens,
                    "total_tokens": len(tokenize(req.prompt)) + result.generated_tokens,
                },
            }

        request_id = f"cmpl-{uuid.uuid4()}"
        prompt_tokens = len(tokenize(req.prompt))

        async def events():
            generated = 0
            async for token in scheduler.stream(req.prompt, req.max_tokens):
                generated += 1
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "text": token, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
            if req.stream_options and req.stream_options.get("include_usage"):
                usage = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": generated,
                        "total_tokens": prompt_tokens + generated,
                    },
                }
                yield f"data: {json.dumps(usage, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/stats")
    async def stats() -> dict:
        return scheduler.stats()

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
