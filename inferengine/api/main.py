from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inferengine.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from inferengine.core.config import EngineConfig
from inferengine.core.scheduler import ContinuousBatchScheduler

scheduler = ContinuousBatchScheduler(EngineConfig())


def create_app() -> FastAPI:
    app = FastAPI(
        title="InferEngine",
        description="Continuous batching inference server with paged KV-cache accounting",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup() -> None:
        await scheduler.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await scheduler.stop()

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

    @app.get("/stats")
    async def stats() -> dict:
        return scheduler.stats()

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
