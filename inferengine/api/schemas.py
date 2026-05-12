from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=64, ge=1, le=512)


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    generated_tokens: int
    latency_ms: float
    finish_reason: str
    model: str


class HealthResponse(BaseModel):
    status: str
