from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class CompletionRequest(BaseModel):
    """OpenAI completions subset consumed by ``vllm bench serve``."""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=1, le=512)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    ignore_eos: bool = False
