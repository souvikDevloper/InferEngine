from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse


class VLLMPagedBackend:
    """OpenAI-compatible backend backed by vLLM's paged-attention engine.

    This backend is intentionally different from the token-step Transformers
    backend. vLLM owns scheduling, block allocation, KV-cache paging, and CUDA
    attention kernels. InferEngine remains the public OpenAI-compatible surface
    and benchmark target.
    """

    openai_compatible = True

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.host = os.getenv("INFERENGINE_VLLM_HOST", "127.0.0.1")
        self.port = int(os.getenv("INFERENGINE_VLLM_PORT", "8002"))
        self.upstream_url = os.getenv("INFERENGINE_VLLM_UPSTREAM_URL", f"http://{self.host}:{self.port}").rstrip("/")
        self.managed = _env_bool("INFERENGINE_VLLM_MANAGED", True)
        self.startup_timeout_sec = int(os.getenv("INFERENGINE_VLLM_STARTUP_TIMEOUT_SEC", "900"))
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def name(self) -> str:
        return f"inferengine-vllm-paged/{self.model_id}"

    async def start(self) -> None:
        if await self._is_healthy():
            return
        if not self.managed:
            raise RuntimeError(f"vLLM upstream is not healthy at {self.upstream_url}")

        log_path = Path(os.getenv("INFERENGINE_VLLM_LOG", "inferengine-vllm-backend.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab")
        cmd = self._serve_command()
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        await self._wait_until_ready()

    async def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=30)
            except TimeoutError:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    async def health(self) -> dict[str, str]:
        if not await self._is_healthy():
            raise HTTPException(status_code=503, detail="vLLM paged backend is not healthy")
        return {"status": "ok"}

    async def models(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.upstream_url}/v1/models")
        return _checked_json(response)

    async def completions(self, request: Any):
        payload = request.model_dump(exclude_none=True)
        payload["model"] = payload.get("model") or self.model_id
        if payload.get("stream"):
            return StreamingResponse(self._stream_completion(payload), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self.upstream_url}/v1/completions", json=payload)
        return _checked_json(response)

    async def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "stream": False,
            "ignore_eos": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self.upstream_url}/v1/completions", json=payload)
        data = _checked_json(response)
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return {
            "text": choice.get("text", ""),
            "generated_tokens": int(usage.get("completion_tokens") or max_new_tokens),
            "finish_reason": choice.get("finish_reason") or "length",
        }

    def encode(self, prompt: str) -> list[int]:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def next_tokens(self, states):  # pragma: no cover - scheduler bypasses this backend
        raise RuntimeError("vllm_paged backend is OpenAI-compatible and bypasses token-step scheduling")

    def detokenize(self, tokens):  # pragma: no cover - scheduler bypasses this backend
        return "".join(tokens)

    def release(self, state) -> None:  # pragma: no cover - no per-request local KV state
        state.backend_state = None

    def _serve_command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            os.getenv("INFERENGINE_VLLM_DTYPE", "auto"),
            "--max-model-len",
            os.getenv("INFERENGINE_MAX_MODEL_LEN", "4096"),
            "--gpu-memory-utilization",
            os.getenv("INFERENGINE_VLLM_GPU_MEMORY_UTILIZATION", "0.90"),
        ]
        tensor_parallel_size = os.getenv("INFERENGINE_TENSOR_PARALLEL_SIZE", "1")
        if tensor_parallel_size != "1":
            cmd.extend(["--tensor-parallel-size", tensor_parallel_size])
        served_model_name = os.getenv("INFERENGINE_VLLM_SERVED_MODEL_NAME", "")
        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])
        return cmd

    async def _stream_completion(self, payload: dict[str, Any]):
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.upstream_url}/v1/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=body.decode("utf-8", "replace"))
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def _is_healthy(self) -> bool:
        with contextlib.suppress(httpx.HTTPError):
            async with httpx.AsyncClient(timeout=2) as client:
                response = await client.get(f"{self.upstream_url}/health")
            return response.status_code == 200
        return False

    async def _wait_until_ready(self) -> None:
        for _ in range(self.startup_timeout_sec):
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited during startup; see {os.getenv('INFERENGINE_VLLM_LOG')}")
            if await self._is_healthy():
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"timed out waiting for vLLM paged backend at {self.upstream_url}")


def _checked_json(response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
