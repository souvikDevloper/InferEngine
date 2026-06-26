import json

import httpx
import pytest
from fastapi.responses import StreamingResponse

from inferengine.api.main import app, scheduler


@pytest.mark.asyncio
async def test_streaming_completions_match_vllm_benchmark_contract():
    await scheduler.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "verify streaming",
                    "max_tokens": 4,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
        assert response.status_code == 200
        events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        payloads = [json.loads(event) for event in events[:-1]]
        token_chunks = [payload for payload in payloads if payload["choices"]]
        assert len(token_chunks) == 4
        assert all("text" in payload["choices"][0] for payload in token_chunks)
        assert payloads[-1]["usage"]["completion_tokens"] == 4
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_openai_completions_can_use_paged_backend_proxy():
    class FakePagedBackend:
        openai_compatible = True
        name = "fake-paged"
        body = b""

        async def completions_raw(self, body):
            self.body = body

            async def events():
                yield b'data: {"choices":[{"text":"x","finish_reason":null}]}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

    original = scheduler.model
    scheduler.model = FakePagedBackend()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "verify paged proxy",
                    "max_tokens": 1,
                    "stream": True,
                },
            )
        assert response.status_code == 200
        assert 'data: {"choices":[{"text":"x","finish_reason":null}]}' in response.text
        assert "data: [DONE]" in response.text
        assert json.loads(scheduler.model.body)["prompt"] == "verify paged proxy"
    finally:
        scheduler.model = original
