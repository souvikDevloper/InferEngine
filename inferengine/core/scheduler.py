from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any

from inferengine.core.cache import KVCacheManager
from inferengine.core.config import EngineConfig
from inferengine.core.tokenizer import detokenize, tokenize
from inferengine.metrics import prom
from inferengine.model.toy_decoder import TorchToyDecoder


@dataclass
class GenerationResult:
    request_id: str
    text: str
    generated_tokens: int
    latency_ms: float
    finish_reason: str
    model: str


@dataclass
class RequestState:
    prompt: str
    max_new_tokens: int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_tokens: list[str] = field(default_factory=list)
    generated: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    admitted_at: float | None = None
    done: asyncio.Future | None = None
    stream: asyncio.Queue[str | None] | None = None

    @property
    def is_complete(self) -> bool:
        return len(self.generated) >= self.max_new_tokens


class ContinuousBatchScheduler:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.cache = KVCacheManager(
            max_pages=self.config.max_pages,
            page_size=self.config.page_size,
            policy=self.config.eviction_policy,
        )
        self.model = TorchToyDecoder()
        self.waiting: asyncio.Queue[RequestState] = asyncio.Queue(maxsize=self.config.max_waiting)
        self.active: dict[str, RequestState] = {}
        self._task: asyncio.Task | None = None
        self._running = False
        self._steps = 0
        self._completed = 0
        self._total_batch_size = 0
        self._max_batch_observed = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            await asyncio.wait([self._task], timeout=1)

    async def submit(self, prompt: str, max_new_tokens: int | None = None) -> GenerationResult:
        state = await self._enqueue(prompt, max_new_tokens, streaming=False)
        return await state.done

    async def stream(self, prompt: str, max_new_tokens: int | None = None) -> AsyncIterator[str]:
        state = await self._enqueue(prompt, max_new_tokens, streaming=True)
        assert state.stream is not None
        while True:
            token = await state.stream.get()
            if token is None:
                break
            yield token
        # Propagate cache admission and scheduler failures after ending the stream.
        await state.done

    async def _enqueue(self, prompt: str, max_new_tokens: int | None, streaming: bool) -> RequestState:
        max_new = max_new_tokens or self.config.max_new_tokens_default
        max_new = min(max_new, self.config.max_new_tokens_limit)
        prompt_tokens = tokenize(prompt)
        if len(prompt_tokens) + max_new > self.config.admission_token_limit:
            prom.REQUESTS_TOTAL.labels(status="rejected").inc()
            raise ValueError("request exceeds admission token limit")

        loop = asyncio.get_running_loop()
        state = RequestState(
            prompt=prompt,
            max_new_tokens=max_new,
            prompt_tokens=prompt_tokens,
            done=loop.create_future(),
            stream=asyncio.Queue() if streaming else None,
        )
        await self.waiting.put(state)
        prom.WAITING_REQUESTS.set(self.waiting.qsize())
        return state

    async def _loop(self) -> None:
        while self._running:
            await self._admit_waiting()
            if not self.active:
                await asyncio.sleep(self.config.decode_interval_ms / 1000)
                continue
            await self._decode_step()
            await asyncio.sleep(self.config.decode_interval_ms / 1000)

    async def _admit_waiting(self) -> None:
        async with self._lock:
            while len(self.active) < self.config.max_batch_size and not self.waiting.empty():
                state = await self.waiting.get()
                try:
                    self.cache.allocate(state.request_id, len(state.prompt_tokens), state.max_new_tokens)
                    state.admitted_at = time.time()
                    self.active[state.request_id] = state
                except MemoryError as exc:
                    if state.done and not state.done.done():
                        state.done.set_exception(exc)
                    if state.stream is not None:
                        state.stream.put_nowait(None)
                    prom.REQUESTS_TOTAL.labels(status="cache_rejected").inc()
            prom.WAITING_REQUESTS.set(self.waiting.qsize())
            prom.ACTIVE_REQUESTS.set(len(self.active))

    async def _decode_step(self) -> None:
        completed: list[str] = []
        batch = list(self.active.values())
        batch_size = len(batch)
        self._steps += 1
        self._total_batch_size += batch_size
        self._max_batch_observed = max(self._max_batch_observed, batch_size)
        prom.BATCH_SIZE.set(batch_size)
        prom.DECODE_STEPS.inc()

        for state in batch:
            out = self.model.next_token(state.prompt, state.generated, len(state.generated))
            state.generated.append(out.token)
            if state.stream is not None:
                state.stream.put_nowait(out.token + " ")
            self.cache.append_token(state.request_id)
            prom.TOKENS_GENERATED.inc()
            if state.is_complete:
                completed.append(state.request_id)

        for rid in completed:
            state = self.active.pop(rid)
            self.cache.complete(rid)
            self.cache.release(rid)
            latency = time.time() - state.created_at
            text = detokenize(state.generated)
            result = GenerationResult(
                request_id=rid,
                text=text,
                generated_tokens=len(state.generated),
                latency_ms=round(latency * 1000, 3),
                finish_reason="length",
                model=self.model.name,
            )
            self._completed += 1
            prom.REQUESTS_TOTAL.labels(status="ok").inc()
            prom.REQUEST_LATENCY.observe(latency)
            if state.done and not state.done.done():
                state.done.set_result(result)
            if state.stream is not None:
                state.stream.put_nowait(None)

        stats = self.cache.stats()
        prom.KV_USED_PAGES.set(float(stats["used_pages"]))
        prom.KV_PRESSURE_EVENTS.set(float(stats["pressure_events"]))
        prom.ACTIVE_REQUESTS.set(len(self.active))

    def stats(self) -> dict[str, Any]:
        cache = self.cache.stats()
        avg_batch = self._total_batch_size / self._steps if self._steps else 0.0
        return {
            "running": self._running,
            "model": self.model.name,
            "waiting_requests": self.waiting.qsize(),
            "active_requests": len(self.active),
            "completed_requests": self._completed,
            "decode_steps": self._steps,
            "average_batch_size": round(avg_batch, 3),
            "max_batch_observed": self._max_batch_observed,
            "cache": cache,
            "config": self.config.__dict__,
        }
