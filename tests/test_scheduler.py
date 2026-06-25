import pytest

from inferengine.core.config import EngineConfig
from inferengine.core.scheduler import ContinuousBatchScheduler


@pytest.mark.asyncio
async def test_scheduler_generates_text():
    s = ContinuousBatchScheduler(EngineConfig(max_batch_size=4, max_pages=64, decode_interval_ms=1))
    await s.start()
    try:
        res = await s.submit("hello batching", 5)
        assert res.generated_tokens == 5
        assert res.text
        assert s.stats()["completed_requests"] == 1
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_scheduler_backend_failure_unblocks_requests():
    class FailingBackend:
        name = "failing"

        def encode(self, prompt: str) -> list[str]:
            return prompt.split()

        def next_tokens(self, states):
            raise RuntimeError("decode failed")

        def detokenize(self, tokens):
            return "".join(tokens)

        def release(self, state) -> None:
            state.backend_state = None

    s = ContinuousBatchScheduler(EngineConfig(max_batch_size=4, max_pages=64, decode_interval_ms=1))
    s.model = FailingBackend()
    await s.start()
    try:
        with pytest.raises(RuntimeError, match="decode failed"):
            await __import__("asyncio").wait_for(s.submit("will fail", 4), timeout=1)
        assert "decode failed" in s.stats()["last_error"]
        assert s.stats()["active_requests"] == 0
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_scheduler_bulk_generate_completes_batched_requests():
    class BulkBackend:
        name = "bulk"

        def encode(self, prompt: str) -> list[str]:
            return prompt.split()

        def next_tokens(self, states):
            raise AssertionError("bulk mode should not use per-token decode")

        def complete_batch(self, states):
            return [["x"] * state.max_new_tokens for state in states]

        def detokenize(self, tokens):
            return "".join(tokens)

        def release(self, state) -> None:
            state.backend_state = None

    s = ContinuousBatchScheduler(EngineConfig(max_batch_size=4, max_pages=64, decode_interval_ms=1, bulk_generate=True))
    s.model = BulkBackend()
    await s.start()
    try:
        results = await __import__("asyncio").gather(*(s.submit(f"bulk {i}", 3) for i in range(4)))
        assert [result.generated_tokens for result in results] == [3, 3, 3, 3]
        assert s.stats()["decode_steps"] == 1
        assert s.stats()["completed_requests"] == 4
    finally:
        await s.stop()


@pytest.mark.asyncio
async def test_scheduler_batches_concurrent_requests():
    s = ContinuousBatchScheduler(EngineConfig(max_batch_size=8, max_pages=256, decode_interval_ms=1))
    await s.start()
    try:
        results = await __import__("asyncio").gather(*(s.submit(f"prompt {i}", 8) for i in range(16)))
        assert len(results) == 16
        stats = s.stats()
        assert stats["completed_requests"] == 16
        assert stats["max_batch_observed"] > 1
    finally:
        await s.stop()
