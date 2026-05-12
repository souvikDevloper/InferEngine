#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int((len(s) - 1) * p))
    return s[idx]


async def one(client: httpx.AsyncClient, i: int, tokens: int) -> float:
    t0 = time.perf_counter()
    prompt = f"request {i}: benchmark continuous batching and kv cache scheduling"
    r = await client.post("/v1/generate", json={"prompt": prompt, "max_new_tokens": tokens})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


async def run(args: argparse.Namespace) -> None:
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency * 2)
    async with httpx.AsyncClient(base_url=args.url, timeout=120, limits=limits) as client:
        start = time.perf_counter()
        latencies: list[float] = []
        sem = asyncio.Semaphore(args.concurrency)

        async def bounded(i: int) -> None:
            async with sem:
                latencies.append(await one(client, i, args.tokens))

        await asyncio.gather(*(bounded(i) for i in range(args.requests)))
        wall = time.perf_counter() - start
        stats = (await client.get("/stats")).json()

    generated = args.requests * args.tokens
    print(f"requests={args.requests} concurrency={args.concurrency} generated_tokens={generated}")
    print(f"wall_time_sec={wall:.3f} token_throughput={generated / wall:.1f}_tok/sec")
    print(f"latency_ms p50={statistics.median(latencies):.2f} p95={pct(latencies, 0.95):.2f} p99={pct(latencies, 0.99):.2f}")
    print(f"avg_batch_size={stats['average_batch_size']} max_batch_observed={stats['max_batch_observed']}")
    print(f"cache_used_pages={stats['cache']['used_pages']} pressure_events={stats['cache']['pressure_events']} evictions={stats['cache']['evictions']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("-n", "--requests", type=int, default=64)
    p.add_argument("-c", "--concurrency", type=int, default=16)
    p.add_argument("--tokens", type=int, default=80)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
