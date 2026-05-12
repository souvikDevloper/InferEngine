#!/usr/bin/env python3
from __future__ import annotations

import httpx

import os

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")


def main() -> None:
    prompts = [
        "Explain quorum replication in one sentence",
        "Why does KV cache matter for LLM serving",
        "Describe continuous batching",
    ]
    for p in prompts:
        r = httpx.post(f"{BASE}/v1/generate", json={"prompt": p, "max_new_tokens": 24}, timeout=30)
        r.raise_for_status()
        data = r.json()
        print(f"[{data['request_id'][:8]}] tokens={data['generated_tokens']} latency_ms={data['latency_ms']}")
        print(data["text"])
        print()
    print("stats:")
    print(httpx.get(f"{BASE}/stats", timeout=10).json())


if __name__ == "__main__":
    main()
