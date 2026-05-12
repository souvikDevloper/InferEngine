from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    max_batch_size: int = 8
    max_waiting: int = 2048
    max_pages: int = 1024
    page_size: int = 16
    eviction_policy: str = "lru"  # lru or fifo
    decode_interval_ms: int = 2
    max_new_tokens_default: int = 64
    max_new_tokens_limit: int = 512
    admission_token_limit: int = 4096
