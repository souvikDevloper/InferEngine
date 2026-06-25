from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class EngineConfig:
    max_batch_size: int = 8
    max_waiting: int = 2048
    max_pages: int = 1024
    page_size: int = 16
    eviction_policy: str = "lru"  # lru or fifo
    decode_interval_ms: int = 0
    bulk_generate: bool = False
    max_new_tokens_default: int = 64
    max_new_tokens_limit: int = 512
    admission_token_limit: int = 4096

    @classmethod
    def from_env(cls) -> "EngineConfig":
        defaults = cls()
        return cls(
            max_batch_size=_env_int("INFERENGINE_MAX_BATCH_SIZE", defaults.max_batch_size),
            max_waiting=_env_int("INFERENGINE_MAX_WAITING", defaults.max_waiting),
            max_pages=_env_int("INFERENGINE_MAX_PAGES", defaults.max_pages),
            page_size=_env_int("INFERENGINE_PAGE_SIZE", defaults.page_size),
            eviction_policy=os.getenv("INFERENGINE_EVICTION_POLICY", defaults.eviction_policy),
            decode_interval_ms=_env_int("INFERENGINE_DECODE_INTERVAL_MS", defaults.decode_interval_ms),
            bulk_generate=_env_bool("INFERENGINE_BULK_GENERATE", defaults.bulk_generate),
            max_new_tokens_default=_env_int("INFERENGINE_MAX_NEW_TOKENS_DEFAULT", defaults.max_new_tokens_default),
            max_new_tokens_limit=_env_int("INFERENGINE_MAX_NEW_TOKENS_LIMIT", defaults.max_new_tokens_limit),
            admission_token_limit=_env_int("INFERENGINE_ADMISSION_TOKEN_LIMIT", defaults.admission_token_limit),
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
