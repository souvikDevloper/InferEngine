from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from math import ceil
from threading import Lock


@dataclass
class SequenceRecord:
    sequence_id: str
    prompt_tokens: int
    generated_tokens: int = 0
    pages: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    completed: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


class KVCacheManager:
    """Paged KV-cache accounting layer.

    This models the memory-management behavior used by LLM serving systems:
    requests reserve fixed-size pages, generated tokens extend sequences, and
    completed or least-recently-used sequences are evicted under pressure.
    """

    def __init__(self, max_pages: int = 1024, page_size: int = 16, policy: str = "lru") -> None:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if policy not in {"lru", "fifo"}:
            raise ValueError("policy must be 'lru' or 'fifo'")
        self.max_pages = max_pages
        self.page_size = page_size
        self.policy = policy
        self._free_pages: deque[int] = deque(range(max_pages))
        self._seqs: OrderedDict[str, SequenceRecord] = OrderedDict()
        self._lock = Lock()
        self.pressure_events = 0
        self.evictions = 0

    def _pages_needed(self, tokens: int) -> int:
        return max(1, ceil(tokens / self.page_size))

    def allocate(self, sequence_id: str, prompt_tokens: int, max_new_tokens: int) -> None:
        with self._lock:
            if sequence_id in self._seqs:
                raise ValueError(f"sequence already exists: {sequence_id}")
            required = self._pages_needed(prompt_tokens + max_new_tokens)
            if required > self.max_pages:
                raise MemoryError("request exceeds total KV-cache capacity")
            self._ensure_free_pages(required)
            pages = [self._free_pages.popleft() for _ in range(required)]
            self._seqs[sequence_id] = SequenceRecord(sequence_id=sequence_id, prompt_tokens=prompt_tokens, pages=pages)

    def append_token(self, sequence_id: str) -> None:
        with self._lock:
            seq = self._seqs[sequence_id]
            seq.generated_tokens += 1
            seq.last_accessed = time.time()
            if self.policy == "lru":
                self._seqs.move_to_end(sequence_id)

    def touch(self, sequence_id: str) -> None:
        with self._lock:
            if sequence_id in self._seqs:
                self._seqs[sequence_id].last_accessed = time.time()
                if self.policy == "lru":
                    self._seqs.move_to_end(sequence_id)

    def complete(self, sequence_id: str) -> None:
        with self._lock:
            if sequence_id in self._seqs:
                self._seqs[sequence_id].completed = True

    def release(self, sequence_id: str) -> None:
        with self._lock:
            seq = self._seqs.pop(sequence_id, None)
            if not seq:
                return
            for page in seq.pages:
                self._free_pages.append(page)

    def _ensure_free_pages(self, required: int) -> None:
        while len(self._free_pages) < required:
            victim_id = self._pick_victim()
            if victim_id is None:
                self.pressure_events += 1
                raise MemoryError("KV-cache is full and no evictable sequence exists")
            victim = self._seqs.pop(victim_id)
            for page in victim.pages:
                self._free_pages.append(page)
            self.evictions += 1

    def _pick_victim(self) -> str | None:
        # Prefer completed sequences. Then fall back to FIFO/LRU victim.
        for sid, seq in self._seqs.items():
            if seq.completed:
                return sid
        if not self._seqs:
            return None
        return next(iter(self._seqs.keys()))

    def stats(self) -> dict[str, int | float | str]:
        with self._lock:
            used = self.max_pages - len(self._free_pages)
            return {
                "page_size": self.page_size,
                "max_pages": self.max_pages,
                "used_pages": used,
                "free_pages": len(self._free_pages),
                "active_sequences": len(self._seqs),
                "pressure_events": self.pressure_events,
                "evictions": self.evictions,
                "utilization": round(used / self.max_pages, 4),
                "policy": self.policy,
            }
