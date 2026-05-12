from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter("inferengine_requests_total", "Total generation requests", ["status"])
TOKENS_GENERATED = Counter("inferengine_tokens_generated_total", "Total generated tokens")
BATCH_SIZE = Gauge("inferengine_batch_size", "Current decode batch size")
ACTIVE_REQUESTS = Gauge("inferengine_active_requests", "Active requests in decode loop")
WAITING_REQUESTS = Gauge("inferengine_waiting_requests", "Waiting requests in admission queue")
KV_USED_PAGES = Gauge("inferengine_kv_used_pages", "KV-cache pages currently in use")
KV_PRESSURE_EVENTS = Gauge("inferengine_kv_pressure_events", "KV-cache pressure events")
REQUEST_LATENCY = Histogram(
    "inferengine_request_latency_seconds",
    "End-to-end request latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
DECODE_STEPS = Counter("inferengine_decode_steps_total", "Total decode scheduler steps")
