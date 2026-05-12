# InferEngine Architecture

InferEngine is organized around a serving loop with four layers:

1. API ingress through FastAPI.
2. Async request admission into a waiting queue.
3. Continuous batching decode loop.
4. Paged KV-cache accounting and sequence release.

The implementation deliberately uses a lightweight decoder so the system remains runnable on a normal laptop while still exposing the operational behavior of an inference server.
