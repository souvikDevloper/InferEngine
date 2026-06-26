# InferEngine

InferEngine is an inference-serving systems prototype with continuous request admission, decode-step scheduling, paged KV-capacity accounting, a real Hugging Face CUDA backend, an optional vLLM-backed paged-attention backend, streaming OpenAI-compatible completions, and Prometheus telemetry.

The repository includes a reproducible comparison gate based exclusively on vLLM's official `vllm bench serve` client. The vLLM-backed paged-attention path has a passing A100/Qwen evidence run; the exact LLaMA-3/A10G resume variant remains a separate gated-model/hardware run.

## Implemented

- async waiting and active queues;
- continuous admission between decode steps;
- fixed-page KV-capacity allocation with LRU/FIFO pressure policies;
- `/v1/generate`, `/v1/completions`, `/v1/models`, `/health`, `/stats`, and `/metrics`;
- Server-Sent Events with per-token OpenAI completion chunks and usage totals;
- vLLM benchmark-compatible request/response contract;
- official paired benchmark orchestration and a strict 0.91 output-token-throughput gate;
- tests for scheduling, cache lifecycle, concurrent batching, and streaming API compatibility;
- real batched prefill/decode for Hugging Face causal language models with per-request KV state;
- `INFERENGINE_BACKEND=vllm_paged`, which keeps InferEngine's API surface but delegates scheduling, block allocation, KV-cache paging, and attention kernels to vLLM's paged-attention engine;
- optional Hugging Face `device_map=auto` loading for tight or multi-GPU exploratory environments;
- a standalone Triton fused-QKV projection kernel with CPU layout and CUDA numerical-correctness tests;
- zero-delay active decode loop by default, avoiding artificial inter-token sleeps;
- sequential single-GPU comparison orchestration plus 200 ms NVIDIA utilization/VRAM sampling.

## Verification status

| Resume statement | Status | Required evidence |
|---|---|---|
| vLLM-backed paged-attention mode matched vLLM within 9% on Qwen2.5-7B/A100 | **verified** | [`docs/evidence/inferengine-vllm-a100-qwen-20260626.md`](docs/evidence/inferengine-vllm-a100-qwen-20260626.md) |
| matched vLLM within 9% on LLaMA-3 8B/A10G | **not yet verified** | two successful official vLLM JSON results + passing `comparison.json` |
| 38% lower GPU fragmentation | **not yet verified** | real CUDA allocator traces for fixed naive and paged-cache experiments |
| 2.1x longer context at the same VRAM | **not yet verified** | maximum admitted context under a fixed measured VRAM cap |
| 76% vs 41% GPU utilization | **not yet verified** | timestamped DCGM/NVML samples over identical 1,000-request runs |

The laptop-safe toy backend remains the default. Set `INFERENGINE_BACKEND=transformers` for the custom Hugging Face CUDA scheduler path. Set `INFERENGINE_BACKEND=vllm_paged` for the production paged-attention path backed by vLLM. The latter is expected to be the only near-term path capable of passing the 0.91 vLLM-throughput gate because it uses real paged-attention/block-manager kernels instead of repacking Hugging Face KV tensors in Python.

## Architecture

```mermaid
flowchart LR
    V["vllm bench serve"] --> O["OpenAI /v1/completions SSE"]
    O --> B{Backend}
    B -->|toy / transformers| Q[Waiting queue]
    Q --> S[Continuous scheduler]
    S --> A[Active decode batch]
    A --> M[Toy or Hugging Face CUDA decoder]
    A --> K[Paged KV capacity manager]
    S --> P[Prometheus metrics]
    B -->|vllm_paged| X[vLLM OpenAI backend]
    X --> Y[Paged attention + block manager]
```

## Local development

The default decoder is intentionally small and runs on CPU. It validates serving mechanics, not LLaMA performance.

Runtime knobs are environment-driven so GPU benchmark runs can tune without code changes:

```bash
INFERENGINE_MAX_BATCH_SIZE=8
INFERENGINE_MAX_PAGES=1024
INFERENGINE_PAGE_SIZE=16
INFERENGINE_DECODE_INTERVAL_MS=0
INFERENGINE_MAX_NEW_TOKENS_LIMIT=512
INFERENGINE_DEVICE_MAP=auto
INFERENGINE_TORCH_DTYPE=float16
INFERENGINE_MAX_MEMORY=0:14GiB,1:14GiB,cpu:48GiB
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
uvicorn inferengine.api.main:app --host 127.0.0.1 --port 8000
```

Streaming completion:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"torch-toy-decoder/cpu","prompt":"Explain batching","max_tokens":8,"stream":true,"stream_options":{"include_usage":true}}'
```

## Official vLLM comparison

The benchmark tooling is pinned to vLLM 0.23.0 and invokes the same command twice with the same arguments:

```bash
pip install -r bench/vllm/requirements.txt
INFERENGINE_URL=http://127.0.0.1:8000 \
VLLM_URL=http://127.0.0.1:8001 \
MODEL=meta-llama/Meta-Llama-3-8B \
./bench/vllm/run_pair.sh
```

To benchmark the real paged-attention path:

```bash
INFERENGINE_BACKEND=vllm_paged \
MODEL=meta-llama/Meta-Llama-3-8B \
./bench/vllm/run_pair.sh
```

In this mode InferEngine starts a private vLLM server on port `8002`, exposes InferEngine on `8000`, and proxies OpenAI completions through the paged-attention backend. The baseline still starts direct vLLM on `8001`, so the comparison measures InferEngine API/proxy overhead over the same underlying paged-attention engine.

It starts InferEngine and vLLM sequentially on the same GPU, then retains raw console output, complete official JSON results, per-request details, GPU/software environment, Git revision, NVIDIA utilization/VRAM samples, and a machine-readable comparison. See [GPU setup](docs/GPU_SETUP.md) and [the protocol](docs/benchmark.md).

For a no-card Lightning Studio exploratory run, use:

```bash
export HF_TOKEN=hf_your_token
./bench/vllm/run_lightning.sh
```

See [Lightning AI setup](docs/LIGHTNING_AI.md). A Lightning result is publishable only with the exact GPU name shown in `environment.txt`; it is not an A10G claim unless the assigned hardware is A10/A10G-class.

## Development benchmark

For scheduler regressions only:

```bash
python scripts/bench.py -n 64 -c 16 --tokens 80
```

Do not compare this script's output with vLLM. It is not the official harness and uses the toy model.

## Repository map

```text
inferengine/api/       HTTP and OpenAI-compatible streaming API
inferengine/core/      scheduler, tokenizer, and page allocator
inferengine/model/     toy, Hugging Face CUDA, and vLLM paged-attention backends
inferengine/kernels/   optional Triton fused-QKV projection
inferengine/metrics/   Prometheus instruments
bench/vllm/            official paired benchmark orchestration and gate
tests/                 cache, scheduler, and protocol tests
docs/                  architecture and benchmark contract
```

## Remaining claim boundary

The custom Hugging Face execution path and fused projection kernel now exist, but they do not pass the vLLM parity gate because mixed-length KV tensors are still repacked and attention is not page-native. The `vllm_paged` backend is the practical paged-attention rewrite path; a resume claim should state clearly whether the measured engine is the custom Transformers scheduler or the vLLM-backed paged-attention mode.
