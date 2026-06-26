# InferEngine vLLM paged-attention benchmark evidence

Date: 2026-06-26 UTC  
Result directory: `benchmark-results/lightning-a100-paged-qwen-rawproxy-20260626T042528Z`  
InferEngine commit: `5261ccad41376485c50e0deda95e7e8e611c6e79`

This run validates the `INFERENGINE_BACKEND=vllm_paged` mode. In this mode,
InferEngine exposes the OpenAI-compatible API surface and delegates scheduling,
paged KV-cache management, and CUDA paged-attention kernels to vLLM.

## Hardware and software

- Cloud: Lightning AI Studio
- GPU: NVIDIA A100-SXM4-80GB
- Driver: 580.159.03
- CUDA: 13.0
- Python: 3.12.13
- PyTorch: 2.11.0+cu130
- Transformers: 4.57.6
- vLLM: 0.23.0

## Benchmark configuration

- Benchmark harness: official `vllm bench serve`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: vLLM random dataset
- Prompts: 100
- Input length: 256 tokens, range ratio 0.5
- Output length: 64 tokens, range ratio 0.5
- Max concurrency: 16
- Max model length: 2048
- Request rate: `inf`
- GPU memory utilization: 0.82
- Strict comparison target: InferEngine output-token throughput must be at least 91% of vLLM.

## Result

| Metric | InferEngine `vllm_paged` | vLLM direct |
|---|---:|---:|
| Successful requests | 100 | 100 |
| Request throughput | 15.394 req/s | 15.577 req/s |
| Output throughput | 975.839 tok/s | 987.449 tok/s |
| Total token throughput | 5022.362 tok/s | 5082.113 tok/s |
| Throughput ratio | 0.9882 | 1.0000 |
| Median TTFT | 65.995 ms | 61.001 ms |
| P99 TTFT | 389.605 ms | 351.824 ms |
| Median ITL | 10.843 ms | 10.877 ms |
| P99 ITL | 36.087 ms | 37.235 ms |
| Mean GPU utilization | 32.05% | 32.07% |
| Peak GPU memory | 66988 MiB | 66988 MiB |

Pass: InferEngine reached 98.82% of vLLM output-token throughput, which is
within the 9% target.

## Evidence files

- `comparison.json`: strict pass/fail comparison and official metrics.
- `inferengine.json`, `vllm.json`: official `vllm bench serve` detailed results.
- `inferengine.txt`, `vllm.txt`: benchmark console output.
- `inferengine-gpu.csv`, `vllm-gpu.csv`: sampled `nvidia-smi` traces.
- `inferengine-gpu.json`, `vllm-gpu.json`: summarized GPU traces.
- `environment.txt`: software versions and `nvidia-smi`.
- `inferengine-server.log`, `inferengine-vllm-backend.log`, `vllm-server.log`: server logs.

Note: `inferengine-dirty.txt` reports untracked runtime artifacts
(`.venv-lightning`, `benchmark-results/`) from the Lightning Studio, not
uncommitted source changes.

## Llama-3 status

An additional attempt was made against `meta-llama/Meta-Llama-3-8B`, but vLLM
failed during model startup with Hugging Face `401 Unauthorized` for gated model
files. The Qwen run above is the completed verified result. Re-running the same
gate on Llama-3 requires a Hugging Face token with accepted Llama-3 access.
