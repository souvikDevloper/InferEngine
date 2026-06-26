# vLLM comparison protocol

The only accepted source for the headline comparison is `vllm bench serve`. The same installed vLLM benchmark client must drive InferEngine and vLLM on the same machine, model, tokenizer, dataset, request order, input/output lengths, concurrency, and precision.

## Resume gate

The statement “matched vLLM within 9% throughput” passes only when:

- both runs complete all requests with zero failures;
- the saved configurations and total input-token counts match;
- InferEngine output-token throughput / vLLM output-token throughput is at least 0.91;
- the GPU model, driver, CUDA stack, model revision, precision, and server commands are retained with the result.

The gate uses output-token throughput because “throughput” is otherwise ambiguous. Request throughput, total-token throughput, TTFT, TPOT, and ITL remain in the official result files.

## Fixed comparison profile

```text
model:             meta-llama/Meta-Llama-3-8B
dataset:           vLLM random dataset
input length:      512 tokens (fixed)
output length:     128 tokens (fixed, ignore EOS)
requests:          1,000
max concurrency:   32
arrival rate:      infinite/closed-loop saturation
seed:              42
percentiles:       p50 and p99
hardware:          one NVIDIA A10G
```

The gated Meta model requires an accepted Hugging Face license and token.

## Run

Use Linux with an NVIDIA GPU. Install the pinned official benchmark CLI:

```bash
python -m venv .venv-bench
source .venv-bench/bin/activate
pip install -r bench/vllm/requirements.txt
```

The default harness starts the two servers sequentially with identical model and precision settings. Sequential execution is required on a 24 GiB A10G. Run:

```bash
INFERENGINE_URL=http://127.0.0.1:8000 \
VLLM_URL=http://127.0.0.1:8001 \
MODEL=meta-llama/Meta-Llama-3-8B \
./bench/vllm/run_pair.sh
```

`run_pair.sh` is only an orchestrator. Both measurements are executed by `vllm bench serve`; it does not generate requests or calculate latency. `verify.py` reads the two official JSON results and applies the 0.91 gate.

## Paged-attention backend mode

The custom `transformers` backend is useful for scheduler/cache experiments, but it does not use page-native attention kernels. For the real paged-attention path, run:

```bash
INFERENGINE_BACKEND=vllm_paged \
MODEL=meta-llama/Meta-Llama-3-8B \
./bench/vllm/run_pair.sh
```

This starts:

- InferEngine API on port `8000`;
- a private vLLM paged-attention backend on port `8002`;
- the direct vLLM baseline on port `8001`.

The official benchmark client still hits InferEngine and vLLM separately. This profile measures the overhead of InferEngine's OpenAI-compatible API over vLLM's real paged-attention/block-manager engine. It should be reported as a vLLM-backed paged-attention mode, not as proof that the pure Transformers scheduler has achieved vLLM parity.

## Current verification status

No A10G result is checked into this repository. InferEngine now has two GPU paths:

- `INFERENGINE_BACKEND=transformers`: custom scheduler with Hugging Face CUDA execution; this remains below the 0.91 vLLM parity gate on the latest exploratory A100 smoke runs.
- `INFERENGINE_BACKEND=vllm_paged`: OpenAI-compatible InferEngine API backed by vLLM's real paged-attention backend; this is the intended path for a near-term paged-attention parity gate.

The resume claim remains unverified until a retained GPU run passes the 0.91 gate and the backend mode is stated with the result.
