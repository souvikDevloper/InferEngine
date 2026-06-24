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

## Current verification status

No A10G result is checked into this repository. InferEngine now has a real Hugging Face LLaMA CUDA backend with batched prefill/decode and a standalone Triton fused-QKV kernel, but the fused kernel is not yet wired into every attention layer and mixed-length KV tensors are repacked. The resume claim remains unverified until the retained A10G run passes the 0.91 gate.
