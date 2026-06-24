# Lightning AI no-card GPU run

Lightning Studio is useful for a free/no-card exploratory GPU run. It is not automatically an A10G result, so publish the GPU name that appears in the retained evidence.

Use this path to verify:

- the real Hugging Face CUDA backend starts;
- the official `vllm bench serve` client can hit InferEngine and vLLM;
- the benchmark artifacts are retained;
- rough throughput/latency behavior on the assigned Studio GPU.

Do not use this path to claim "A10G" unless Lightning actually assigns an A10/A10G-class GPU and the evidence file shows that hardware.

## Studio setup

1. Open a Lightning Studio.
2. Switch the Studio from CPU to a GPU machine.
3. Open the Studio terminal.
4. Clone the repo and run the Lightning harness:

```bash
git clone https://github.com/souvikDevloper/InferEngine.git
cd InferEngine
export HF_TOKEN=hf_your_token
./bench/vllm/run_lightning.sh
```

The script creates `.venv-lightning`, installs vLLM plus InferEngine GPU dependencies, runs preflight checks, runs tests, and then starts the two servers sequentially.

## Default low-credit profile

The default profile is intentionally small:

```bash
NUM_PROMPTS=100
INPUT_LEN=256
OUTPUT_LEN=64
MAX_CONCURRENCY=4
MAX_MODEL_LEN=2048
VERIFY_STRICT=0
```

This is a smoke/bench run, not the final resume gate. It records the official vLLM JSON and GPU telemetry even if the "within 9%" threshold fails.

## Multi-GPU Studio profile

If the Studio provides multiple 16 GB GPUs, keep tensor parallelism and Hugging Face automatic device placement enabled:

```bash
export TENSOR_PARALLEL_SIZE=2
export INFERENGINE_DEVICE_MAP=auto
export INFERENGINE_TORCH_DTYPE=float16
./bench/vllm/run_lightning.sh
```

If the Studio provides one 24 GB GPU, this is usually enough:

```bash
export TENSOR_PARALLEL_SIZE=1
export INFERENGINE_DEVICE_MAP=
export INFERENGINE_TORCH_DTYPE=auto
./bench/vllm/run_lightning.sh
```

## If Llama-3 8B does not fit

Use a smaller open model only to debug the harness:

```bash
export MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export MIN_TOTAL_VRAM_GIB=8
./bench/vllm/run_lightning.sh
```

A smaller-model run proves the benchmark plumbing, not the resume Llama-3 8B claim.

## Evidence to keep

The script writes to `benchmark-results/lightning-<timestamp>/`.

Keep these files:

- `environment.txt`
- `inferengine.json`
- `vllm.json`
- `comparison.json`
- `inferengine-gpu.json`
- `vllm-gpu.json`
- `inferengine-server.log`
- `vllm-server.log`
- `inferengine-commit.txt`
- `inferengine-dirty.txt`

Resume wording must match the evidence. If the evidence says L4, T4, A100, or A10, use that exact GPU name.
