# A10G GPU setup

The resume comparison must run on Linux and on one NVIDIA A10G (24 GiB). Native Windows is not a valid vLLM CUDA target. Use an Ubuntu 22.04/24.04 cloud instance, or WSL2 only for development.

## Required host

- NVIDIA A10G 24 GiB or an explicitly disclosed replacement with at least 22 GiB usable VRAM;
- NVIDIA driver capable of loading the CUDA runtime selected by vLLM;
- Ubuntu Linux x86-64;
- Python 3.12;
- a Hugging Face account with access to `meta-llama/Meta-Llama-3-8B` and an `HF_TOKEN`.

vLLM's current official requirements are Linux, Python 3.10-3.13, and NVIDIA compute capability 7.5 or newer. Its current prebuilt CUDA wheel uses CUDA 12.9; installing the full CUDA toolkit is unnecessary for the wheel, but is required for compiling custom CUDA/C++ extensions.

## Install

```bash
sudo apt-get update
sudo apt-get install -y git curl python3.12 python3.12-venv build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/souvikDevloper/InferEngine.git
cd InferEngine
uv venv --python 3.12 .venv-gpu
source .venv-gpu/bin/activate

uv pip install "vllm[bench]==0.23.0" --torch-backend=auto
uv pip install -e ".[gpu,dev]" --no-deps
export HF_TOKEN=hf_your_token
python scripts/gpu_preflight.py
pytest -q
```

Do not install another PyTorch build into this environment after vLLM. vLLM wheels are compiled against specific PyTorch/CUDA combinations; replacing Torch can create binary incompatibilities.

## Run the real backend

```bash
INFERENGINE_BACKEND=transformers \
INFERENGINE_MODEL=meta-llama/Meta-Llama-3-8B \
INFERENGINE_DECODE_INTERVAL_MS=0 \
INFERENGINE_MAX_BATCH_SIZE=8 \
INFERENGINE_MAX_PAGES=1024 \
python -m uvicorn inferengine.api.main:app --host 127.0.0.1 --port 8000
```

Tune `INFERENGINE_MAX_BATCH_SIZE` and `INFERENGINE_MAX_PAGES` only if the preflight or server logs show unused VRAM or admission-limit rejections. Keep `INFERENGINE_DECODE_INTERVAL_MS=0` for the benchmark run; adding a fixed decode sleep directly hurts inter-token latency and token throughput.

## Run the controlled comparison

```bash
MODEL=meta-llama/Meta-Llama-3-8B ./bench/vllm/run_pair.sh
```

The harness runs the two servers sequentially on the same GPU, because two FP16 8B servers do not fit together on a 24 GiB A10G. It retains the official vLLM JSON, per-request detail, server logs, exact environment, Git state, and 200 ms `nvidia-smi` utilization/VRAM samples for each system.

For free/no-card exploratory runs on Lightning Studio, use `docs/LIGHTNING_AI.md` instead. Those results are useful, but the resume wording must name the actual GPU Lightning assigns.
