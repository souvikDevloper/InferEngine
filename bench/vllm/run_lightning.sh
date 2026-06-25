#!/usr/bin/env bash
set -euo pipefail

# Lightning Studio no-card exploratory runner.
#
# This is not the strict A10G resume gate. It is a low-credit GPU smoke/bench
# run that verifies installation, CUDA backend loading, official vLLM client
# compatibility, and a first paired throughput/latency comparison on whatever
# GPU shape Lightning assigns.

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
PYTHON_BIN="${PYTHON_BIN:-3.12}"
MIN_SINGLE_VRAM_GIB="${MIN_SINGLE_VRAM_GIB:-0}"
MIN_TOTAL_VRAM_GIB="${MIN_TOTAL_VRAM_GIB:-22}"

if [[ -z "${HF_TOKEN:-}" && ( "$MODEL" == meta-llama/* || "${REQUIRE_HF_TOKEN:-0}" == "1" ) ]]; then
  echo "HF_TOKEN is required for gated Llama models. Export it before running." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; switch the Lightning Studio to a GPU machine first." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

uv venv --clear --python "$PYTHON_BIN" .venv-lightning
# shellcheck disable=SC1091
source .venv-lightning/bin/activate

uv pip install --upgrade pip wheel setuptools
uv pip install "vllm[bench]==${VLLM_VERSION}" --torch-backend=auto
uv pip install "pyarrow<21"
uv pip install -e ".[gpu,dev]"

preflight_args=(
  --min-single-vram-gib "$MIN_SINGLE_VRAM_GIB"
  --min-total-vram-gib "$MIN_TOTAL_VRAM_GIB"
)
if [[ "$MODEL" == meta-llama/* || "${REQUIRE_HF_TOKEN:-0}" == "1" ]]; then
  preflight_args+=(--require-hf-token)
fi
python scripts/gpu_preflight.py "${preflight_args[@]}"
python -m compileall -q inferengine scripts bench/vllm
pytest -q

GPU_COUNT="$(
  python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

export MODEL
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$GPU_COUNT}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
if [[ -z "${INFERENGINE_DEVICE_MAP+x}" ]]; then
  if [[ "$GPU_COUNT" -gt 1 ]]; then
    export INFERENGINE_DEVICE_MAP=auto
  else
    export INFERENGINE_DEVICE_MAP=
  fi
fi
export INFERENGINE_TORCH_DTYPE="${INFERENGINE_TORCH_DTYPE:-auto}"
export INFERENGINE_DECODE_INTERVAL_MS="${INFERENGINE_DECODE_INTERVAL_MS:-0}"
export INFERENGINE_MAX_BATCH_SIZE="${INFERENGINE_MAX_BATCH_SIZE:-4}"
export NUM_PROMPTS="${NUM_PROMPTS:-100}"
export INPUT_LEN="${INPUT_LEN:-256}"
export OUTPUT_LEN="${OUTPUT_LEN:-64}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export VERIFY_STRICT="${VERIFY_STRICT:-0}"
export RESULT_DIR="${RESULT_DIR:-benchmark-results/lightning-$(date -u +%Y%m%dT%H%M%SZ)}"

bash ./bench/vllm/run_pair.sh
