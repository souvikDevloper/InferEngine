#!/usr/bin/env bash
set -euo pipefail

# Both servers run sequentially because two FP16 8B instances do not fit on one
# 24 GiB A10G. The same official vLLM client, model, request order, GPU, and
# environment are used for both measurements.
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
INFERENGINE_URL="${INFERENGINE_URL:-http://127.0.0.1:8000}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8001}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
RESULT_DIR="${RESULT_DIR:-benchmark-results/$(date -u +%Y%m%dT%H%M%SZ)}"
MANAGE_SERVERS="${MANAGE_SERVERS:-1}"
VERIFY_STRICT="${VERIFY_STRICT:-1}"

mkdir -p "$RESULT_DIR"

python - <<'PY' > "$RESULT_DIR/environment.txt"
import platform
import subprocess
import sys
import torch
import transformers
import vllm

print("python=", sys.version.replace("\n", " "))
print("platform=", platform.platform())
print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("vllm=", vllm.__version__)
print("cuda=", torch.version.cuda)
print("cudnn=", torch.backends.cudnn.version())
subprocess.run(["nvidia-smi"], check=True)
PY
git rev-parse HEAD > "$RESULT_DIR/inferengine-commit.txt"
git status --porcelain > "$RESULT_DIR/inferengine-dirty.txt"

common=(
  --backend vllm
  --endpoint /v1/completions
  --model "$MODEL"
  --tokenizer "$MODEL"
  --dataset-name random
  --random-input-len "$INPUT_LEN"
  --random-output-len "$OUTPUT_LEN"
  --random-range-ratio 1
  --num-prompts "$NUM_PROMPTS"
  --max-concurrency "$MAX_CONCURRENCY"
  --request-rate inf
  --ignore-eos
  --seed 42
  --percentile-metrics ttft,tpot,itl
  --metric-percentiles 50,99
  --save-result
  --save-detailed
  --result-dir "$RESULT_DIR"
)

server_pid=""
monitor_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
  monitor_pid=""
  server_pid=""
}
trap cleanup EXIT

wait_for_server() {
  local url="$1"
  for _ in $(seq 1 600); do
    if curl --silent --fail "$url/health" >/dev/null; then return 0; fi
    if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
      echo "server exited during startup" >&2
      return 1
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  return 1
}

start_inferengine() {
  INFERENGINE_BACKEND=transformers \
    INFERENGINE_MODEL="$MODEL" \
    INFERENGINE_DECODE_INTERVAL_MS="${INFERENGINE_DECODE_INTERVAL_MS:-0}" \
    INFERENGINE_MAX_BATCH_SIZE="${INFERENGINE_MAX_BATCH_SIZE:-8}" \
    INFERENGINE_DEVICE_MAP="${INFERENGINE_DEVICE_MAP:-}" \
    INFERENGINE_MAX_MEMORY="${INFERENGINE_MAX_MEMORY:-}" \
    INFERENGINE_TORCH_DTYPE="${INFERENGINE_TORCH_DTYPE:-auto}" \
    python -m uvicorn inferengine.api.main:app --host 127.0.0.1 --port 8000 \
    >"$RESULT_DIR/inferengine-server.log" 2>&1 &
  server_pid=$!
  wait_for_server "$INFERENGINE_URL"
}

start_vllm() {
  vllm_args=(
    serve "$MODEL"
    --host 127.0.0.1
    --port 8001
    --dtype auto
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  )
  if [[ "$TENSOR_PARALLEL_SIZE" != "1" ]]; then
    vllm_args+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
  fi
  vllm "${vllm_args[@]}" \
    >"$RESULT_DIR/vllm-server.log" 2>&1 &
  server_pid=$!
  wait_for_server "$VLLM_URL"
}

run_official_benchmark() {
  local label="$1" url="$2"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits --loop-ms=200 >"$RESULT_DIR/$label-gpu.csv" &
  monitor_pid=$!
  vllm bench serve "${common[@]}" \
    --base-url "$url" \
    --label "$label" \
    --metadata "system=$label" \
    --result-filename "$label.json" \
    2>&1 | tee "$RESULT_DIR/$label.txt"
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=""
  python bench/vllm/summarize_gpu.py "$RESULT_DIR/$label-gpu.csv" "$RESULT_DIR/$label-gpu.json"
}

if [[ "$MANAGE_SERVERS" == "1" ]]; then
  start_inferengine
fi
run_official_benchmark inferengine "$INFERENGINE_URL"
if [[ "$MANAGE_SERVERS" == "1" ]]; then
  cleanup
  start_vllm
fi
run_official_benchmark vllm "$VLLM_URL"
if [[ "$MANAGE_SERVERS" == "1" ]]; then cleanup; fi

if [[ "$VERIFY_STRICT" == "1" ]]; then
  python bench/vllm/verify.py "$RESULT_DIR/inferengine.json" "$RESULT_DIR/vllm.json" "$RESULT_DIR/comparison.json"
else
  python bench/vllm/verify.py "$RESULT_DIR/inferengine.json" "$RESULT_DIR/vllm.json" "$RESULT_DIR/comparison.json" || true
fi
echo "Evidence: $RESULT_DIR"
