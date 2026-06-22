#!/usr/bin/env bash
set -euo pipefail

# This script orchestrates vLLM's official benchmark twice. It does not create
# requests or calculate latency itself; both systems are measured by the same
# vllm bench serve installation.
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
INFERENGINE_URL="${INFERENGINE_URL:-http://127.0.0.1:8000}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8001}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
RESULT_DIR="${RESULT_DIR:-benchmark-results/$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$RESULT_DIR"

python - <<'PY' > "$RESULT_DIR/environment.txt"
import platform
import subprocess
import sys
import torch
import vllm

print("python=", sys.version.replace("\n", " "))
print("platform=", platform.platform())
print("torch=", torch.__version__)
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

vllm bench serve "${common[@]}" \
  --base-url "$INFERENGINE_URL" \
  --label inferengine \
  --metadata "system=inferengine" \
  --result-filename inferengine.json \
  2>&1 | tee "$RESULT_DIR/inferengine.txt"

vllm bench serve "${common[@]}" \
  --base-url "$VLLM_URL" \
  --label vllm \
  --metadata "system=vllm" \
  --result-filename vllm.json \
  2>&1 | tee "$RESULT_DIR/vllm.txt"

python bench/vllm/verify.py "$RESULT_DIR/inferengine.json" "$RESULT_DIR/vllm.json" "$RESULT_DIR/comparison.json"
echo "Evidence: $RESULT_DIR"
