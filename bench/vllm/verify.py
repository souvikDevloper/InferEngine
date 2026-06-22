from __future__ import annotations

import json
import sys
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    infer_path, vllm_path, output_path = sys.argv[1:4]
    infer, baseline = read(infer_path), read(vllm_path)
    comparable = [
        "model_id",
        "tokenizer_id",
        "num_prompts",
        "request_rate",
        "max_concurrency",
        "total_input_tokens",
    ]
    mismatches = {
        key: {"inferengine": infer.get(key), "vllm": baseline.get(key)}
        for key in comparable
        if infer.get(key) != baseline.get(key)
    }
    infer_throughput = float(infer.get("output_throughput", 0))
    vllm_throughput = float(baseline.get("output_throughput", 0))
    ratio = infer_throughput / vllm_throughput if vllm_throughput > 0 else 0.0
    all_requests_succeeded = all(
        int(result.get("completed", 0)) == int(result.get("num_prompts", -1))
        and int(result.get("failed", 0)) == 0
        for result in (infer, baseline)
    )
    passed = not mismatches and all_requests_succeeded and ratio >= 0.91
    comparison = {
        "claim": "InferEngine output-token throughput is within 9% of vLLM",
        "pass": passed,
        "throughput_ratio": ratio,
        "inferengine_output_tokens_per_second": infer_throughput,
        "vllm_output_tokens_per_second": vllm_throughput,
        "all_requests_succeeded": all_requests_succeeded,
        "configuration_mismatches": mismatches,
        "official_metrics": {
            system: {
                key: result.get(key)
                for key in (
                    "request_throughput",
                    "output_throughput",
                    "total_token_throughput",
                    "mean_ttft_ms",
                    "median_ttft_ms",
                    "p50_ttft_ms",
                    "p99_ttft_ms",
                    "mean_itl_ms",
                    "median_itl_ms",
                    "p50_itl_ms",
                    "p99_itl_ms",
                )
            }
            for system, result in (("inferengine", infer), ("vllm", baseline))
        },
    }
    Path(output_path).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
