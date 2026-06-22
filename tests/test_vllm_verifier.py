import json
from pathlib import Path

from bench.vllm.verify import main


def test_verifier_accepts_matching_official_results(tmp_path: Path, monkeypatch):
    common = {
        "model_id": "llama",
        "tokenizer_id": "llama",
        "num_prompts": 1000,
        "request_rate": "inf",
        "max_concurrency": 32,
        "total_input_tokens": 512000,
        "completed": 1000,
        "failed": 0,
    }
    infer = {**common, "output_throughput": 91.0}
    baseline = {**common, "output_throughput": 100.0}
    infer_path, baseline_path, output_path = (
        tmp_path / "infer.json",
        tmp_path / "vllm.json",
        tmp_path / "comparison.json",
    )
    infer_path.write_text(json.dumps(infer), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["verify.py", str(infer_path), str(baseline_path), str(output_path)])
    assert main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["pass"] is True
