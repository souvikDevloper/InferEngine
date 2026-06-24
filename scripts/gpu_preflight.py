from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate InferEngine/vLLM GPU benchmark prerequisites.")
    parser.add_argument("--min-single-vram-gib", type=float, default=22.0)
    parser.add_argument("--min-total-vram-gib", type=float, default=22.0)
    parser.add_argument("--min-compute-capability", type=float, default=7.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Linux", platform.system() == "Linux", platform.platform()))
    checks.append(("Python 3.10-3.13", (3, 10) <= sys.version_info[:2] <= (3, 13), platform.python_version()))
    checks.append(("CUDA available", torch.cuda.is_available(), torch.__version__))
    if torch.cuda.is_available():
        memories: list[float] = []
        capabilities: list[tuple[int, int]] = []
        names: list[str] = []
        for device in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device)
            memory_gib = props.total_memory / 2**30
            major, minor = torch.cuda.get_device_capability(device)
            memories.append(memory_gib)
            capabilities.append((major, minor))
            names.append(f"{device}:{props.name}:{memory_gib:.1f}GiB:cc{major}.{minor}")
        min_major = int(args.min_compute_capability)
        min_minor = int(round((args.min_compute_capability - min_major) * 10))
        checks.append(
            (f"CUDA capability >= {args.min_compute_capability:.1f}", all(cap >= (min_major, min_minor) for cap in capabilities), ", ".join(names))
        )
        checks.append(
            (
                f"single GPU memory >= {args.min_single_vram_gib:.1f} GiB",
                max(memories) >= args.min_single_vram_gib,
                f"largest={max(memories):.1f} GiB",
            )
        )
        checks.append(
            (
                f"total GPU memory >= {args.min_total_vram_gib:.1f} GiB",
                sum(memories) >= args.min_total_vram_gib,
                f"total={sum(memories):.1f} GiB across {len(memories)} GPU(s)",
            )
        )
    checks.append(("transformers installed", importlib.util.find_spec("transformers") is not None, "python package"))
    checks.append(("triton installed", importlib.util.find_spec("triton") is not None, "python package"))
    checks.append(("vllm installed", importlib.util.find_spec("vllm") is not None, "python package"))
    checks.append(("Hugging Face token", bool(os.getenv("HF_TOKEN")), "HF_TOKEN environment variable"))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL':4}  {name:28} {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
