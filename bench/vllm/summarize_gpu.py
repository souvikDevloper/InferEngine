from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * value))]


source, target = map(Path, sys.argv[1:3])
utilization: list[float] = []
memory: list[float] = []
with source.open(encoding="utf-8") as handle:
    for row in csv.reader(handle):
        if len(row) >= 4:
            utilization.append(float(row[1].strip()))
            memory.append(float(row[2].strip()))
if not utilization:
    raise SystemExit("no nvidia-smi samples captured")
summary = {
    "sample_interval_ms": 200,
    "samples": len(utilization),
    "gpu_utilization_mean_percent": statistics.fmean(utilization),
    "gpu_utilization_p50_percent": percentile(utilization, 0.50),
    "gpu_utilization_p99_percent": percentile(utilization, 0.99),
    "memory_used_peak_mib": max(memory),
}
target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
