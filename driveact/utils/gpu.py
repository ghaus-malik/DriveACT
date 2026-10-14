"""GPU availability on a shared machine: query ``nvidia-smi`` and pick a GPU that has spare capacity."""

from __future__ import annotations

import shutil
import subprocess
import time


def query_gpus() -> list[dict]:
    """[{index, name, memory_total_mb, memory_used_mb, memory_free_mb, utilization}] (empty without nvidia-smi)."""
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    gpus = []
    for line in out.stdout.strip().splitlines():
        idx, name, total, used, util = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "memory_total_mb": int(total), "memory_used_mb": int(used),
                     "memory_free_mb": int(total) - int(used), "utilization": int(util)})
    return gpus


def sample_gpus(samples: int = 6, interval_s: float = 0.5) -> list[dict]:
    """Like :func:`query_gpus`, with utilisation averaged over a few samples (other jobs are often bursty)."""
    runs = []
    for i in range(samples):
        runs.append(query_gpus())
        if i < samples - 1:
            time.sleep(interval_s)
    gpus = runs[-1]
    for g in gpus:
        utils = [r[g["index"]]["utilization"] for r in runs if len(r) > g["index"]]
        g["utilization"] = int(round(sum(utils) / max(1, len(utils))))
        g["memory_free_mb"] = min(r[g["index"]]["memory_free_mb"] for r in runs if len(r) > g["index"])
        g["memory_used_mb"] = g["memory_total_mb"] - g["memory_free_mb"]
    return gpus


def pick_gpu(min_free_gb: float = 10.0, max_util: int = 90) -> tuple[int | None, list[dict]]:
    """Index of the GPU with the most free memory among those with >= ``min_free_gb`` free and an average
    utilisation <= ``max_util`` percent, or None if no GPU has capacity."""
    gpus = sample_gpus()
    ok = [g for g in gpus if g["memory_free_mb"] >= 1024 * min_free_gb and g["utilization"] <= max_util]
    if not ok:
        return None, gpus
    return max(ok, key=lambda g: (g["memory_free_mb"], -g["utilization"]))["index"], gpus


def describe(gpus: list[dict]) -> str:
    return "\n".join(
        f"  GPU {g['index']}: {g['memory_used_mb'] / 1024:.1f}/{g['memory_total_mb'] / 1024:.1f} GB used, "
        f"{g['utilization']} % average utilisation" for g in gpus
    ) or "  (no NVIDIA GPU visible)"
