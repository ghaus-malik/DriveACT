"""Aggregate episode records into the evaluation metrics (rates with 95 % Wilson intervals)."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import asdict

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _rate(records, pred) -> dict:
    n = len(records)
    k = sum(1 for r in records if pred(r))
    lo, hi = wilson(k, n)
    return {"rate": k / n if n else float("nan"), "count": k, "n": n, "ci95": [lo, hi]}


def summarize(records: list) -> dict:
    """Metrics over a list of :class:`EpisodeRecord` (one policy, one evaluation set)."""
    n = len(records)
    if n == 0:
        return {"episodes": 0}
    succ = [r for r in records if r.success]
    groups = sum(r.obstacle_groups for r in records)
    passed = sum(r.obstacle_groups_passed for r in records)

    def mean(key, rs=records):
        vals = [getattr(r, key) for r in rs]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "episodes": n,
        "goal_completion": _rate(records, lambda r: r.success),
        "reached_goal_any": _rate(records, lambda r: r.reached_goal),
        "collision": _rate(records, lambda r: r.outcome.startswith("collision")),
        "wall_collision": _rate(records, lambda r: r.outcome == "collision_wall"),
        "vehicle_collision": _rate(records, lambda r: r.outcome == "collision_vehicle"),
        "cone_collision": _rate(records, lambda r: r.outcome == "collision_cone"),
        "barrier_collision": _rate(records, lambda r: r.outcome == "collision_barrier"),
        "curb_strike": _rate(records, lambda r: r.curb_strike),
        "lane_departure": _rate(records, lambda r: r.lane_departure),
        "timeout_or_stuck": _rate(records, lambda r: r.outcome in ("timeout", "stuck")),
        "off_route_or_overshoot": _rate(records, lambda r: r.outcome in ("off_route", "overshoot")),
        "obstacle_avoidance": {
            "passed": passed,
            "total": groups,
            "rate": passed / groups if groups else float("nan"),
            "ci95": list(wilson(passed, groups)) if groups else [float("nan")] * 2,
        },
        "outcomes": dict(Counter(r.outcome for r in records)),
        "mean_progress": mean("progress"),
        "route_tracking_error_m": mean("mean_track_error"),
        "max_route_tracking_error_m": float(np.max([r.max_track_error for r in records])),
        "lane_departure_time_fraction": mean("lane_departure_fraction"),
        "travel_time_s_success": mean("time_s", succ),
        "mean_speed_mps": mean("mean_speed"),
        "steering_smoothness": mean("mean_abs_steer_rate"),
        "steering_jerk": mean("mean_abs_steer_jerk"),
        "accel_smoothness": mean("mean_abs_accel_rate"),
        "min_clearance_m": float(np.min([r.min_clearance for r in records])),
        "mean_min_clearance_m": mean("min_clearance"),
        "inference_latency_ms_per_env": float(np.mean([r.extra.get("latency_ms_per_env", float("nan")) for r in records])),
    }


def write_records(path_csv: str, records: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path_csv)), exist_ok=True)
    rows = []
    for r in records:
        d = asdict(r)
        extra = d.pop("extra", {})
        d["latency_ms_per_env"] = extra.get("latency_ms_per_env")
        rows.append(d)
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=float)
