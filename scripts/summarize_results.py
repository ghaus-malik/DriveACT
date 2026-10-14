"""Collect every ``summary.json`` under a results folder into one Markdown comparison table.

    python scripts/summarize_results.py --root results/eval --out results/eval/summary.md

Only numbers that exist in the evaluation outputs are printed; missing runs are simply absent.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def fmt_rate(m: dict) -> str:
    r = m.get("rate", float("nan"))
    lo, hi = m.get("ci95", [float("nan")] * 2)
    return f"{100 * r:.1f} % [{100 * lo:.0f}, {100 * hi:.0f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="results/eval")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "**", "summary.json"), recursive=True)):
        with open(path) as f:
            s = json.load(f)
        m = s["metrics"]
        if not m.get("episodes"):
            continue
        name = os.path.basename(os.path.dirname(path))
        policy = s["policy"] + (f" k={s['chunk_size']}" if s.get("chunk_size", 1) > 1 else "") + \
            (" TE" if s.get("temporal_ensemble") else (f" r={s['replan_every']}" if s.get("replan_every") else ""))
        rows.append([name, f"T{s['track']} {s['split']}", policy, str(m["episodes"]), fmt_rate(m["goal_completion"]),
                     fmt_rate(m["collision"]), fmt_rate(m["curb_strike"]), fmt_rate(m["lane_departure"]),
                     f"{100 * m['obstacle_avoidance']['rate']:.1f} %" if m["obstacle_avoidance"]["total"] else "-",
                     f"{m['route_tracking_error_m']:.2f}", f"{m['travel_time_s_success']:.1f}",
                     f"{m['steering_smoothness']:.3f}", f"{m['accel_smoothness']:.3f}",
                     f"{m['inference_latency_ms_per_env']:.2f}"])
    header = ["run", "track/split", "policy", "episodes", "goal completion", "collision", "curb strike", "lane departure",
              "obstacle avoidance", "tracking err [m]", "time [s]", "|Δsteer|", "|Δaccel|", "latency [ms/env]"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
