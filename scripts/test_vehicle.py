"""Stage 1: validate the vehicle (physics, steering, acceleration, braking, turning radius) with open-loop tests.

    python scripts/test_vehicle.py --headless                 # Isaac Lab articulation on an open test pad
    python scripts/test_vehicle.py --backend kinematic        # same manoeuvres on the CPU test double

Six cars run in parallel, one manoeuvre each, on flat ground without track geometry:
  settle     zero command for 3 s: the car must stay put and upright
  accel      +2 m/s^2 for 5 s: measured acceleration vs. command
  brake      accelerate to ~10 m/s, then full brake: stopping distance vs. v^2 / (2 * 6 m/s^2), no reverse
  turn       hold 5 m/s with steering 0.5: steady-state turning radius vs. bicycle model L / tan(delta)
  steer_step steering command 0 -> 1 at standstill... and at 3 m/s: rise time vs. the 45 deg/s slew limit
  no_reverse full brake at standstill: the car must not move backwards
Writes ``results/stage1/vehicle_test_<backend>.json`` and a plot; exits with status 1 if a check fails.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from driveact.utils.launch import base_parser, parse_and_launch, run_main  # noqa: E402

parser = base_parser(__doc__.splitlines()[0])
parser.add_argument("--out", default="results/stage1")
args, app = parse_and_launch(parser)

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from driveact.config import load_config, load_track_config  # noqa: E402
from driveact.tracks.generators import make_layout  # noqa: E402
from driveact.vehicle.control import VehicleGeometry  # noqa: E402

TESTS = ["settle", "accel", "brake", "turn", "steer_step", "no_reverse"]


def command(name: str, t: float, v: float, g: VehicleGeometry) -> tuple[float, float]:
    """Open-loop (steering, longitudinal) normalised command for test ``name`` at time t."""
    if name == "settle":
        return 0.0, 0.0
    if name == "accel":
        return 0.0, (2.0 / g.max_accel) if t < 5.0 else -0.5
    if name == "brake":
        return 0.0, (2.5 / g.max_accel) if t < 4.0 else -1.0
    if name == "turn":
        target = 5.0
        steer = 0.5 if t > 3.0 else 0.0
        return steer, float(np.clip(1.5 * (target - v) / g.max_accel, -1, 1))
    if name == "steer_step":
        return (1.0 if t > 2.0 else 0.0), (float(np.clip(1.5 * (3.0 - v) / g.max_accel, -1, 1)))
    if name == "no_reverse":
        return 0.0, -1.0
    raise ValueError(name)


def main():
    cfg = load_config(overrides=args.set)
    g = VehicleGeometry.from_config(cfg.vehicle)
    N = len(TESTS)
    layout = make_layout(1, 0, cfg)
    if args.backend == "isaac":
        from driveact.environments.isaac_backend import IsaacBackend

        backend = IsaacBackend(cfg, N, [layout] * N, load_track_config(1), device=args.device, build_static=False)
    else:
        from driveact.vehicle.kinematic import KinematicBackend

        backend = KinematicBackend(cfg, N, args.device)
    dev = backend.device
    backend.reset(torch.arange(N, device=dev), torch.zeros(N, 3, device=dev))
    dt = float(cfg.sim.control_dt)
    T = int(16.0 / dt)
    log = {k: [] for k in ("t", "x", "y", "yaw", "v", "steer", "z", "up_z", "cmd_s", "cmd_a")}
    for step in range(T):
        st = backend.state()
        t = step * dt
        cmds = [command(name, t, float(st["v_long"][i]), g) for i, name in enumerate(TESTS)]
        actions = torch.tensor(cmds, device=dev, dtype=torch.float32)
        for key, val in (("t", [t] * N), ("x", st["pos"][:, 0]), ("y", st["pos"][:, 1]), ("yaw", st["yaw"]),
                         ("v", st["v_long"]), ("steer", st["steer"]), ("z", st.get("z", torch.zeros(N))),
                         ("up_z", st.get("up_z", torch.ones(N))), ("cmd_s", actions[:, 0]), ("cmd_a", actions[:, 1])):
            log[key].append(np.asarray(val.cpu() if torch.is_tensor(val) else val, dtype=np.float64))
        backend.step(actions)
    L = {k: np.stack(v) for k, v in log.items()}  # (T, N)
    tt = L["t"][:, 0]
    i = {name: k for k, name in enumerate(TESTS)}
    res, checks = {}, {}

    k = i["settle"]
    drift = float(np.hypot(L["x"][:, k] - L["x"][0, k], L["y"][:, k] - L["y"][0, k]).max())
    res["settle"] = {"max_drift_m": drift, "max_speed": float(np.abs(L["v"][:, k]).max()),
                     "z_range": [float(L["z"][5:, k].min()), float(L["z"][5:, k].max())], "min_up_z": float(L["up_z"][:, k].min())}
    checks["settle"] = bool(drift < 0.05 and res["settle"]["min_up_z"] > 0.99)

    k = i["accel"]
    sel = (tt >= 1.0) & (tt <= 4.5)
    a_meas = float(np.polyfit(tt[sel], L["v"][sel, k], 1)[0])
    res["accel"] = {"commanded": 2.0, "measured": a_meas, "speed_at_5s": float(np.interp(5.0, tt, L["v"][:, k]))}
    checks["accel"] = bool(abs(a_meas - 2.0) / 2.0 < 0.15)

    k = i["brake"]
    j0 = int(np.searchsorted(tt, 4.0))
    v0 = float(L["v"][j0, k])
    stop = j0 + int(np.argmax(L["v"][j0:, k] < 0.05))
    dist = float(np.hypot(L["x"][stop, k] - L["x"][j0, k], L["y"][stop, k] - L["y"][j0, k]))
    ideal = v0**2 / (2 * g.max_brake)
    res["brake"] = {"v0": v0, "stopping_distance_m": dist, "ideal_m": ideal, "stopping_time_s": float(tt[stop] - tt[j0]),
                    "min_speed_after": float(L["v"][j0:, k].min())}
    checks["brake"] = bool(dist < 1.35 * ideal + 0.5 and res["brake"]["min_speed_after"] > -0.1)

    k = i["turn"]
    sel = tt >= 9.0
    yaw = np.unwrap(L["yaw"][:, k])
    omega = float(np.polyfit(tt[sel], yaw[sel], 1)[0])
    v_mean = float(L["v"][sel, k].mean())
    delta = 0.5 * g.max_steer
    radius_bicycle = g.wheelbase / math.tan(delta)  # rear axle
    radius = v_mean / abs(omega) if abs(omega) > 1e-6 else float("inf")
    res["turn"] = {"speed": v_mean, "steer_rad": delta, "yaw_rate": omega, "radius_m": radius,
                   "bicycle_radius_m": radius_bicycle, "measured_steer_rad": float(L["steer"][sel, k].mean())}
    checks["turn"] = bool(abs(radius - radius_bicycle) / radius_bicycle < 0.15 and omega > 0)

    k = i["steer_step"]
    j0 = int(np.searchsorted(tt, 2.0))
    target = g.max_steer
    reach = j0 + int(np.argmax(L["steer"][j0:, k] >= 0.95 * target))
    rise = float(tt[reach] - tt[j0]) if L["steer"][reach, k] >= 0.95 * target else float("inf")
    res["steer_step"] = {"target_rad": target, "final_rad": float(L["steer"][-1, k]), "rise_time_95_s": rise,
                         "slew_limited_time_s": 0.95 * target / g.max_steer_rate}
    checks["steer_step"] = bool(rise < 0.95 * target / g.max_steer_rate + 0.4 and abs(L["steer"][-1, k] - target) < math.radians(2))

    k = i["no_reverse"]
    res["no_reverse"] = {"min_speed": float(L["v"][:, k].min()),
                         "max_drift_m": float(np.hypot(L["x"][:, k] - L["x"][0, k], L["y"][:, k] - L["y"][0, k]).max())}
    checks["no_reverse"] = bool(res["no_reverse"]["min_speed"] > -0.05 and res["no_reverse"]["max_drift_m"] < 0.1)

    out = {"backend": args.backend, "geometry": g.__dict__, "min_turn_radius_rear_axle_m": g.min_turn_radius,
           "results": res, "checks": checks, "passed": all(checks.values())}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"vehicle_test_{args.backend}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for name in ("accel", "brake", "turn", "steer_step"):
            ax[0].plot(tt, L["v"][:, i[name]], label=name)
        ax[0].set_xlabel("t [s]")
        ax[0].set_ylabel("speed [m/s]")
        ax[0].legend()
        ax[1].plot(tt, np.degrees(L["steer"][:, i["steer_step"]]), label="measured")
        ax[1].plot(tt, np.degrees(L["cmd_s"][:, i["steer_step"]] * g.max_steer), "--", label="command")
        ax[1].set_ylabel("steering [deg]")
        ax[1].legend()
        ax[2].plot(L["x"][:, i["turn"]], L["y"][:, i["turn"]])
        ax[2].set_aspect("equal")
        ax[2].set_title(f"turn: R = {radius:.2f} m (bicycle {radius_bicycle:.2f} m)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"vehicle_test_{args.backend}.png"), dpi=120)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(json.dumps({"checks": checks, "results": res}, indent=1, default=float))
    print("PASSED" if out["passed"] else "FAILED", "->", path)
    backend.close()
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    run_main(main, app)
