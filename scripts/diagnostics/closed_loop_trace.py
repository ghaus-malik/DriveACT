"""Step-by-step closed-loop trace of a policy approaching an obstacle (CPU test double by default).

    python scripts/diagnostics/closed_loop_trace.py --checkpoint checkpoints/act_s5_track12/best.pt --track 2 --episode 0

Prints, every few steps: route position, lateral offset, speed, distance to the obstacle group, the policy's command
and the command the expert would give in the same state (the expert is only queried, never executed).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--track", type=int, default=2)
    p.add_argument("--split", default="train")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stream", type=int, default=0)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--replan_every", type=int, default=2)
    args = p.parse_args()

    from driveact.config import load_config
    from driveact.environments.factory import make_env, resolve_layout_seeds
    from driveact.experts.expert import ScriptedExpert
    from driveact.models.controller import PolicyController
    from driveact.training.trainer import load_checkpoint

    cfg = load_config()
    seeds = resolve_layout_seeds(args.track, args.split, None)
    env = make_env(cfg, args.track, seeds[:1], 1, backend="kinematic", obstacle_stream=args.stream)
    model, norm, _ = load_checkpoint(args.checkpoint)
    ctrl = PolicyController(model, norm, 1, "cpu", replan_every=args.replan_every)
    expert = ScriptedExpert(cfg)
    env.reset([0], [args.episode])
    ctrl.reset([0])
    g = env.obstacles[0].groups[0] if env.obstacles[0].groups else None
    spec = env.spec
    for t in range(2000):
        obs = env.observe()
        a = ctrl.act(obs, env)
        e = expert.act(obs, env)
        front = float(obs[0, spec.num_sectors // 2 - 3: spec.num_sectors // 2 + 3].min()) * cfg.sensors.lidar.max_range
        ref = float(env.reference_offset()[0])
        if t % args.every == 0:
            dist = (g.s_min - float(env.s[0])) if g is not None else float("nan")
            print(f"t={t:4d} s={float(env.s[0]):6.1f} to_obst={dist:6.1f} lat={float(env.lateral[0]):+.2f} ref={ref:+.2f} "
                  f"v={float(env.state['v_long'][0]):5.2f} front={front:5.1f} | policy steer={float(a[0, 0]):+.3f} acc={float(a[0, 1]):+.2f} "
                  f"| expert steer={float(e[0, 0]):+.3f} acc={float(e[0, 1]):+.2f}")
        info = env.step(a)
        if bool(info["done"][0]):
            print("done:", env.episode_record(0).outcome, "at s", round(float(env.s[0]), 1))
            break


if __name__ == "__main__":
    main()
