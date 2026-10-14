"""Closed-loop evaluation (headless) of the expert, single-step BC or ACT on a track split.

    python scripts/evaluate.py --headless --policy act --checkpoint checkpoints/act_track4/best.pt --track 4 --split test
    python scripts/evaluate.py --headless --policy expert --track 3 --split test --episodes 50
    python scripts/evaluate.py --backend kinematic --policy act --checkpoint /tmp/ckpt/best.pt --track 1 --episodes 4

Obstacle placements come from the evaluation stream (``eval.seeds.eval_obstacles``), disjoint from the stream used
for data collection, so evaluation configurations are unseen. Writes ``summary.json`` (metrics, settings),
``episodes.csv`` (one row per episode) and top-down plots of the first ``--plots`` episodes to ``--out``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from driveact.utils.launch import base_parser, parse_and_launch, run_main  # noqa: E402

parser = base_parser(__doc__.splitlines()[0])
parser.add_argument("--policy", choices=("expert", "act", "bc"), required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--track", type=int, required=True, choices=(1, 2, 3, 4))
parser.add_argument("--split", default="test")
parser.add_argument("--layout_seeds", type=int, nargs="*", default=None)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--first_episode", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--stream", type=int, default=None, help="obstacle stream (default: eval.seeds.eval_obstacles)")
parser.add_argument("--replan_every", type=int, default=None)
parser.add_argument("--temporal_ensemble", action="store_true")
parser.add_argument("--plots", type=int, default=4, help="top-down trajectory plots of the first N episodes")
parser.add_argument("--out", required=True)
args, app = parse_and_launch(parser)

import json  # noqa: E402

from driveact.config import load_config, load_track_config  # noqa: E402
from driveact.environments.factory import make_env, resolve_layout_seeds  # noqa: E402
from driveact.evaluation.metrics import summarize, write_json, write_records  # noqa: E402
from driveact.evaluation.rollout import run_episodes  # noqa: E402
from driveact.experts.expert import ScriptedExpert  # noqa: E402


def make_controller(cfg, env):
    if args.policy == "expert":
        return ScriptedExpert(cfg), {}
    if not args.checkpoint:
        sys.exit("--checkpoint is required for learned policies")
    from driveact.models.controller import PolicyController
    from driveact.training.trainer import load_checkpoint

    model, norm, ckpt = load_checkpoint(args.checkpoint, env.device)
    if ckpt["kind"] != args.policy:
        sys.exit(f"checkpoint is a '{ckpt['kind']}' policy, not '{args.policy}'")
    inf = cfg.act.inference
    replan = int(args.replan_every if args.replan_every is not None else inf.replan_every)
    te = bool(args.temporal_ensemble or inf.temporal_ensemble)
    ctrl = PolicyController(model, norm, env.N, env.device, replan_every=replan, temporal_ensemble=te,
                            ensemble_k=float(inf.ensemble_k))
    return ctrl, {"checkpoint": os.path.abspath(args.checkpoint), "replan_every": ctrl.replan_every,
                  "temporal_ensemble": ctrl.temporal_ensemble, "chunk_size": ctrl.k, "epoch": ckpt.get("epoch")}


def main():
    cfg = load_config(overrides=args.set)
    seeds = resolve_layout_seeds(args.track, args.split, args.layout_seeds)
    stream = int(cfg.eval.seeds.eval_obstacles if args.stream is None else args.stream)
    env = make_env(cfg, args.track, seeds, args.num_envs, backend=args.backend, device=args.device, obstacle_stream=stream)
    ctrl, settings = make_controller(cfg, env)
    episodes = list(range(args.first_episode, args.first_episode + args.episodes))
    records = run_episodes(env, ctrl, episodes, trace=args.plots > 0, progress=True)
    records.sort(key=lambda r: r.episode)
    summary = {
        "policy": args.policy,
        "track": args.track,
        "split": args.split,
        "layout_seeds": seeds,
        "backend": args.backend,
        "obstacle_stream": stream,
        "episodes": episodes[0:1] + episodes[-1:],
        **settings,
        "metrics": summarize(records),
    }
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "summary.json"), summary)
    write_records(os.path.join(args.out, "episodes.csv"), records)
    if args.plots:
        from driveact.tracks.generators import make_layout
        from driveact.tracks.obstacles import sample_obstacles
        from driveact.utils.seeding import episode_rng
        from driveact.visualization.topdown import plot_layout_png

        tc = load_track_config(args.track)
        for r in records[: args.plots]:
            layout = make_layout(args.track, r.layout_seed, cfg)
            obs = sample_obstacles(layout, tc.obstacles, cfg, episode_rng(stream, args.track, r.layout_seed, r.episode))
            plot_layout_png(os.path.join(args.out, f"episode_{r.episode:04d}_{r.outcome}.png"), layout, obs,
                            trajectories={f"{args.policy} ({r.outcome})": r.extra["trace"][:, :2]},
                            title=f"Track {args.track} layout {r.layout_seed} episode {r.episode}: {r.outcome}")
    m = summary["metrics"]
    print(json.dumps({"goal_completion": m["goal_completion"], "collision": m["collision"]["rate"],
                      "curb_strike": m["curb_strike"]["rate"], "obstacle_avoidance": m["obstacle_avoidance"]["rate"],
                      "outcomes": m["outcomes"]}, indent=1))
    env.backend.close()


if __name__ == "__main__":
    run_main(main, app)
