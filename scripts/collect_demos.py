"""Collect expert demonstrations (headless) into an HDF5 file.

    python scripts/collect_demos.py --headless --track 1 --episodes 200 --num_envs 16 --out data/track1_train.h5
    python scripts/collect_demos.py --backend kinematic --track 1 --episodes 8 --out /tmp/t1.h5     # CPU dry run

Episodes ``first_episode .. first_episode + episodes - 1`` are run on the layouts of ``--split`` (round-robin over
environments). A fraction of the episodes (``expert.noise.episode_fraction``) executes the expert action plus
Ornstein-Uhlenbeck noise while recording the clean expert action as the label (DART). By default only successful
episodes are written. A JSON summary is written next to the HDF5 file.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from driveact.utils.launch import base_parser, parse_and_launch, run_main  # noqa: E402

parser = base_parser(__doc__.splitlines()[0])
parser.add_argument("--track", type=int, required=True, choices=(1, 2, 3, 4))
parser.add_argument("--split", default="train")
parser.add_argument("--layout_seeds", type=int, nargs="*", default=None, help="override the split's layout seeds")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--first_episode", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--noise_fraction", type=float, default=None, help="override expert.noise.episode_fraction")
parser.add_argument("--keep_failures", action="store_true")
parser.add_argument("--out", required=True)
args, app = parse_and_launch(parser)

import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

from driveact.config import load_config  # noqa: E402
from driveact.datasets.hdf5_store import EnvDemoRecorder  # noqa: E402
from driveact.environments.factory import make_env, resolve_layout_seeds  # noqa: E402
from driveact.evaluation.metrics import summarize, write_json  # noqa: E402
from driveact.evaluation.rollout import OUNoise, run_episodes  # noqa: E402
from driveact.experts.expert import ScriptedExpert  # noqa: E402
from driveact.utils.seeding import episode_rng  # noqa: E402


def main():
    cfg = load_config(overrides=args.set)
    seeds = resolve_layout_seeds(args.track, args.split, args.layout_seeds)
    stream = int(cfg.eval.seeds.train_obstacles)
    env = make_env(cfg, args.track, seeds, args.num_envs, backend=args.backend, device=args.device, obstacle_stream=stream)
    frac = float(cfg.expert.noise.episode_fraction if args.noise_fraction is None else args.noise_fraction)
    episodes = list(range(args.first_episode, args.first_episode + args.episodes))
    noisy = {ep: bool(episode_rng(7, args.track, 0, ep).random() < frac) for ep in episodes}
    noise = OUNoise(cfg.expert.noise, env.N, env.device, seed=args.first_episode)
    rec = EnvDemoRecorder(args.out, cfg, args.track, args.backend, env.spec, keep_failures=args.keep_failures, env=env)
    t0 = time.time()
    records = run_episodes(env, ScriptedExpert(cfg), episodes, noise=noise, noisy_flags=noisy, recorder=rec, progress=True)
    rec.close()
    clean = [r for r in records if not r.noisy]
    dist = [r for r in records if r.noisy]
    summary = {
        "track": args.track,
        "backend": args.backend,
        "layout_seeds": seeds,
        "episodes_run": len(records),
        "episodes_written": rec.count,
        "episodes_dropped": rec.dropped,
        "timesteps_written": int(sum(r.steps for r in records if r.success or args.keep_failures)),
        "wall_time_s": time.time() - t0,
        "outcomes": dict(Counter(r.outcome for r in records)),
        "expert_clean": summarize(clean),
        "expert_disturbed": summarize(dist),
        "out": os.path.abspath(args.out),
    }
    write_json(os.path.splitext(args.out)[0] + "_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("episodes_run", "episodes_written", "episodes_dropped", "timesteps_written",
                                               "wall_time_s", "outcomes")}, indent=1))
    env.backend.close()


if __name__ == "__main__":
    run_main(main, app)
