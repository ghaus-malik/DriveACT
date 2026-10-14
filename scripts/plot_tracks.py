"""Top-down PNGs of track layouts with an example obstacle configuration and the expert's reference path.

    python scripts/plot_tracks.py --out results/tracks            # canonical layout of every track
    python scripts/plot_tracks.py --track 4 --split test --out results/tracks/track4_test

No simulator needed.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--track", type=int, nargs="*", default=[1, 2, 3, 4])
    p.add_argument("--split", default=None, help="plot every layout of this split instead of the canonical one")
    p.add_argument("--episode", type=int, default=0, help="obstacle configuration (evaluation stream)")
    p.add_argument("--out", default="results/tracks")
    args = p.parse_args()

    from driveact.config import load_config, load_track_config
    from driveact.experts.planner import plan_lateral_offsets
    from driveact.tracks.generators import SHOWCASE_SEED, make_layout, split_seeds
    from driveact.tracks.obstacles import sample_obstacles
    from driveact.utils.seeding import episode_rng
    from driveact.visualization.topdown import plot_layout_png

    cfg = load_config()
    os.makedirs(args.out, exist_ok=True)
    stream = int(cfg.eval.seeds.eval_obstacles)
    for track in args.track:
        tc = load_track_config(track)
        seeds = split_seeds(track, args.split) if args.split else [SHOWCASE_SEED if track == 4 else 0]
        for seed in seeds:
            layout = make_layout(track, seed, cfg)
            obs = sample_obstacles(layout, tc.obstacles, cfg, episode_rng(stream, track, seed, args.episode))
            offsets = plan_lateral_offsets(layout, obs, cfg)
            name = f"track{track}_layout{'showcase' if seed == SHOWCASE_SEED else seed}.png"
            title = f"{layout.name} | layout {seed if seed != SHOWCASE_SEED else 'showcase'} | " \
                    f"{layout.goal_s - layout.start_s:.0f} m, {len(obs.groups)} obstacle groups"
            plot_layout_png(os.path.join(args.out, name), layout, obs, offsets=offsets, title=title)
            print("wrote", os.path.join(args.out, name))


if __name__ == "__main__":
    main()
