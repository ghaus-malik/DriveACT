"""Visual demonstration mode: render an episode from top-down, front and chase cameras and write videos.

    python scripts/visualize.py --policy act --checkpoint checkpoints/act_track4/best.pt --track 4 --layout_seeds -1 \
        --episode 0 --out videos/act_track4_showcase
    python scripts/visualize.py --policy expert --track 2 --episode 3 --out videos/expert_track2
    python scripts/visualize.py --policy expert --track 4 --layout_seeds -1 --still --out results/screens/track4

Not used for training. Uses the same layouts, obstacle sampling and vehicle physics as headless runs (the visual
mode only adds lights, decoration behind the walls and cameras). Writes ``composite.mp4`` (chase view with front
and top insets), one MP4 per view, PNG screenshots and ``episode.json``. Requires RTX rendering (GPU).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from driveact.utils.launch import base_parser, parse_and_launch, run_main  # noqa: E402

parser = base_parser(__doc__.splitlines()[0])
parser.add_argument("--policy", choices=("expert", "act", "bc"), default="expert")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--track", type=int, required=True, choices=(1, 2, 3, 4))
parser.add_argument("--split", default="test")
parser.add_argument("--layout_seeds", type=int, nargs="*", default=None)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--stream", type=int, default=None)
parser.add_argument("--replan_every", type=int, default=None)
parser.add_argument("--temporal_ensemble", action="store_true")
parser.add_argument("--top", choices=("follow", "overview"), default="follow")
parser.add_argument("--show_reference", action="store_true", help="draw the expert reference path (debug)")
parser.add_argument("--no_decor", action="store_true")
parser.add_argument("--still", action="store_true", help="only render screenshots at the start pose")
parser.add_argument("--out", required=True)
args, app = parse_and_launch(parser, enable_cameras=True)

import json  # noqa: E402
from dataclasses import asdict  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from driveact.config import load_config  # noqa: E402
from driveact.environments.factory import make_env, resolve_layout_seeds  # noqa: E402
from driveact.evaluation.rollout import run_episodes  # noqa: E402
from driveact.experts.expert import ScriptedExpert  # noqa: E402
from driveact.visualization.overlay import VideoWriter, compose, draw_hud  # noqa: E402
from driveact.visualization.predicted_path import rollout_chunk  # noqa: E402


def main():
    if args.backend != "isaac":
        sys.exit("visualize.py needs --backend isaac")
    cfg = load_config(overrides=args.set)
    seeds = resolve_layout_seeds(args.track, args.split, args.layout_seeds)[:1]
    stream = int(cfg.eval.seeds.eval_obstacles if args.stream is None else args.stream)
    env = make_env(cfg, args.track, seeds, 1, backend="isaac", device=args.device, obstacle_stream=stream, visual=True,
                   decor=bool(cfg.visual.decor) and not args.no_decor)
    backend = env.backend
    rig = backend.cameras
    rig.top_mode = args.top
    os.makedirs(args.out, exist_ok=True)

    if args.policy == "expert":
        ctrl = ScriptedExpert(cfg)
        label = "scripted expert"
    else:
        from driveact.models.controller import PolicyController
        from driveact.training.trainer import load_checkpoint

        model, norm, ckpt = load_checkpoint(args.checkpoint, env.device)
        inf = cfg.act.inference
        ctrl = PolicyController(model, norm, 1, env.device,
                                replan_every=int(args.replan_every if args.replan_every is not None else inf.replan_every),
                                temporal_ensemble=bool(args.temporal_ensemble or inf.temporal_ensemble),
                                ensemble_k=float(inf.ensemble_k))
        label = f"{args.policy.upper()} (chunk {ctrl.k}, replan {ctrl.replan_every}{', TE' if ctrl.temporal_ensemble else ''})"
    title = f"Track {args.track} | {label}"

    if args.still:
        env.reset([0], [args.episode])
        for _ in range(3):  # a few physics steps at rest so that rendered transforms are up to date
            env.step(torch.zeros(1, 2, device=env.device))
        rig.update_views(force=True)
        for _ in range(5):
            backend.sim.render()
        frames = rig.capture()
        for name, img in frames.items():
            imageio.imwrite(os.path.join(args.out, f"{name}.png"), img)
        rig.top_mode = "overview"
        rig.update_views(force=True)
        for _ in range(5):
            backend.sim.render()
        imageio.imwrite(os.path.join(args.out, "overview.png"), rig.capture()["top"])
        print(f"screenshots written to {args.out}")
        return

    fps = int(cfg.visual.fps)
    writers = {name: VideoWriter(os.path.join(args.out, f"{name}.mp4"), fps) for name in ("composite", *rig.VIEWS)}
    trail: list[np.ndarray] = []
    state = {"pred": None, "frame": 0, "shots": []}
    legend = "green: ACT chunk rolled out with a kinematic bicycle model | orange: driven path" if args.policy == "act" else \
        "orange: driven path"

    def on_step(env_, obs, actions):
        st = env_.state
        trail.append(st["pos"][0].cpu().numpy())
        pred = None
        if args.policy == "act" and cfg.visual.overlay.show_prediction:
            path = rollout_chunk(ctrl.last_chunk[:1], st["pos"][:1], st["yaw"][:1], st["v_long"][:1], st["steer"][:1],
                                 env_.g, env_.dt)
            pred = path[0].cpu().numpy()
        ref = None
        if args.show_reference:
            s_ahead = (env_.s[:1, None] + torch.arange(0, 40, 2.0, device=env_.device)[None]).clamp(max=env_.route.length[:1, None])
            ref = env_.route.point_at(s_ahead, env_.route.interp(env_.offsets[:1], s_ahead))[0].cpu().numpy()
        n_trail = int(cfg.visual.overlay.trail_length)
        rig.set_markers(pred, ref, np.asarray(trail[-n_trail:][::2]))

    def on_render(be):
        rig.update_views()
        frames = rig.capture()
        st = env.state
        outcome = int(env.outcome[0])
        span = max(1e-6, float(env.route.goal_s[0] - env.route.start_s[0]))
        hud = {
            "title": title,
            "speed": float(st["v_long"][0]) if st else 0.0,
            "steer_cmd": float(env.prev_action[0, 0]),
            "accel_cmd": float(env.prev_action[0, 1]),
            "progress": float(np.clip((float(env.s[0]) - float(env.route.start_s[0])) / span, 0, 1)),
            "collision": 2 <= outcome <= 5,
            "success": outcome == 1,
            "status": {1: "GOAL REACHED"}.get(outcome, ""),
            "legend": legend,
        }
        main = draw_hud(frames["chase"], hud)
        writers["composite"].add(compose(main, [frames["front"], frames["top"]]))
        for name in rig.VIEWS:
            writers[name].add(draw_hud(frames[name], hud) if name != "chase" else main)
        state["frame"] += 1
        if state["frame"] in (1, 150, 400, 800):
            for name in rig.VIEWS:
                imageio.imwrite(os.path.join(args.out, f"{name}_frame{state['frame']:05d}.png"), frames[name])

    backend.render_hooks.append(on_render)
    records = run_episodes(env, ctrl, [args.episode], trace=True, on_step=on_step, progress=True)
    # a short hold on the final state
    for _ in range(fps):
        backend.sim.render()
        on_render(backend)
    for w in writers.values():
        w.close()
    rec = records[0]
    d = asdict(rec)
    d["extra"] = {k: v for k, v in rec.extra.items() if k != "trace"}
    d["policy"] = args.policy
    d["checkpoint"] = args.checkpoint
    with open(os.path.join(args.out, "episode.json"), "w") as f:
        json.dump(d, f, indent=1, default=float)
    print(f"{rec.outcome}: {rec.time_s:.1f} s, videos in {args.out}")


if __name__ == "__main__":
    run_main(main, app)
