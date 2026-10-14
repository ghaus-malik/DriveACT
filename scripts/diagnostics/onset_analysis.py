"""Open-loop check of a trained policy around the onset of the expert's avoidance manoeuvres.

    python scripts/diagnostics/onset_analysis.py --data data/track2_train.h5 --checkpoint checkpoints/act_s5_track12/best.pt

For undisturbed demonstration episodes, aligns time on the first step where the expert's lateral offset exceeds
0.3 m and prints the mean expert steering label, the policy's prediction, and the prediction with the
previous-command features zeroed (to see how much the policy relies on them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--before", type=int, default=30)
    p.add_argument("--after", type=int, default=6)
    args = p.parse_args()
    import numpy as np
    import torch

    from driveact.datasets.hdf5_store import load_episodes
    from driveact.environments.observations import ObsSpec
    from driveact.training.trainer import load_checkpoint

    eps, attrs = load_episodes(args.data)
    spec = ObsSpec.from_dict(json.loads(attrs["obs_spec"]))
    model, norm, _ = load_checkpoint(args.checkpoint)
    s0 = spec.state.start
    rows = []
    for ep in [e for e in eps if not e["noisy"]][: args.episodes]:
        lat = ep["state"][:, 6]
        if not (lat > 0.3).any():
            continue
        on = int(np.argmax(lat > 0.3))
        if on < args.before:
            continue
        w = slice(on - args.before, on + args.after)
        o = torch.tensor(ep["obs"][w])
        with torch.no_grad():
            pr = norm.denorm_act(model.predict(norm.norm_obs(o))).numpy()
            o2 = o.clone()
            o2[:, s0 + 3] = 0.0
            o2[:, s0 + 4] = 0.0
            pr2 = norm.denorm_act(model.predict(norm.norm_obs(o2))).numpy()
        front = ep["obs"][w][:, spec.lidar.start + spec.num_sectors // 2 - 3: spec.lidar.start + spec.num_sectors // 2 + 3].min(1) * 40
        rows.append((ep["actions"][w, 0], pr[:, 0, 0], pr2[:, 0, 0], front))
    L, P, P2, F = (np.array([r[i] for r in rows]) for i in range(4))
    print(f"episodes {len(rows)}")
    for k in range(0, args.before + args.after, 2):
        print(f"t-onset={k - args.before:+3d} front={F[:, k].mean():5.1f} m  label={L[:, k].mean():+.3f}  pred={P[:, k].mean():+.3f}  "
              f"pred_without_prev={P2[:, k].mean():+.3f}  |label-pred|={np.abs(L[:, k] - P[:, k]).mean():.3f}")


if __name__ == "__main__":
    main()
