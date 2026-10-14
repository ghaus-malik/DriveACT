"""Inspect the software environment, GPU availability, remote assets and the track configuration.

    python scripts/check_environment.py            # no simulator start, safe on a shared machine
    python scripts/check_environment.py --assets   # also check that the Isaac Sim cloud assets are reachable

Prints Python / PyTorch / CUDA / Isaac Sim / Isaac Lab versions, per-GPU memory and utilisation (and which GPU
``--gpu auto`` would pick), builds every track layout of every split and samples obstacle configurations.
Nothing is written.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
REMOTE_ASSETS = ["Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                 "Isaac/Environments/Grid/default_environment.usd"]


def version(module: str) -> str:
    try:
        m = importlib.import_module(module)
        return str(getattr(m, "__version__", "installed"))
    except Exception as e:  # noqa: BLE001
        return f"not available ({type(e).__name__})"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--assets", action="store_true")
    args = p.parse_args()

    import numpy as np
    import torch

    print(f"python      {platform.python_version()} ({sys.executable})")
    print(f"torch       {torch.__version__}, CUDA available: {torch.cuda.is_available()}, "
          f"CUDA {torch.version.cuda}, MPS: {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
    print(f"numpy       {np.__version__}")
    for mod in ("h5py", "yaml", "matplotlib", "cv2", "imageio", "isaacsim", "isaaclab", "pxr"):
        print(f"{mod:<11} {version(mod)}")
    try:
        from importlib.metadata import version as dist_version

        print(f"isaacsim    {dist_version('isaacsim')} (pip)")
        print(f"isaaclab    {dist_version('isaaclab')} (pip)")
    except Exception:  # noqa: BLE001
        pass

    from driveact.utils.gpu import describe, pick_gpu

    idx, gpus = pick_gpu()
    print("GPUs (utilisation averaged over 3 s):\n" + describe(gpus))
    if gpus:
        print(f"--gpu auto would use: {idx if idx is not None else 'none (no GPU has capacity)'}")

    if args.assets:
        import urllib.request

        for a in REMOTE_ASSETS:
            try:
                req = urllib.request.Request(f"{ASSET_ROOT}/{a}", method="HEAD")
                with urllib.request.urlopen(req, timeout=15) as r:
                    print(f"asset ok    {a} ({r.status})")
            except Exception as e:  # noqa: BLE001
                print(f"asset FAIL  {a}: {e}")

    from driveact.config import load_config, load_track_config
    from driveact.tracks.generators import make_layout, split_seeds
    from driveact.tracks.obstacles import sample_obstacles
    from driveact.utils.seeding import episode_rng

    cfg = load_config()
    for track in (1, 2, 3, 4):
        tc = load_track_config(track)
        t0 = time.time()
        lengths, groups = [], []
        for split in ("train", "test"):
            for seed in split_seeds(track, split):
                layout = make_layout(track, seed, cfg)
                lengths.append(layout.goal_s - layout.start_s)
                for ep in range(3):
                    groups.append(len(sample_obstacles(layout, tc.obstacles, cfg, episode_rng(0, track, seed, ep)).groups))
        print(f"track {track}: {len(lengths)} layouts, start->goal {min(lengths):.0f}-{max(lengths):.0f} m, "
              f"obstacle groups per episode {min(groups)}-{max(groups)} ({time.time() - t0:.1f} s)")
    print("environment check finished")


if __name__ == "__main__":
    main()
