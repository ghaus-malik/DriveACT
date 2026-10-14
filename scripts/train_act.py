"""Train ACT (or the single-step BC baseline) offline on recorded demonstrations. No simulator needed.

    python scripts/train_act.py --data data/track1_train.h5 --out checkpoints/act_track1
    python scripts/train_act.py --data data/track*_train.h5 --policy bc --out checkpoints/bc_all
    python scripts/train_act.py --data /tmp/t1.h5 --out /tmp/ckpt --epochs 3 --device cpu      # smoke test

``--device auto`` uses a CUDA GPU with spare capacity (see ``--min_free_gb``), else Apple MPS, else the CPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def resolve_device(name: str, min_free_gb: float) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        from driveact.utils.gpu import describe, pick_gpu

        idx, gpus = pick_gpu(min_free_gb=min_free_gb)
        if gpus:
            print("GPU status:\n" + describe(gpus))
        if idx is not None:
            return f"cuda:{idx}"
        print("no GPU with spare capacity -> CPU")
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", nargs="+", required=True, help="HDF5 demonstration files (globs allowed)")
    p.add_argument("--policy", choices=("act", "bc"), default="act")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_steps_per_epoch", type=int, default=None, help="cap the batches per epoch (smoke tests)")
    p.add_argument("--device", default="auto")
    p.add_argument("--min_free_gb", type=float, default=4.0)
    p.add_argument("--threads", type=int, default=8, help="CPU threads (shared machine: keep this small)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    import torch

    from driveact.config import load_config
    from driveact.training.trainer import train

    torch.set_num_threads(args.threads)
    paths = sorted({f for pattern in args.data for f in glob.glob(pattern)})
    if not paths:
        sys.exit("no data files found")
    cfg = load_config(overrides=args.set)
    device = resolve_device(args.device, args.min_free_gb)
    print(f"device {device}; data {paths}")
    summary = train(cfg, paths, args.out, kind=args.policy, device=device, epochs=args.epochs,
                    max_steps_per_epoch=args.max_steps_per_epoch)
    print(json.dumps({k: summary[k] for k in ("parameters", "epochs", "steps", "best_val_l1_prior", "time_s")}, indent=1))


if __name__ == "__main__":
    main()
