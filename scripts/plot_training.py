"""Training curves (per epoch) of one or more runs: L1 with the CVAE posterior, L1 with z = 0, and KL.

    python scripts/plot_training.py --runs checkpoints/act_s4_track1 checkpoints/bc_s4_track1 --out results/stage4/training.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", required=True, help="training output folders (containing last.pt)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for run in args.runs:
        ckpt = torch.load(os.path.join(run, "last.pt"), map_location="cpu", weights_only=False)
        h = ckpt["history"]
        ep = [r["epoch"] for r in h]
        name = os.path.basename(os.path.normpath(run))
        line = ax[0].plot(ep, [r["train_l1"] for r in h], "--", label=f"{name} train")[0]
        if "val_l1_prior" in h[0]:
            ax[0].plot(ep, [r["val_l1_prior"] for r in h], color=line.get_color(), label=f"{name} val (z = 0)")
        if ckpt["kind"] == "act":
            ax[1].plot(ep, [r["train_kl"] for r in h], "--", color=line.get_color(), label=f"{name} train")
            if "val_kl" in h[0]:
                ax[1].plot(ep, [r["val_kl"] for r in h], color=line.get_color(), label=f"{name} val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("L1 (normalised actions)")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("KL")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
