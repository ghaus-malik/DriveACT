"""Offline training of ACT (or the single-step BC baseline) on recorded demonstrations.

One epoch = ceil(#training timesteps / batch_size) random batches. After every epoch the validation episodes are
evaluated twice: with the CVAE posterior (training objective) and with the prior latent z = 0 (what is deployed).
The checkpoint with the lowest prior-latent validation L1 is kept as ``best.pt``; ``last.pt`` is written every epoch.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time

import torch

from driveact.datasets.chunk_dataset import ChunkSampler, Normalizer, split_episodes
from driveact.datasets.hdf5_store import load_episodes
from driveact.environments.observations import ObsSpec
from driveact.models.act_policy import ACTPolicy, masked_l1
from driveact.models.bc_policy import BCPolicy
from driveact.models.transformer import count_parameters
from driveact.utils.seeding import seed_everything


def build_model(kind: str, spec: ObsSpec, cfg):
    if kind == "act":
        return ACTPolicy(spec, cfg.act.model)
    if kind == "bc":
        return BCPolicy(spec, cfg.act.model, cfg.act.bc)
    raise ValueError(f"unknown policy kind '{kind}'")


def save_checkpoint(path: str, model, normalizer: Normalizer, spec: ObsSpec, cfg, kind: str, extra: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "kind": kind,
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "obs_spec": spec.to_dict(),
        "config": cfg.to_dict(),
        **extra,
    }, path)


def load_checkpoint(path: str, device="cpu"):
    """-> (model, normalizer, ckpt dict). The model is rebuilt from the configuration stored in the checkpoint."""
    from driveact.config import _wrap

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = _wrap(ckpt["config"])
    spec = ObsSpec.from_dict(ckpt["obs_spec"])
    model = build_model(ckpt["kind"], spec, cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), Normalizer.from_state_dict(ckpt["normalizer"]), ckpt


def augment_prev_actions(obs: torch.Tensor, spec: ObsSpec, dropout: float, noise: float) -> torch.Tensor:
    """Training-time perturbation of the previous-command features (normalised obs): replaced by 0 (their mean)
    with probability ``dropout``, otherwise Gaussian noise of std ``noise`` is added."""
    if dropout <= 0 and noise <= 0:
        return obs
    obs = obs.clone()
    sl = spec.prev_actions
    prev = obs[:, sl]
    if noise > 0:
        prev = prev + noise * torch.randn_like(prev)
    if dropout > 0:
        drop = torch.rand(prev.shape[0], 1, device=obs.device) < dropout
        prev = torch.where(drop, torch.zeros_like(prev), prev)
    obs[:, sl] = prev
    return obs


@torch.no_grad()
def evaluate_loss(model, sampler: ChunkSampler, kl_weight: float, batch_size: int) -> dict:
    model.eval()
    tot = {"l1": 0.0, "kl": 0.0, "l1_prior": 0.0}
    n = 0
    for obs, act, is_pad in sampler.batches(batch_size):
        out = model.compute_loss(obs, act, is_pad, kl_weight)
        pred = model.predict(obs)
        b = obs.shape[0]
        tot["l1"] += float(out["l1"]) * b
        tot["kl"] += float(out["kl"]) * b
        tot["l1_prior"] += float(masked_l1(pred, act[:, : pred.shape[1]], is_pad[:, : pred.shape[1]])) * b
        n += b
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


class _Logger:
    def __init__(self, out_dir: str, tensorboard: bool = True):
        self.csv_path = os.path.join(out_dir, "train_log.csv")
        self.rows = []
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(os.path.join(out_dir, "tb"))
            except Exception:  # tensorboard is optional
                self.tb = None

    def log(self, step: int, values: dict) -> None:
        row = {"step": step, **{k: float(v) for k, v in values.items()}}
        self.rows.append(row)
        if self.tb is not None:
            for k, v in values.items():
                self.tb.add_scalar(k, float(v), step)

    def close(self) -> None:
        keys = sorted({k for r in self.rows for k in r}, key=lambda k: (k != "step", k))
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        if self.tb is not None:
            self.tb.close()


def train(cfg, data_paths: list[str], out_dir: str, kind: str = "act", device: str = "cpu", epochs: int | None = None,
          max_steps_per_epoch: int | None = None, verbose: bool = True) -> dict:
    tc = cfg.act.train
    seed_everything(int(tc.seed))
    os.makedirs(out_dir, exist_ok=True)
    episodes, attrs = load_episodes(data_paths)
    spec = ObsSpec.from_dict(json.loads(attrs["obs_spec"]))
    train_eps, val_eps = split_episodes(episodes, float(tc.val_fraction), int(tc.seed))
    norm = Normalizer.fit(train_eps)
    model = build_model(kind, spec, cfg).to(device)
    k = int(cfg.act.model.chunk_size) if kind == "act" else 1
    train_s = ChunkSampler(train_eps, norm, k, device)
    val_s = ChunkSampler(val_eps, norm, k, device) if val_eps else None
    B = int(tc.batch_size)
    epochs = int(epochs if epochs is not None else tc.epochs)
    steps_per_epoch = math.ceil(len(train_s) / B)
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, int(max_steps_per_epoch))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
    warmup = int(tc.warmup_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    kl_weight = float(tc.kl_weight) if kind == "act" else 0.0
    use_amp = bool(tc.amp) and str(device).startswith("cuda")
    gen = torch.Generator().manual_seed(int(tc.seed))
    logger = _Logger(out_dir, bool(tc.get("tensorboard", True)))
    info = {
        "kind": kind,
        "parameters": count_parameters(model),
        "train_episodes": len(train_eps),
        "val_episodes": len(val_eps),
        "train_steps_per_epoch": steps_per_epoch,
        "train_timesteps": len(train_s),
        "data": [os.path.abspath(p) for p in data_paths],
        "chunk_size": k,
    }
    if verbose:
        print(json.dumps(info, indent=1), flush=True)
    best = math.inf
    step = 0
    t0 = time.time()
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        run = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
        for _ in range(steps_per_epoch):
            obs, act, is_pad = train_s.sample(B, gen)
            obs = augment_prev_actions(obs, spec, float(tc.get("prev_action_dropout", 0.0)),
                                       float(tc.get("prev_action_noise", 0.0)))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model.compute_loss(obs, act, is_pad, kl_weight)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.grad_clip))
            opt.step()
            sched.step()
            step += 1
            for key in run:
                run[key] += out[key].item()
            if step % int(tc.log_every) == 0:
                logger.log(step, {"train/loss": out["loss"], "train/l1": out["l1"], "train/kl": out["kl"],
                                  "train/grad_norm": gn, "train/lr": sched.get_last_lr()[0]})
        run = {k2: v / steps_per_epoch for k2, v in run.items()}
        val = evaluate_loss(model, val_s, kl_weight, 4 * B) if val_s is not None else {}
        rec = {"epoch": epoch, "step": step, "time_s": time.time() - t0, **{f"train_{k2}": v for k2, v in run.items()},
               **{f"val_{k2}": v for k2, v in val.items()}}
        history.append(rec)
        logger.log(step, {f"epoch/{k2}": v for k2, v in rec.items() if k2 not in ("epoch", "step")})
        extra = {"epoch": epoch, "step": step, "history": history, "info": info}
        save_checkpoint(os.path.join(out_dir, "last.pt"), model, norm, spec, cfg, kind, extra)
        score = val.get("l1_prior", run["l1"])
        if score < best:
            best = score
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, norm, spec, cfg, kind, extra)
        if epoch % int(tc.save_every_epochs) == 0:
            save_checkpoint(os.path.join(out_dir, f"epoch_{epoch:04d}.pt"), model, norm, spec, cfg, kind, extra)
        if verbose:
            print(f"epoch {epoch:3d} step {step:6d} | train L1 {run['l1']:.4f} KL {run['kl']:.4f} | "
                  f"val L1 {val.get('l1', float('nan')):.4f} KL {val.get('kl', float('nan')):.4f} "
                  f"L1(z=0) {val.get('l1_prior', float('nan')):.4f} | {time.time() - t0:.0f}s", flush=True)
    logger.close()
    summary = {**info, "epochs": epochs, "steps": step, "best_val_l1_prior": best, "time_s": time.time() - t0,
               "final": history[-1] if history else {}}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
