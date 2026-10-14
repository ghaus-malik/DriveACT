"""Action-chunk sampling for ACT (and single-step samples for the BC baseline), fully vectorised on one device.

A sample is (obs[t], actions[t : t + k], is_pad) for a uniformly drawn timestep t of a training episode. Chunks
that run past the end of the episode are zero-padded and masked (``is_pad``), as in the reference ACT
implementation. Observations and actions are z-normalised with statistics of the training episodes (actions: std
floor 0.01 as in ACT; observations: std floor ``obs_std_floor`` so that rarely-active one-hot features are not
blown up).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Normalizer:
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    act_mean: torch.Tensor
    act_std: torch.Tensor

    @staticmethod
    def fit(episodes: list[dict], obs_std_floor: float = 0.1, act_std_floor: float = 0.01) -> "Normalizer":
        obs = np.concatenate([e["obs"] for e in episodes], axis=0)
        act = np.concatenate([e["actions"] for e in episodes], axis=0)
        return Normalizer(
            torch.as_tensor(obs.mean(0), dtype=torch.float32),
            torch.as_tensor(np.maximum(obs.std(0), obs_std_floor), dtype=torch.float32),
            torch.as_tensor(act.mean(0), dtype=torch.float32),
            torch.as_tensor(np.maximum(act.std(0), act_std_floor), dtype=torch.float32),
        )

    def to(self, device) -> "Normalizer":
        return Normalizer(*(t.to(device) for t in (self.obs_mean, self.obs_std, self.act_mean, self.act_std)))

    def norm_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def norm_act(self, act):
        return (act - self.act_mean) / self.act_std

    def denorm_act(self, act):
        return act * self.act_std + self.act_mean

    def state_dict(self) -> dict:
        return {k: getattr(self, k).cpu() for k in ("obs_mean", "obs_std", "act_mean", "act_std")}

    @staticmethod
    def from_state_dict(d: dict) -> "Normalizer":
        return Normalizer(d["obs_mean"], d["obs_std"], d["act_mean"], d["act_std"])


def split_episodes(episodes: list[dict], val_fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split by episode (never by timestep), deterministic."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = int(round(val_fraction * len(episodes)))
    if len(episodes) > 1:
        n_val = min(max(n_val, 1 if val_fraction > 0 else 0), len(episodes) - 1)
    val = [episodes[i] for i in order[:n_val]]
    train = [episodes[i] for i in order[n_val:]]
    return train, val


class ChunkSampler:
    """All timesteps of a set of episodes, stacked; draws normalised (obs, chunk, is_pad) batches."""

    def __init__(self, episodes: list[dict], normalizer: Normalizer, chunk_size: int, device):
        if not episodes:
            raise ValueError("no episodes")
        self.device = torch.device(device)
        self.k = int(chunk_size)
        lengths = np.array([len(e["obs"]) for e in episodes])
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        self.obs = normalizer.norm_obs(torch.as_tensor(np.concatenate([e["obs"] for e in episodes]))).to(self.device)
        act = normalizer.norm_act(torch.as_tensor(np.concatenate([e["actions"] for e in episodes])))
        # one zero row at the end so that padded gathers stay in bounds
        self.actions = torch.cat((act, torch.zeros(1, act.shape[1]))).to(self.device)
        self.end = torch.as_tensor(np.repeat(starts + lengths, lengths), device=self.device)  # exclusive end per step
        self.num_steps = int(lengths.sum())
        self.num_episodes = len(episodes)
        self.offsets = torch.arange(self.k, device=self.device)

    def __len__(self) -> int:
        return self.num_steps

    def gather(self, idx: torch.Tensor):
        t = idx[:, None] + self.offsets[None, :]
        is_pad = t >= self.end[idx][:, None]
        t = torch.where(is_pad, torch.full_like(t, self.num_steps), t)  # points at the zero row
        return self.obs[idx], self.actions[t], is_pad

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        idx = torch.randint(0, self.num_steps, (batch_size,), generator=generator, device="cpu").to(self.device)
        return self.gather(idx)

    def batches(self, batch_size: int):
        """Deterministic sweep over all timesteps (validation)."""
        for i in range(0, self.num_steps, batch_size):
            yield self.gather(torch.arange(i, min(i + batch_size, self.num_steps), device=self.device))
