"""Closed-loop execution of a trained policy in N environments.

ACT (chunk size k):
  * receding horizon (default): predict a chunk, execute ``replan_every`` actions from it, re-observe, predict again;
  * temporal ensembling (optional, as in the ACT paper): predict every step and average all predictions for the
    current step with weights w_i = exp(-m * i), i = 0 being the OLDEST prediction (reference implementation).
BC: one prediction per step.

Actions are returned in the normalised action space [-1, 1]. ``last_chunk`` keeps the most recent predicted chunk
(unnormalised) for visualisation.
"""

from __future__ import annotations

import torch

from driveact.datasets.chunk_dataset import Normalizer


class PolicyController:
    def __init__(self, model, normalizer: Normalizer, num_envs: int, device, replan_every: int = 1,
                 temporal_ensemble: bool = False, ensemble_k: float = 0.01):
        self.model = model.to(device).eval()
        self.norm = normalizer.to(device)
        self.device = torch.device(device)
        self.N = num_envs
        self.k = int(getattr(model, "chunk_size", 1))
        self.name = getattr(model, "kind", "policy")
        self.replan_every = max(1, min(int(replan_every), self.k))
        self.temporal_ensemble = bool(temporal_ensemble) and self.k > 1
        self.m = float(ensemble_k)
        A = 2
        self.chunk = torch.zeros(num_envs, self.k, A, device=self.device)
        self.since_plan = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.history = torch.zeros(num_envs, self.k, self.k, A, device=self.device)  # [n, j] = chunk from j steps ago
        self.age = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # steps since reset
        self.last_chunk = torch.zeros(num_envs, self.k, A, device=self.device)

    def reset(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.since_plan[env_ids] = 0
        self.age[env_ids] = 0
        self.history[env_ids] = 0.0
        self.chunk[env_ids] = 0.0

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Unnormalised (but clipped) action chunks (N, k, 2)."""
        pred = self.model.predict(self.norm.norm_obs(obs))
        return self.norm.denorm_act(pred).clamp(-1.0, 1.0)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, env=None) -> torch.Tensor:
        if self.temporal_ensemble:
            chunk = self.predict(obs)
            self.last_chunk = chunk
            self.history = torch.roll(self.history, shifts=1, dims=1)
            self.history[:, 0] = chunk
            j = torch.arange(self.k, device=self.device)
            preds = self.history[:, j, j]  # (N, k, 2): action for "now" from the chunk predicted j steps ago
            n_avail = torch.minimum(self.age + 1, torch.full_like(self.age, self.k))  # (N,)
            valid = j[None, :] < n_avail[:, None]
            age_rank = (n_avail[:, None] - 1 - j[None, :]).clamp(min=0)  # 0 = oldest available prediction
            w = torch.exp(-self.m * age_rank.float()) * valid.float()
            action = (w[..., None] * preds).sum(1) / w.sum(1, keepdim=True)
            self.age += 1
            return action
        need = self.since_plan % self.replan_every == 0
        if bool(need.any()):
            chunk = self.predict(obs)
            self.chunk = torch.where(need[:, None, None], chunk, self.chunk)
            self.last_chunk = torch.where(need[:, None, None], chunk, self.last_chunk)
            self.since_plan = torch.where(need, torch.zeros_like(self.since_plan), self.since_plan)
        idx = self.since_plan.clamp(max=self.k - 1)
        action = self.chunk[torch.arange(self.N, device=self.device), idx]
        self.since_plan += 1
        self.age += 1
        return action
