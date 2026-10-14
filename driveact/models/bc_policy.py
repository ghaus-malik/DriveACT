"""Single-step behaviour-cloning baseline: same observation (and LiDAR stem) as ACT, an MLP head, one action.

No action chunking, no CVAE, no Transformer; trained with L1 on the next expert action only.
"""

from __future__ import annotations

import torch
from torch import nn

from driveact.environments.observations import ObsSpec
from driveact.models.act_policy import mask_prev_actions
from driveact.models.encoders import LidarEncoder, mlp


class BCPolicy(nn.Module):
    kind = "bc"
    chunk_size = 1

    def __init__(self, spec: ObsSpec, mcfg, bccfg, action_dim: int = 2):
        super().__init__()
        self.spec = spec
        self.use_prev_actions = bool(mcfg.get("use_prev_actions", True))
        self.lidar = LidarEncoder(spec.num_sectors, int(mcfg.lidar_tokens), mcfg.lidar_channels, d=1)
        lidar_feat = self.lidar.out_channels * int(mcfg.lidar_tokens)
        h = int(bccfg.hidden_dim)
        self.head = mlp([lidar_feat + spec.state_dim + spec.route_dim] + [h] * int(bccfg.mlp_layers) + [action_dim],
                        dropout=float(bccfg.dropout))
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = mask_prev_actions(obs, self.spec, self.use_prev_actions)
        feat = self.lidar.features(obs[:, self.spec.lidar]).flatten(1)
        return self.head(torch.cat((feat, obs[:, self.spec.state], obs[:, self.spec.route]), dim=-1))

    def compute_loss(self, obs, actions, is_pad, kl_weight: float = 0.0) -> dict:
        l1 = (self(obs) - actions[:, 0]).abs().mean()
        return {"loss": l1, "l1": l1, "kl": torch.zeros((), device=obs.device)}

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)[:, None]  # (B, 1, 2)
