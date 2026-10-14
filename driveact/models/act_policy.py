"""Action Chunking with Transformers (ACT) for continuous vehicle control.

Architecture (Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023),
re-implemented for low-dimensional driving observations:

  CVAE encoder (training only)   [CLS, state, a_t..a_{t+k-1}] -> mu, logvar -> z ~ N(mu, sigma^2)
  Conditioning tokens            [proj(z), proj(state), MLP(route), LiDAR tokens (1-D conv)]
                                 + learned positions (z, state, route) and sinusoidal positions (LiDAR azimuth)
  Transformer encoder            self-attention over the conditioning tokens
  Transformer decoder            k learned action queries cross-attend to the encoder memory
  Action head                    Linear -> (B, k, 2) normalised [steering, longitudinal acceleration]

Loss: masked L1 between predicted and expert chunks + kl_weight * KL(q(z | state, chunk) || N(0, I)).
Inference: z = 0 (prior mean); the CVAE encoder is not used.

Inputs are the same "state" (5-D proprioception: speed, steering angle, yaw rate, previous commands) that ACT feeds
to its CVAE encoder as joint positions; images are replaced by LiDAR sectors and route features.
"""

from __future__ import annotations

import torch
from torch import nn

from driveact.environments.observations import ObsSpec
from driveact.models.cvae import CVAEEncoder, kl_divergence, reparameterize
from driveact.models.encoders import LidarEncoder, mlp
from driveact.models.transformer import Transformer, sinusoid_table


class ACTPolicy(nn.Module):
    kind = "act"

    def __init__(self, spec: ObsSpec, mcfg, action_dim: int = 2):
        super().__init__()
        d = int(mcfg.hidden_dim)
        self.spec = spec
        self.use_prev_actions = bool(mcfg.get("use_prev_actions", True))
        self.chunk_size = int(mcfg.chunk_size)
        self.latent_dim = int(mcfg.latent_dim)
        self.action_dim = action_dim
        dropout = float(mcfg.dropout)
        self.lidar = LidarEncoder(spec.num_sectors, int(mcfg.lidar_tokens), mcfg.lidar_channels, d)
        self.state_proj = nn.Linear(spec.state_dim, d)
        self.route_enc = mlp([spec.route_dim, d, d])
        self.latent_proj = nn.Linear(self.latent_dim, d)
        self.extra_pos = nn.Embedding(3, d)  # positions of the latent, state and route tokens
        self.register_buffer("lidar_pos", sinusoid_table(int(mcfg.lidar_tokens), d), persistent=False)
        self.transformer = Transformer(d, int(mcfg.nheads), int(mcfg.dim_feedforward), int(mcfg.enc_layers),
                                       int(mcfg.dec_layers), dropout)
        self.query_embed = nn.Embedding(self.chunk_size, d)
        self.action_head = nn.Linear(d, action_dim)
        self.cvae = CVAEEncoder(spec.state_dim, action_dim, self.chunk_size, d, int(mcfg.nheads),
                                int(mcfg.dim_feedforward), int(mcfg.cvae_layers), self.latent_dim, dropout)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None, is_pad: torch.Tensor | None = None):
        """obs (B, D) normalised; actions (B, k, 2) normalised (training) -> (a_hat (B, k, 2), mu, logvar)."""
        B = obs.shape[0]
        obs = mask_prev_actions(obs, self.spec, self.use_prev_actions)
        state = obs[:, self.spec.state]
        if actions is not None:
            if is_pad is None:
                is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=obs.device)
            mu, logvar = self.cvae(state, actions, is_pad)
            z = reparameterize(mu, logvar)
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=obs.device, dtype=obs.dtype)
        tokens = torch.cat(
            (
                self.latent_proj(z)[:, None],
                self.state_proj(state)[:, None],
                self.route_enc(obs[:, self.spec.route])[:, None],
                self.lidar(obs[:, self.spec.lidar]),
            ),
            dim=1,
        )
        pos = torch.cat((self.extra_pos.weight, self.lidar_pos[0]), dim=0)[None].expand(B, -1, -1)
        hs = self.transformer(tokens, pos, self.query_embed.weight)
        return self.action_head(hs), mu, logvar

    def compute_loss(self, obs, actions, is_pad, kl_weight: float) -> dict:
        a_hat, mu, logvar = self(obs, actions, is_pad)
        l1 = masked_l1(a_hat, actions, is_pad)
        kl = kl_divergence(mu, logvar)
        return {"loss": l1 + kl_weight * kl, "l1": l1, "kl": kl}

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalised action chunk (B, k, 2) with the prior latent z = 0."""
        return self(obs)[0]


def mask_prev_actions(obs: torch.Tensor, spec: ObsSpec, use: bool) -> torch.Tensor:
    """Ablation: zero the (normalised) previous-command features when they must not be used."""
    if use:
        return obs
    obs = obs.clone()
    obs[:, spec.prev_actions] = 0.0
    return obs


def masked_l1(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over the non-padded entries."""
    valid = (~is_pad)[..., None].to(pred.dtype)
    return ((pred - target).abs() * valid).sum() / (valid.sum() * pred.shape[-1]).clamp(min=1.0)
