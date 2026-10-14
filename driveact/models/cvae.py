"""ACT's conditional-VAE posterior encoder ("style variable" z).

Training only: a Transformer encoder reads [CLS, vehicle state, a_t ... a_{t+k-1}] (with fixed sinusoidal positions
and a padding mask for actions past the episode end), and the CLS output is projected to the mean and log-variance
of a ``latent_dim`` Gaussian. z is drawn with the reparameterisation trick and regularised towards N(0, I) by the KL
term. At inference the encoder is not used: the policy is conditioned on the prior mean z = 0.
"""

from __future__ import annotations

import torch
from torch import nn

from driveact.models.transformer import Encoder, sinusoid_table


class CVAEEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, d: int, nheads: int, dim_ff: int,
                 num_layers: int, latent_dim: int, dropout: float):
        super().__init__()
        self.cls_embed = nn.Embedding(1, d)
        self.state_proj = nn.Linear(state_dim, d)
        self.action_proj = nn.Linear(action_dim, d)
        self.encoder = Encoder(d, nheads, dim_ff, num_layers, dropout)
        self.to_latent = nn.Linear(d, 2 * latent_dim)  # -> (mu, logvar)
        self.register_buffer("pos_table", sinusoid_table(2 + chunk_size, d), persistent=False)
        self.latent_dim = latent_dim

    def forward(self, state: torch.Tensor, actions: torch.Tensor, is_pad: torch.Tensor):
        """state (B, S), actions (B, k, A), is_pad (B, k) -> mu, logvar (B, latent_dim)."""
        B = state.shape[0]
        tokens = torch.cat(
            (self.cls_embed.weight[None].expand(B, -1, -1), self.state_proj(state)[:, None], self.action_proj(actions)),
            dim=1,
        )
        mask = torch.cat((torch.zeros(B, 2, dtype=torch.bool, device=state.device), is_pad), dim=1)
        out = self.encoder(tokens, pos=self.pos_table[:, : tokens.shape[1]], key_padding_mask=mask)
        mu, logvar = self.to_latent(out[:, 0]).chunk(2, dim=-1)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)), summed over latent dimensions and averaged over the batch (as in ACT)."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
