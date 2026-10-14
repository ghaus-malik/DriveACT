"""Observation encoders producing Transformer tokens from LiDAR, vehicle state and route features."""

from __future__ import annotations

import math

import torch
from torch import nn


class LidarEncoder(nn.Module):
    """1-D convolutional stem over the LiDAR sectors (circular padding: the scan wraps around).

    (B, num_sectors) -> (B, num_tokens, d). Each token summarises ``num_sectors / num_tokens`` neighbouring sectors,
    the analogue of ACT's ResNet feature-map cells for camera images.
    """

    def __init__(self, num_sectors: int, num_tokens: int, channels=(32, 64), d: int = 256):
        super().__init__()
        ratio = num_sectors // num_tokens
        if num_sectors % num_tokens or ratio & (ratio - 1):
            raise ValueError("num_sectors / lidar_tokens must be a power of two")
        channels = list(channels)
        layers: list[nn.Module] = [nn.Conv1d(1, channels[0], 5, padding=2, padding_mode="circular"), nn.ReLU()]
        c = channels[0]
        for i in range(int(math.log2(ratio))):
            c_out = channels[min(i + 1, len(channels) - 1)]
            layers += [nn.Conv1d(c, c_out, 3, stride=2, padding=1, padding_mode="circular"), nn.ReLU()]
            c = c_out
        self.stem = nn.Sequential(*layers)
        self.proj = nn.Conv1d(c, d, 1)
        self.num_tokens = num_tokens
        self.out_channels = c

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x[:, None, :])  # (B, C, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x)).transpose(1, 2)


def mlp(sizes: list[int], dropout: float = 0.0, last_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
