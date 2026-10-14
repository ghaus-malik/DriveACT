"""Deterministic random streams.

Every random draw (layout variation, obstacle placement, start pose, expert noise) comes from a NumPy generator
seeded by a tuple of non-negative integers, so data collection and evaluation are reproducible and their streams are
disjoint (``stream`` differs between collection and evaluation, see ``configs/eval.yaml``).
"""

from __future__ import annotations

import random

import numpy as np
import torch

_OFFSET = 1_000_000  # layout seeds may be negative (Track 4 showcase = -1)


def episode_rng(stream: int, track: int, layout_seed: int, episode: int) -> np.random.Generator:
    return np.random.default_rng([int(stream), int(track), int(layout_seed) + _OFFSET, int(episode)])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
