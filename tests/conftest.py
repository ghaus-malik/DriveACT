import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch.set_num_threads(2)

from driveact.config import load_config  # noqa: E402
from driveact.tracks.generators import make_layout  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def layouts(cfg):
    """Canonical layouts of every track (Track 4: the hand-designed showcase)."""
    return {1: make_layout(1, 0, cfg), 2: make_layout(2, 0, cfg), 3: make_layout(3, 0, cfg), 4: make_layout(4, -1, cfg)}
