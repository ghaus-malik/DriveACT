import math

import torch

from driveact.datasets.chunk_dataset import Normalizer
from driveact.models.controller import PolicyController


class CountingModel(torch.nn.Module):
    """Predicts chunk[j] = (call_index, j) so tests can see which chunk an action came from."""

    kind = "act"

    def __init__(self, k):
        super().__init__()
        self.chunk_size = k
        self.calls = 0
        self.p = torch.nn.Parameter(torch.zeros(1))

    def predict(self, obs):
        self.calls += 1
        j = torch.arange(self.chunk_size, dtype=torch.float32)
        out = torch.stack((torch.full_like(j, float(self.calls)), j), -1)
        return out[None].expand(obs.shape[0], -1, -1) / 100.0


def _norm():
    return Normalizer(torch.zeros(3), torch.ones(3), torch.zeros(2), torch.ones(2))


def test_receding_horizon_replans_every_n_steps():
    m = CountingModel(5)
    c = PolicyController(m, _norm(), 1, "cpu", replan_every=2)
    c.reset([0])
    acts = [c.act(torch.zeros(1, 3)) * 100 for _ in range(6)]
    assert m.calls == 3
    assert [tuple(a[0].round().tolist()) for a in acts] == [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]


def test_temporal_ensemble_weights_favour_oldest_prediction():
    k, m_coef = 4, 0.5
    m = CountingModel(k)
    c = PolicyController(m, _norm(), 1, "cpu", temporal_ensemble=True, ensemble_k=m_coef)
    c.reset([0])
    a = [c.act(torch.zeros(1, 3)) * 100 for _ in range(3)]
    # step 2: predictions for "now" come from calls 1 (j=2), 2 (j=1), 3 (j=0); call 1 is the oldest -> weight 1
    w = torch.tensor([math.exp(-m_coef * i) for i in range(3)])  # oldest first
    calls = torch.tensor([1.0, 2.0, 3.0])
    js = torch.tensor([2.0, 1.0, 0.0])
    expected = torch.stack(((w * calls).sum() / w.sum(), (w * js).sum() / w.sum()))
    assert torch.allclose(a[2][0], expected, atol=1e-4)
    assert torch.allclose(a[0][0], torch.tensor([1.0, 0.0]), atol=1e-4)
