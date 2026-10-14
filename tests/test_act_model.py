import copy

import pytest
import torch

from driveact.datasets.chunk_dataset import ChunkSampler, Normalizer
from driveact.environments.observations import ObsSpec
from driveact.models.act_policy import ACTPolicy, masked_l1
from driveact.models.bc_policy import BCPolicy
from driveact.models.cvae import kl_divergence
from driveact.training.trainer import load_checkpoint, save_checkpoint


def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c.act.model.hidden_dim = 64
    c.act.model.dim_feedforward = 128
    c.act.model.enc_layers = 2
    c.act.model.dec_layers = 2
    c.act.model.cvae_layers = 2
    return c


@pytest.fixture(scope="module")
def spec(cfg):
    return ObsSpec.from_config(cfg)


def test_shapes_and_latent(cfg, spec):
    c = small_cfg(cfg)
    model = ACTPolicy(spec, c.act.model)
    B, k = 4, c.act.model.chunk_size
    obs = torch.randn(B, spec.dim)
    act = torch.randn(B, k, 2)
    pad = torch.zeros(B, k, dtype=torch.bool)
    model.train()
    a_hat, mu, logvar = model(obs, act, pad)
    assert a_hat.shape == (B, k, 2) and mu.shape == (B, c.act.model.latent_dim) and logvar.shape == mu.shape
    model.eval()
    p1, mu0, _ = model(obs)
    p2 = model.predict(obs)
    assert mu0 is None and p1.shape == (B, k, 2) and torch.allclose(p1, p2)  # prior z = 0: deterministic
    # the posterior latent changes the prediction (the CVAE path is wired in)
    z_a = model(obs, act, pad)[0]
    z_b = model(obs, -act, pad)[0]
    assert not torch.allclose(z_a, z_b)


def test_kl_and_masked_l1():
    mu, logvar = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 0.5]])
    manual = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum()
    assert torch.isclose(kl_divergence(mu, logvar), manual)
    assert float(kl_divergence(torch.zeros(3, 4), torch.zeros(3, 4))) == 0.0
    pred, tgt = torch.zeros(1, 3, 2), torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
    pad = torch.tensor([[False, False, True]])
    assert torch.isclose(masked_l1(pred, tgt, pad), torch.tensor(1.5))


def test_backprop_reaches_every_parameter(cfg, spec):
    c = small_cfg(cfg)
    model = ACTPolicy(spec, c.act.model)
    B, k = 3, c.act.model.chunk_size
    out = model.compute_loss(torch.randn(B, spec.dim), torch.randn(B, k, 2), torch.zeros(B, k, dtype=torch.bool), 10.0)
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing


def test_checkpoint_round_trip(cfg, spec, tmp_path):
    c = small_cfg(cfg)
    model = ACTPolicy(spec, c.act.model).eval()
    norm = Normalizer(torch.zeros(spec.dim), torch.ones(spec.dim), torch.zeros(2), torch.ones(2))
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, norm, spec, c, "act", {"epoch": 1})
    loaded, norm2, ckpt = load_checkpoint(path)
    obs = torch.randn(2, spec.dim)
    assert torch.allclose(model.predict(obs), loaded.predict(obs)) and ckpt["epoch"] == 1


def test_overfits_a_tiny_dataset(cfg, spec):
    """Stage 3 sanity check: ACT can memorise a handful of chunks (L1 in normalised units drops below 0.1)."""
    torch.manual_seed(0)
    c = small_cfg(cfg)
    c.act.model.dropout = 0.0
    T = 60
    t = torch.linspace(0, 6, T)
    eps = [{"obs": torch.randn(T, spec.dim).numpy() * 0.5 + i, "actions": torch.stack((torch.sin(t + i), torch.cos(t)), 1).numpy()}
           for i in range(2)]
    norm = Normalizer.fit(eps)
    sampler = ChunkSampler(eps, norm, c.act.model.chunk_size, "cpu")
    model = ACTPolicy(spec, c.act.model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(300):
        obs, act, pad = sampler.sample(64)
        loss = model.compute_loss(obs, act, pad, 1.0)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    obs, act, pad = sampler.gather(torch.arange(len(sampler)))
    assert float(masked_l1(model.predict(obs), act, pad)) < 0.1


def test_bc_policy(cfg, spec):
    model = BCPolicy(spec, cfg.act.model, cfg.act.bc)
    obs = torch.randn(5, spec.dim)
    assert model(obs).shape == (5, 2) and model.predict(obs).shape == (5, 1, 2)
    out = model.compute_loss(obs, torch.randn(5, 3, 2), torch.zeros(5, 3, dtype=torch.bool))
    out["loss"].backward()


def test_prev_action_ablation_and_augmentation(cfg, spec):
    from driveact.training.trainer import augment_prev_actions

    c = small_cfg(cfg)
    c.act.model.use_prev_actions = False
    model = ACTPolicy(spec, c.act.model).eval()
    obs = torch.randn(3, spec.dim)
    obs2 = obs.clone()
    obs2[:, spec.prev_actions] += 5.0
    assert torch.allclose(model.predict(obs), model.predict(obs2))  # previous commands are ignored
    torch.manual_seed(0)
    aug = augment_prev_actions(obs.repeat(200, 1), spec, 0.5, 0.0)
    dropped = (aug[:, spec.prev_actions] == 0).all(dim=1).float().mean()
    assert 0.35 < float(dropped) < 0.65
    other = [i for i in range(spec.dim) if i not in range(spec.prev_actions.start, spec.prev_actions.stop)]
    assert torch.equal(aug[:, other], obs.repeat(200, 1)[:, other])
