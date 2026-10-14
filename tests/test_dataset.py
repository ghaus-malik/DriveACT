import numpy as np
import torch

from driveact.datasets.chunk_dataset import ChunkSampler, Normalizer, split_episodes
from driveact.datasets.hdf5_store import EnvDemoRecorder, load_episodes, verify_alignment
from driveact.environments.factory import make_env
from driveact.evaluation.rollout import OUNoise, run_episodes
from driveact.experts.expert import ScriptedExpert


def test_record_load_and_alignment(cfg, tmp_path):
    env = make_env(cfg, 2, [0], 2, backend="kinematic")
    path = str(tmp_path / "demo.h5")
    rec = EnvDemoRecorder(path, cfg, 2, "kinematic", env.spec, env=env)
    noise = OUNoise(cfg.expert.noise, env.N, env.device)
    records = run_episodes(env, ScriptedExpert(cfg), [0, 1, 2], noise=noise, noisy_flags={1: True}, recorder=rec)
    rec.close()
    episodes, attrs = load_episodes(path)
    assert len(episodes) == sum(r.success for r in records) == 3
    for ep in episodes:
        T = len(ep["obs"])
        assert ep["actions"].shape == (T, 2) and ep["executed"].shape == (T, 2) and ep["state"].shape[0] == T
        assert verify_alignment(ep, env.spec) < 1e-6  # obs[t+1] carries executed[t]
        assert ep["obstacles"]
    noisy = [ep for ep in episodes if ep["noisy"]]
    clean = [ep for ep in episodes if not ep["noisy"]]
    assert np.abs(noisy[0]["executed"] - noisy[0]["actions"]).max() > 0.01
    assert np.abs(clean[0]["executed"] - clean[0]["actions"]).max() == 0.0


def test_chunk_sampler_padding():
    eps = [{"obs": np.arange(10, dtype=np.float32)[:, None].repeat(3, 1), "actions": np.stack([np.arange(10)] * 2, 1).astype(np.float32)},
           {"obs": np.zeros((4, 3), np.float32), "actions": np.full((4, 2), 100.0, np.float32)}]
    norm = Normalizer(torch.zeros(3), torch.ones(3), torch.zeros(2), torch.ones(2))
    s = ChunkSampler(eps, norm, 5, "cpu")
    obs, act, pad = s.gather(torch.tensor([0, 7, 10, 13]))
    assert act[0, :, 0].tolist() == [0, 1, 2, 3, 4] and not pad[0].any()
    assert pad[1].tolist() == [False, False, False, True, True] and act[1, :3, 0].tolist() == [7, 8, 9]
    assert act[1, 3:].abs().sum() == 0  # padding never leaks into the next episode
    assert pad[2].tolist() == [False] * 4 + [True] and act[2, 0, 0] == 100
    assert pad[3].tolist() == [False] + [True] * 4


def test_split_by_episode():
    eps = [{"id": i} for i in range(20)]
    train, val = split_episodes(eps, 0.1, 0)
    assert len(val) == 2 and len(train) == 18
    assert not {e["id"] for e in train} & {e["id"] for e in val}
