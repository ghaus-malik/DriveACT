import pytest
import torch

from driveact.environments.factory import make_env
from driveact.environments.observations import ObsSpec
from driveact.evaluation.rollout import run_episodes
from driveact.experts.expert import ScriptedExpert


@pytest.mark.parametrize("track,seed", [(1, 0), (2, 0), (3, 0), (4, -1), (4, 1000)])
def test_expert_completes_tracks_on_test_double(cfg, track, seed):
    env = make_env(cfg, track, [seed], 2, backend="kinematic")
    recs = run_episodes(env, ScriptedExpert(cfg), [0, 1])
    assert [r.outcome for r in recs] == ["success", "success"]
    for r in recs:
        assert not r.curb_strike and not r.lane_departure
        assert r.obstacle_groups_passed == r.obstacle_groups
        assert r.min_clearance > 0.3


def test_observation_layout(cfg):
    env = make_env(cfg, 4, [-1], 1, backend="kinematic")
    env.reset([0], [0])
    obs = env.observe()
    spec = ObsSpec.from_config(cfg)
    assert obs.shape == (1, spec.dim)
    assert torch.isfinite(obs).all()
    assert obs[:, spec.lidar].min() >= 0 and obs[:, spec.lidar].max() <= 1
    # at the start: speed 0, previous commands 0, next intersection is straight ahead within the horizon or none
    st = obs[0, spec.state]
    assert float(st[0]) == 0.0 and float(st[3]) == 0.0 and float(st[4]) == 0.0


def test_collision_is_detected(cfg):
    env = make_env(cfg, 2, [0], 1, backend="kinematic")
    env.reset([0], [0])
    done = False
    for _ in range(400):  # full throttle, steering hard right: hits the curb then the wall
        info = env.step(torch.tensor([[-1.0, 1.0]]))
        if bool(info["done"][0]):
            done = True
            break
    assert done
    rec = env.episode_record(0)
    assert rec.curb_strike and rec.outcome == "collision_wall"
