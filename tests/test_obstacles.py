import numpy as np
import pytest

from driveact.config import load_track_config
from driveact.experts.planner import plan_lateral_offsets, plan_speed
from driveact.tracks.obstacles import allowed_zones, route_frame, sample_obstacles
from driveact.utils.seeding import episode_rng


@pytest.mark.parametrize("track", [2, 3, 4])
def test_sampled_obstacles_are_feasible_and_on_the_road(cfg, layouts, track):
    layout = layouts[track]
    tc = load_track_config(track)
    lane = cfg.road.lane_width
    for ep in range(15):
        obs = sample_obstacles(layout, tc.obstacles, cfg, episode_rng(0, track, layout.seed, ep))
        lo, hi = tc.obstacles.groups[next(iter(tc.obstacles.groups))]
        assert len(obs.groups) >= 1
        for o in obs.items:
            s, d = route_frame(layout, o.outline(), float(route_frame(layout, np.array([[o.x, o.y]]), layout.length / 2,
                                                                       window=1e4)[0][0]))
            assert d.min() > -lane / 2 - 0.15 and d.max() < 1.5 * lane + 0.05
        # spacing between groups
        spans = sorted((g.s_min, g.s_max) for g in obs.groups)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert b0 - a1 >= float(tc.obstacles.min_gap) - 1.0
        d = plan_lateral_offsets(layout, obs, cfg)  # must not raise
        assert np.all(np.isfinite(d))


def test_obstacles_respect_zones(cfg, layouts):
    layout = layouts[4]
    tc = load_track_config(4)
    zones = allowed_zones(layout, tc.obstacles)
    for ep in range(10):
        obs = sample_obstacles(layout, tc.obstacles, cfg, episode_rng(0, 4, layout.seed, ep))
        for g in obs.groups:
            mid = 0.5 * (g.s_min + g.s_max)
            assert any(a - 1.0 <= mid <= b + 1.0 for a, b in zones)
            for p in layout.pieces:
                if p.kind == "junction":
                    assert g.s_max < p.s_start - 20 or g.s_min > p.s_end + 20


def test_obstacle_sampling_is_deterministic(cfg, layouts):
    tc = load_track_config(3)
    a = sample_obstacles(layouts[3], tc.obstacles, cfg, episode_rng(0, 3, 0, 5))
    b = sample_obstacles(layouts[3], tc.obstacles, cfg, episode_rng(0, 3, 0, 5))
    c = sample_obstacles(layouts[3], tc.obstacles, cfg, episode_rng(100000, 3, 0, 5))
    assert a.to_json() == b.to_json()
    assert a.to_json() != c.to_json()


def test_offsets_clear_every_obstacle(cfg, layouts):
    layout = layouts[3]
    tc = load_track_config(3)
    half_w = cfg.vehicle.body.width / 2
    for ep in range(10):
        obs = sample_obstacles(layout, tc.obstacles, cfg, episode_rng(0, 3, 0, ep))
        d = plan_lateral_offsets(layout, obs, cfg)
        for g in obs.groups:
            near = (layout.route_s >= g.s_min - 2.2) & (layout.route_s <= g.s_max + 2.2)
            car_lo, car_hi = d[near] - half_w, d[near] + half_w
            gap = np.maximum(g.d_min - car_hi, car_lo - g.d_max)
            assert gap.min() > 0.3, (g.kind, gap.min())


def test_speed_profile_stops_at_goal(cfg, layouts):
    layout = layouts[2]
    v = plan_speed(layout, np.zeros_like(layout.route_s), cfg, 10.0)
    assert v.max() <= 10.0 + 1e-6
    assert v[layout.route_s >= layout.goal_s].max() == 0.0
    # deceleration feasible: v^2 drops by at most 2 a ds
    dv2 = np.diff(v**2)
    assert dv2.min() >= -2 * cfg.expert.speed.plan_decel * 0.5 - 1e-6
