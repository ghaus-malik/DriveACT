import numpy as np
import pytest

from driveact.config import load_track_config
from driveact.geometry import point_segment_distance, wrap_angle
from driveact.tracks.generators import make_layout, split_seeds
from driveact.tracks.layout import MANEUVER_NONE, TURNS


def all_layouts(cfg):
    for track in (1, 2, 3, 4):
        for split in ("train", "test"):
            for seed in split_seeds(track, split):
                yield track, split, seed


@pytest.mark.parametrize("track", [1, 2, 3, 4])
def test_every_split_layout_builds(cfg, track):
    for split in ("train", "test"):
        for seed in split_seeds(track, split):
            layout = make_layout(track, seed, cfg)
            assert layout.length > 200
            assert layout.start_s < layout.goal_s < layout.length


def test_route_is_continuous_and_smooth(layouts):
    for layout in layouts.values():
        steps = np.linalg.norm(np.diff(layout.route, axis=0), axis=1)
        assert steps.max() < 0.65 and steps.min() > 0.1
        dh = np.abs(wrap_angle(np.diff(layout.route_heading)))
        # tightest route arc is the right turn through an intersection (radius rc + lane/2 = 7.75 m)
        assert dh.max() < 0.5 / 7.0 + 1e-3


def test_route_lane_stays_between_the_curbs(cfg, layouts):
    lane = cfg.road.lane_width
    for layout in layouts.values():
        d = point_segment_distance(layout.route, layout.curb_segments()).min(axis=1)
        # the right-lane centre is lane/2 from the right curb (and at intersections never closer)
        assert d.min() > lane / 2 - 0.05, layout.name
        dw = point_segment_distance(layout.route, layout.wall_segments()).min(axis=1)
        # (the route begins and ends at the closing walls; the goal is goal_margin before the end)
        far_from_ends = (layout.route_s > layout.start_s + 5.0) & (layout.route_s < layout.goal_s + 5.0)
        assert dw[far_from_ends].min() > lane / 2 + cfg.road.sidewalk_width - 0.05


def test_track4_has_left_right_and_straight_manoeuvres(layouts):
    layout = layouts[4]
    turns = {j.turn for j in layout.junctions}
    assert turns == {"left", "right", "straight"}
    # before the first intersection the manoeuvre code announces its turn; after the last one: none
    first = layout.junctions[0]
    i = int(np.searchsorted(layout.route_s, first.s_entry - 10))
    assert layout.next_maneuver[i] == TURNS.index(first.turn)
    assert np.isclose(layout.dist_to_maneuver[i], first.s_entry - layout.route_s[i])
    assert layout.next_maneuver[-1] == MANEUVER_NONE


def test_track4_train_and_test_layouts_are_distinct(cfg):
    train = {tuple(map(str, sum(make_layout(4, s, cfg).meta["pieces"], []))) for s in split_seeds(4, "train")}
    test = {tuple(map(str, sum(make_layout(4, s, cfg).meta["pieces"], []))) for s in split_seeds(4, "test")}
    assert not set(split_seeds(4, "train")) & set(split_seeds(4, "test"))
    assert not train & test


def test_layouts_are_deterministic(cfg):
    a, b = make_layout(4, 3, cfg), make_layout(4, 3, cfg)
    assert np.array_equal(a.route, b.route)


def test_track1_has_no_obstacle_groups():
    assert not (load_track_config(1).obstacles.get("groups") or {})


def test_parallel_environments_never_overlap(cfg):
    """With the layout bounding boxes centred on 600 m grid points, no two Track 4 environments touch."""
    from driveact.environments.placement import centered_origins, check_no_overlap, grid_points

    seeds = split_seeds(4, "train") + split_seeds(4, "test")
    layouts = [make_layout(4, seeds[i % len(seeds)], cfg) for i in range(32)]
    origins = centered_origins(grid_points(32, cfg.sim.env_spacing), layouts)
    check_no_overlap(layouts, origins, cfg.sim.env_margin)
    # without centring (start point on the grid point) the same assignment overlaps
    with pytest.raises(RuntimeError):
        check_no_overlap(layouts, grid_points(32, cfg.sim.env_spacing), cfg.sim.env_margin)
