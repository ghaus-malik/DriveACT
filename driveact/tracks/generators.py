"""Layouts of the four tracks.

Tracks 1-3: layout seed 0 is the canonical recipe from ``configs/tracks/trackN.yaml``; every other seed jitters
straight lengths, curve radii and gentle-curve angles and may mirror all turns.
Track 4: seed -1 ("showcase") is the hand-designed recipe; every other seed is a procedural city route with
3-4 intersections and 1-3 curves. Train and test seeds are listed in the track config (``splits``) and are
disjoint, so held-out Track 4 geometries are never used for training.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from driveact.config import load_track_config
from driveact.tracks.layout import LayoutOverlapError, TrackLayout, build_layout

SHOWCASE_SEED = -1
_GENTLE_DEG = 80.0


def _rng(track: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([track, seed + 10_000])


def _jitter_recipe(pieces: list, var, rng: np.random.Generator, min_radius: float) -> list:
    mirror = rng.random() < float(var.mirror_prob)
    out = []
    for spec in pieces:
        spec = list(spec)
        if spec[0] == "straight":
            spec[1] = float(spec[1]) * (1.0 + rng.uniform(-1, 1) * float(var.length_jitter))
        elif spec[0] == "curve":
            radius = float(spec[1]) * (1.0 + rng.uniform(-1, 1) * float(var.radius_jitter))
            spec[1] = max(min_radius, radius)
            angle = float(spec[2])
            if abs(angle) < _GENTLE_DEG:
                angle = math.copysign(max(10.0, abs(angle) + rng.uniform(-1, 1) * float(var.angle_jitter_deg)), angle)
            spec[2] = -angle if mirror else angle
        elif spec[0] == "junction" and mirror:
            spec[1] = {"left": "right", "right": "left"}.get(spec[1], spec[1])
        out.append(spec)
    return out


def _procedural_recipe(proc, rng: np.random.Generator) -> list:
    nj = int(rng.integers(proc.num_junctions[0], proc.num_junctions[1] + 1))
    nc = int(rng.integers(proc.num_curves[0], proc.num_curves[1] + 1))
    while True:  # at least one left and one right turn
        turns = list(rng.choice(["left", "straight", "right"], size=nj, p=[0.38, 0.24, 0.38]))
        if "left" in turns and "right" in turns:
            break
    features = ["J"] * nj + ["C"] * nc
    rng.shuffle(features)
    lo, hi = proc.straight_length
    pieces = [["straight", float(rng.uniform(lo, hi))]]
    for f in features:
        if f == "J":
            pieces.append(["junction", str(turns.pop())])
        else:
            angle = float(rng.uniform(*proc.curve_angle_deg)) * (1 if rng.random() < 0.5 else -1)
            pieces.append(["curve", float(rng.uniform(*proc.curve_radius)), angle])
        pieces.append(["straight", float(rng.uniform(lo, hi))])
    return pieces


def layout_recipe(track: int, seed: int, cfg, track_cfg=None) -> list:
    """Piece list for (track, layout seed)."""
    tc = track_cfg or load_track_config(track)
    road = cfg.road
    base = [list(p) for p in tc.pieces]
    if track == 4:
        if seed == SHOWCASE_SEED:
            return base
        raise RuntimeError("procedural Track 4 recipes are drawn inside make_layout (they need overlap retries)")
    if seed == 0:
        return base
    return _jitter_recipe(base, tc.variation, _rng(track, seed), float(road.min_curve_radius))


def make_layout(track: int, seed: int, cfg, track_cfg=None, max_extent: float | None = None) -> TrackLayout:
    """Build the layout for (track, layout seed). Raises if no valid layout can be produced."""
    tc = track_cfg or load_track_config(track)
    road = cfg.road
    max_extent = max_extent if max_extent is not None else float(cfg.sim.env_spacing) - 80.0
    kw = dict(name=str(tc.name), track=int(track), seed=int(seed), start_s=float(tc.start.s),
              goal_margin=float(tc.goal_margin))
    if track == 4 and seed != SHOWCASE_SEED:
        rng = _rng(track, seed)
        for _ in range(int(tc.procedural.max_tries)):
            pieces = _procedural_recipe(tc.procedural, rng)
            try:
                layout = build_layout(pieces, road, **kw)
            except LayoutOverlapError:
                continue
            x0, y0, x1, y1 = layout.bounds()
            if max(x1 - x0, y1 - y0) <= max_extent and _has_room_for_obstacles(layout, tc, cfg):
                return layout
        raise RuntimeError(f"no valid procedural layout for track 4 seed {seed}")
    if seed == 0 or (track == 4 and seed == SHOWCASE_SEED):
        layout = build_layout(layout_recipe(track, seed, cfg, tc), road, **kw)
    else:
        rng = _rng(track, seed)
        base = [list(p) for p in tc.pieces]
        for _ in range(200):
            pieces = _jitter_recipe(copy.deepcopy(base), tc.variation, rng, float(road.min_curve_radius))
            try:
                layout = build_layout(pieces, road, **kw)
                break
            except LayoutOverlapError:
                continue
        else:
            raise RuntimeError(f"no valid layout for track {track} seed {seed}")
    x0, y0, x1, y1 = layout.bounds()
    if max(x1 - x0, y1 - y0) > max_extent:
        raise RuntimeError(f"track {track} seed {seed} is larger than the environment spacing allows")
    return layout


def _has_room_for_obstacles(layout, tc, cfg, trials: int = 5) -> bool:
    """A layout is usable only if feasible obstacle configurations can be drawn for it (checked on a few draws)."""
    from driveact.tracks.obstacles import sample_obstacles

    if not (tc.obstacles.get("groups") or {}):
        return True
    for i in range(trials):
        try:
            sample_obstacles(layout, tc.obstacles, cfg, np.random.default_rng([99, i]), max_tries=60)
        except RuntimeError:
            return False
    return True


def split_seeds(track: int, split: str, track_cfg=None) -> list[int]:
    tc = track_cfg or load_track_config(track)
    if split not in tc.splits:
        raise KeyError(f"track {track} has no split '{split}'")
    return [int(s) for s in tc.splits[split]]
