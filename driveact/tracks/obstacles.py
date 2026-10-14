"""Static obstacles (parked cars, traffic cones, concrete barriers) placed along the route.

Positions are expressed in route coordinates (arc length ``s`` along the right-lane centre and lateral offset ``d``,
+ = left; right curb -lane/2, centre line +lane/2, left curb +3 lane/2) and converted to world poses. Every sampled
configuration is checked by the expert's lateral planner (:func:`driveact.experts.planner.plan_lateral_offsets`):
a configuration is accepted only if a smooth, collision-free offset profile exists, so obstacles never make the
predefined route impossible.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from driveact.geometry import box_corners

KIND_IDS = {"parked_car": 0, "left_parked_car": 1, "cone": 2, "barrier": 3}


@dataclass
class Obstacle:
    kind: str  # parked_car | left_parked_car | cone | barrier
    shape: str  # box | circle
    x: float
    y: float
    yaw: float
    length: float = 0.0  # box size along its yaw
    width: float = 0.0
    radius: float = 0.0  # circles
    height: float = 1.0
    group: int = 0

    def corners(self) -> np.ndarray:
        if self.shape != "box":
            raise ValueError("circle obstacles have no corners")
        return box_corners(self.x, self.y, self.length, self.width, self.yaw)

    def outline(self) -> np.ndarray:
        """Points on the obstacle outline (used to compute route-frame extents)."""
        if self.shape == "box":
            return self.corners()
        a = np.linspace(0, 2 * np.pi, 9)[:-1]
        return np.stack((self.x + self.radius * np.cos(a), self.y + self.radius * np.sin(a)), axis=1)


@dataclass
class ObstacleGroup:
    kind: str
    s_min: float  # route-frame extents of all obstacles of the group
    s_max: float
    d_min: float
    d_max: float


@dataclass
class ObstacleSet:
    items: list[Obstacle] = field(default_factory=list)
    groups: list[ObstacleGroup] = field(default_factory=list)

    def boxes(self) -> np.ndarray:
        """(K, 6) cx, cy, half_length, half_width, yaw, kind_id."""
        rows = [[o.x, o.y, o.length / 2, o.width / 2, o.yaw, KIND_IDS[o.kind]] for o in self.items if o.shape == "box"]
        return np.asarray(rows, dtype=np.float64).reshape(-1, 6)

    def circles(self) -> np.ndarray:
        """(K, 4) cx, cy, radius, kind_id."""
        rows = [[o.x, o.y, o.radius, KIND_IDS[o.kind]] for o in self.items if o.shape == "circle"]
        return np.asarray(rows, dtype=np.float64).reshape(-1, 4)

    def count(self, kind: str) -> int:
        return sum(1 for o in self.items if o.kind == kind)

    def to_json(self) -> str:
        return json.dumps({"items": [asdict(o) for o in self.items], "groups": [asdict(g) for g in self.groups]})

    @staticmethod
    def from_json(text: str) -> "ObstacleSet":
        d = json.loads(text)
        return ObstacleSet([Obstacle(**o) for o in d["items"]], [ObstacleGroup(**g) for g in d["groups"]])


# ------------------------------------------------------------------------------------------ route coordinates
def route_frame(layout, points: np.ndarray, s_hint: float, window: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Project world points onto the route near ``s_hint`` -> (s, d) arrays."""
    sel = np.nonzero(np.abs(layout.route_s - s_hint) <= window)[0]
    route = layout.route[sel]
    dist = np.linalg.norm(points[:, None, :] - route[None, :, :], axis=-1)
    k = sel[dist.argmin(axis=1)]
    h = layout.route_heading[k]
    rel = points - layout.route[k]
    along = rel[:, 0] * np.cos(h) + rel[:, 1] * np.sin(h)
    lat = -rel[:, 0] * np.sin(h) + rel[:, 1] * np.cos(h)
    return layout.route_s[k] + along, lat


def _pose(layout, s: float, d: float) -> tuple[float, float, float]:
    return layout.pose_at(s, d)


def allowed_zones(layout, ocfg) -> list[tuple[float, float]]:
    """Route intervals where obstacle groups may be placed."""
    lo, hi = layout.start_s + float(ocfg.start_clearance), layout.goal_s - float(ocfg.goal_clearance)
    blocked = []
    for p in layout.pieces:
        if p.kind == "junction":
            blocked.append((p.s_start - float(ocfg.junction_clearance), p.s_end + float(ocfg.junction_clearance)))
        elif p.kind == "curve" and p.radius < float(ocfg.min_radius):
            blocked.append((p.s_start, p.s_end))
    zones = [(lo, hi)]
    for a, b in blocked:
        nxt = []
        for z0, z1 in zones:
            if b <= z0 or a >= z1:
                nxt.append((z0, z1))
                continue
            if a > z0:
                nxt.append((z0, a))
            if b < z1:
                nxt.append((b, z1))
        zones = nxt
    return [(a, b) for a, b in zones if b - a > 1.0]


# ---------------------------------------------------------------------------------------------- group makers
def _make_group(kind: str, layout, s0: float, rng: np.random.Generator, ocfg_types, gid: int) -> list[Obstacle]:
    t = ocfg_types[kind]
    if kind in ("parked_car", "left_parked_car"):
        d = rng.uniform(*t.offset)
        x, y, h = _pose(layout, s0, d)
        yaw = h + math.radians(rng.uniform(-1, 1) * float(t.yaw_deg))
        if kind == "left_parked_car":  # parked in the opposite lane, facing the other way
            yaw += math.pi
        return [Obstacle(kind, "box", x, y, yaw, float(t.length), float(t.width), height=float(t.height), group=gid)]
    if kind == "cone_taper":
        n = int(rng.integers(t.count[0], t.count[1] + 1))
        d_end = rng.uniform(*t.offset_end)
        out = []
        for i in range(n):
            f = i / max(1, n - 1)
            x, y, _ = _pose(layout, s0 + i * float(t.spacing), float(t.offset_start) + f * (d_end - float(t.offset_start)))
            out.append(Obstacle("cone", "circle", x, y, 0.0, radius=float(t.radius), height=float(t.height), group=gid))
        return out
    if kind == "barrier":
        n = int(rng.integers(t.count[0], t.count[1] + 1))
        d0 = rng.uniform(*t.offset)
        out = []
        for i in range(n):
            x, y, h = _pose(layout, s0 + 0.6 * i, d0 + 1.6 * i)
            yaw = h + 0.5 * math.pi + math.radians(rng.uniform(-1, 1) * float(t.yaw_deg))
            out.append(Obstacle("barrier", "box", x, y, yaw, float(t.length), float(t.width), height=float(t.height), group=gid))
        return out
    raise ValueError(f"unknown obstacle group '{kind}'")


def _group_extent(layout, items: list[Obstacle], s_hint: float) -> ObstacleGroup:
    pts = np.concatenate([o.outline() for o in items], axis=0)
    s, d = route_frame(layout, pts, s_hint)
    return ObstacleGroup(items[0].kind if items[0].kind != "cone" else "cone_taper", float(s.min()), float(s.max()),
                         float(d.min()), float(d.max()))


def _nominal_length(kind: str, ocfg_types) -> float:
    t = ocfg_types[kind]
    if kind == "cone_taper":
        return float(t.spacing) * (t.count[1] - 1) + 1.0
    return float(t.length) + 1.0


def sample_obstacles(layout, track_obstacle_cfg, cfg, rng: np.random.Generator, max_tries: int = 300) -> ObstacleSet:
    """Random feasible obstacle configuration for a layout (empty if the track has no obstacle groups)."""
    from driveact.experts.planner import InfeasibleError, plan_lateral_offsets

    groups_cfg = track_obstacle_cfg.get("groups", {}) or {}
    if not groups_cfg:
        return ObstacleSet()
    types = cfg.obstacles
    zones = allowed_zones(layout, track_obstacle_cfg)
    if not zones:
        raise RuntimeError(f"layout {layout.name} seed {layout.seed} has no room for obstacles")
    min_gap = float(track_obstacle_cfg.min_gap)
    for _ in range(max_tries):
        kinds = []
        for kind, (lo, hi) in groups_cfg.items():
            kinds += [kind] * int(rng.integers(lo, hi + 1))
        rng.shuffle(kinds)
        placed: list[tuple[float, float]] = []
        items: list[Obstacle] = []
        groups: list[ObstacleGroup] = []
        ok = True
        for gid, kind in enumerate(kinds):
            length = _nominal_length(kind, types)
            for _attempt in range(60):
                z0, z1 = zones[int(rng.choice(len(zones), p=_zone_weights(zones, length)))]
                if z1 - z0 < length:
                    continue
                s0 = rng.uniform(z0, z1 - length)
                if all(s0 + length + min_gap <= a or s0 >= b + min_gap for a, b in placed):
                    break
            else:
                ok = False
                break
            new = _make_group(kind, layout, s0, rng, types, gid)
            g = _group_extent(layout, new, s0 + length / 2)
            placed.append((g.s_min, g.s_max))
            items += new
            groups.append(g)
        if not ok:
            continue
        obs = ObstacleSet(items, groups)
        try:
            plan_lateral_offsets(layout, obs, cfg)
        except InfeasibleError:
            continue
        return obs
    raise RuntimeError(f"could not place a feasible obstacle configuration on {layout.name} seed {layout.seed}")


def _zone_weights(zones, length: float) -> np.ndarray:
    w = np.array([max(0.0, b - a - length) for a, b in zones])
    if w.sum() <= 0:
        w = np.ones(len(zones))
    return w / w.sum()
