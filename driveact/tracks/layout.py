"""Track layouts built from road pieces: straights, circular curves and 4-way intersections.

A layout is described by a list of pieces that a "turtle" lays down one after the other along the road axis:

    [straight, L]            straight road of length L
    [curve, R, angle_deg]    circular arc of the road axis with radius R (+ angle = left turn)
    [junction, turn]         4-way intersection; the route turns left/right or goes straight. The two arms that the
                             route does not use become short dead-end roads closed by a wall.

Road cross-section (``configs/road.yaml``): two lanes of ``lane_width`` (one per direction), curbs at the road edges,
sidewalks of ``sidewalk_width`` and a concrete wall behind the sidewalk. The route follows the centre of the right
lane. At intersections the curbs are rounded with radius ``junction_corner_radius``; the route arcs through the
intersection concentrically with the corner curbs (right turn radius rc + W/4, left turn rc + 3W/4).

Everything here is NumPy and deterministic; the same :class:`TrackLayout` feeds the USD scene builder (visual and
collision meshes), the analytic LiDAR, the collision checks, the expert and the observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from driveact.geometry import (
    arc_length,
    left_normal,
    offset_polyline,
    polyline_curvature,
    polyline_headings,
    polyline_segments,
    resample_polyline,
    rot2,
    unit,
    wrap_angle,
)

TURNS = ("left", "straight", "right")
MANEUVER_NONE = 3  # index of "no intersection ahead" in the manoeuvre one-hot (left, straight, right, none)
CHORD_ERROR = 0.02  # m, maximum deviation of the faceted geometry from the true arcs


@dataclass
class Strip:
    """Surface between two polylines with the same number of vertices (``left`` and ``right`` of the travel direction)."""

    left: np.ndarray
    right: np.ndarray
    kind: str  # "road" | "sidewalk"


@dataclass
class Fan:
    """Star-shaped polygon (triangulated as a fan around ``center``); used for intersection surfaces."""

    center: np.ndarray
    boundary: np.ndarray  # (K, 2), counter-clockwise, not closed
    kind: str


@dataclass
class Wall:
    """Inner face of a wall as a polyline; the solid lies on side ``side`` (+1 left / -1 right of the direction)."""

    points: np.ndarray
    side: int


@dataclass
class Piece:
    kind: str  # straight | curve | junction
    s_start: float  # route arc length where the piece starts / ends
    s_end: float
    radius: float = math.inf  # axis radius (curves)
    angle: float = 0.0  # signed turn angle (rad)
    turn: str = ""  # junctions


@dataclass
class Junction:
    center: np.ndarray
    heading_in: float
    turn: str
    half_size: float  # half side of the square intersection cell (W/2 + corner radius)
    s_entry: float
    s_exit: float


@dataclass
class TrackLayout:
    name: str
    track: int
    seed: int
    lane_width: float
    sidewalk_width: float
    route: np.ndarray  # (P, 2) right-lane centre line, uniform spacing
    route_heading: np.ndarray  # (P,)
    route_s: np.ndarray  # (P,)
    route_curvature: np.ndarray  # (P,)
    next_maneuver: np.ndarray  # (P,) int: 0 left, 1 straight, 2 right, 3 none (next intersection not yet exited)
    dist_to_maneuver: np.ndarray  # (P,) route metres to the next intersection entry (0 inside it)
    pieces: list[Piece]
    junctions: list[Junction]
    curbs: list[np.ndarray]
    walls: list[Wall]
    strips: list[Strip]
    fans: list[Fan]
    center_lines: list[np.ndarray]  # dense axis polylines for dashed centre markings
    edge_lines: list[np.ndarray]  # dense polylines for solid edge markings
    start_s: float
    goal_s: float
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------------------------------ queries
    @property
    def length(self) -> float:
        return float(self.route_s[-1])

    @property
    def spacing(self) -> float:
        return float(self.route_s[1] - self.route_s[0])

    def wall_segments(self) -> np.ndarray:
        return np.concatenate([polyline_segments(w.points) for w in self.walls], axis=0)

    def curb_segments(self) -> np.ndarray:
        return np.concatenate([polyline_segments(c) for c in self.curbs], axis=0)

    def pose_at(self, s: float, lateral: float = 0.0) -> tuple[float, float, float]:
        """(x, y, yaw) on the route at arc length ``s`` shifted ``lateral`` metres to the left."""
        x = np.interp(s, self.route_s, self.route[:, 0])
        y = np.interp(s, self.route_s, self.route[:, 1])
        h = self.heading_at(s)
        n = left_normal(h)
        return float(x + lateral * n[0]), float(y + lateral * n[1]), float(h)

    def heading_at(self, s) -> np.ndarray:
        c = np.interp(s, self.route_s, np.cos(self.route_heading))
        si = np.interp(s, self.route_s, np.sin(self.route_heading))
        return np.arctan2(si, c)

    def bounds(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        pts = np.concatenate([w.points for w in self.walls], axis=0)
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def piece_at(self, s: float) -> Piece:
        for p in self.pieces:
            if p.s_start <= s <= p.s_end:
                return p
        return self.pieces[-1]

    @property
    def axis_samples(self) -> np.ndarray:
        """Road-axis samples (2 m) of every road band incl. intersections and dead-end arms (decor placement)."""
        return self.meta["axis_samples"]


# ------------------------------------------------------------------------------------------------------ builder
def _arc_steps(angle: float, radius_max: float) -> int:
    dtheta = 2.0 * math.acos(max(-1.0, 1.0 - CHORD_ERROR / max(radius_max, 1e-6)))
    return max(1, int(math.ceil(abs(angle) / dtheta)))


def _sample_straight(p: np.ndarray, h: float, length: float, spacing: float | None = None):
    n = 1 if spacing is None else max(1, int(math.ceil(length / spacing)))
    t = np.linspace(0.0, length, n + 1)
    pts = p[None, :] + t[:, None] * unit(h)[None, :]
    return pts, np.full(n + 1, h)


def _sample_arc(p: np.ndarray, h: float, radius: float, angle: float, steps: int):
    """Arc of the given radius starting at ``p`` with heading ``h``; ``angle`` > 0 turns left."""
    sign = 1.0 if angle >= 0 else -1.0
    center = p + sign * radius * left_normal(h)
    phi0 = math.atan2(p[1] - center[1], p[0] - center[0])
    phis = phi0 + np.linspace(0.0, angle, steps + 1)
    pts = center[None, :] + radius * np.stack((np.cos(phis), np.sin(phis)), axis=1)
    headings = h + np.linspace(0.0, angle, steps + 1)
    return pts, headings


def _solid_side(points: np.ndarray, interior: np.ndarray) -> int:
    """+1 if the solid (the side away from ``interior``) is on the left of the polyline direction, else -1."""
    i = len(points) // 2
    a, b = points[max(0, i - 1)], points[min(len(points) - 1, i + 1)]
    t = b - a
    m = 0.5 * (a + b)
    cross = t[0] * (interior[1] - m[1]) - t[1] * (interior[0] - m[0])
    return -1 if cross > 0 else 1


class LayoutBuilder:
    """Lays down road pieces and records route, surfaces, curbs, walls and markings."""

    def __init__(self, road, spacing: float | None = None):
        self.road = road
        self.lane = float(road.lane_width)
        self.W = 2.0 * self.lane
        self.sw = float(road.sidewalk_width)
        self.rc = float(road.junction_corner_radius)
        if self.rc <= self.sw:
            raise ValueError("junction_corner_radius must exceed sidewalk_width")
        self.cell = self.W / 2 + self.rc
        self.spacing = float(spacing or road.route_spacing)
        self.p = np.zeros(2)
        self.h = 0.0
        self.route_parts: list[np.ndarray] = []
        self.part_kinds: list[dict] = []
        self.curbs: list[np.ndarray] = []
        self.walls: list[Wall] = []
        self.strips: list[Strip] = []
        self.fans: list[Fan] = []
        self.center_lines: list[np.ndarray] = []
        self.edge_lines: list[np.ndarray] = []
        self.corridors: list[tuple[np.ndarray, int]] = []  # (axis samples, chain index) for the overlap check
        self.junction_specs: list[dict] = []
        self.first_axis_point = self.p.copy()
        self.first_heading = 0.0

    # ----------------------------------------------------------------------------------------- road bands
    def _band(self, axis: np.ndarray, headings: np.ndarray, dense_axis: np.ndarray, dense_h: np.ndarray, chain: int):
        W2 = self.W / 2
        lc, rc_ = offset_polyline(axis, headings, W2), offset_polyline(axis, headings, -W2)
        lw, rw = offset_polyline(axis, headings, W2 + self.sw), offset_polyline(axis, headings, -W2 - self.sw)
        self.strips.append(Strip(lc, rc_, "road"))
        self.strips.append(Strip(lw, lc, "sidewalk"))
        self.strips.append(Strip(rc_, rw, "sidewalk"))
        self.curbs += [lc, rc_]
        self.walls += [Wall(lw, 1), Wall(rw, -1)]
        inset = float(self.road.marking.edge_inset)
        self.center_lines.append(dense_axis)
        self.edge_lines += [
            offset_polyline(dense_axis, dense_h, W2 - inset),
            offset_polyline(dense_axis, dense_h, -W2 + inset),
        ]
        self.corridors.append((resample_polyline(dense_axis, 2.0), chain))

    def _end_wall(self, p: np.ndarray, h: float, facing_forward: bool):
        """Wall across the road and both sidewalks at ``p``; the solid lies ahead (forward) or behind."""
        half = self.W / 2 + self.sw
        n = left_normal(h)
        pts = np.stack((p - half * n, p + half * n))
        interior = p - unit(h) if facing_forward else p + unit(h)
        self.walls.append(Wall(pts, _solid_side(pts, interior)))

    # ------------------------------------------------------------------------------------------------ pieces
    def straight(self, length: float, chain: int):
        axis, hs = _sample_straight(self.p, self.h, length)
        dense, dh = _sample_straight(self.p, self.h, length, self.spacing)
        self._band(axis, hs, dense, dh, chain)
        self.route_parts.append(offset_polyline(dense, dh, -self.lane / 2))
        self.part_kinds.append({"kind": "straight"})
        self.p, self.h = axis[-1].copy(), float(hs[-1])

    def curve(self, radius: float, angle: float, chain: int):
        if radius < float(self.road.min_curve_radius) - 1e-9:
            raise ValueError(f"curve radius {radius} below min_curve_radius")
        steps = _arc_steps(angle, radius + self.W / 2 + self.sw)
        axis, hs = _sample_arc(self.p, self.h, radius, angle, steps)
        dense_steps = max(steps, int(math.ceil(abs(angle) * (radius + self.lane) / self.spacing)))
        dense, dh = _sample_arc(self.p, self.h, radius, angle, dense_steps)
        self._band(axis, hs, dense, dh, chain)
        self.route_parts.append(offset_polyline(dense, dh, -self.lane / 2))
        self.part_kinds.append({"kind": "curve", "radius": radius, "angle": angle})
        self.p, self.h = axis[-1].copy(), float(hs[-1])

    def junction(self, turn: str, chain: int, stub_length: float):
        if turn not in TURNS:
            raise ValueError(f"unknown turn '{turn}'")
        c, W2, lane, rc, sw = self.cell, self.W / 2, self.lane, self.rc, self.sw
        R = rot2(self.h)
        J = self.p + c * unit(self.h)

        def world(pts):
            return np.asarray(pts, dtype=np.float64) @ R.T + J

        # corner curbs / walls / sidewalks: corner (sx, sy) has its fillet centre at (sx c, sy c)
        corner_steps = _arc_steps(0.5 * math.pi, rc)  # same vertex count for curb and wall arcs (sidewalk strips)

        def corner_arc(sx, sy, radius, edge_offset):
            C = np.array([sx * c, sy * c])
            a = np.array([sx * c, sy * edge_offset]) - C  # on the x-arm side
            b = np.array([sx * edge_offset, sy * c]) - C  # on the y-arm side
            pa, pb = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
            d = wrap_angle(pb - pa)
            phis = pa + np.linspace(0.0, d, corner_steps + 1)
            return C + radius * np.stack((np.cos(phis), np.sin(phis)), axis=1)

        for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
            curb = corner_arc(sx, sy, rc, W2)
            wall = corner_arc(sx, sy, rc - sw, W2 + sw)
            self.curbs.append(world(curb))
            wall_w = world(wall)
            self.walls.append(Wall(wall_w, _solid_side(wall_w, J)))
            self.strips.append(Strip(world(wall), world(curb), "sidewalk"))
        # road surface polygon, counter-clockwise: the four corner curbs, joined by the arm openings at the cell edges
        ring = [
            _ccw(corner_arc(1, 1, rc, W2), start=np.array([c, W2])),
            _ccw(corner_arc(-1, 1, rc, W2), start=np.array([-W2, c])),
            _ccw(corner_arc(-1, -1, rc, W2), start=np.array([-c, -W2])),
            _ccw(corner_arc(1, -1, rc, W2), start=np.array([W2, -c])),
        ]
        self.fans.append(Fan(J.copy(), world(np.concatenate(ring, axis=0)), "road"))

        # route through the cell (local frame: incoming heading +x, right lane at y = -lane/2)
        if turn == "straight":
            n = max(1, int(math.ceil(2 * c / self.spacing)))
            xs = np.linspace(-c, c, n + 1)
            lane_pts = np.stack((xs, np.full_like(xs, -lane / 2)), axis=1)
            exit_local, exit_h = np.array([c, 0.0]), 0.0
        elif turn == "right":
            Rl = c - lane / 2
            n = max(2, int(math.ceil(0.5 * math.pi * Rl / self.spacing)))
            phis = np.linspace(0.5 * math.pi, 0.0, n + 1)
            lane_pts = np.array([-c, -c]) + Rl * np.stack((np.cos(phis), np.sin(phis)), axis=1)
            exit_local, exit_h = np.array([0.0, -c]), -0.5 * math.pi
        else:
            Rl = c + lane / 2
            n = max(2, int(math.ceil(0.5 * math.pi * Rl / self.spacing)))
            phis = np.linspace(-0.5 * math.pi, 0.0, n + 1)
            lane_pts = np.array([-c, c]) + Rl * np.stack((np.cos(phis), np.sin(phis)), axis=1)
            exit_local, exit_h = np.array([0.0, c]), 0.5 * math.pi
        self.route_parts.append(world(lane_pts))
        self.part_kinds.append({"kind": "junction", "turn": turn})
        self.junction_specs.append({"center": J.copy(), "heading_in": self.h, "turn": turn, "half": c})
        self.corridors.append((world(np.array([[-c, 0.0], [0.0, 0.0], [c, 0.0], [0.0, c], [0.0, -c]])), chain))

        # dead-end arms on the directions the route does not use (local directions: +x straight, +y left, -y right)
        used = {"straight": 0.0, "left": 0.5 * math.pi, "right": -0.5 * math.pi}[turn]
        for arm in (0.0, 0.5 * math.pi, -0.5 * math.pi):
            if abs(arm - used) < 1e-9:
                continue
            h_arm = self.h + arm
            start = J + c * unit(h_arm)
            axis, hs = _sample_straight(start, h_arm, stub_length)
            dense, dh = _sample_straight(start, h_arm, stub_length, self.spacing)
            self._band(axis, hs, dense, dh, chain)
            self._end_wall(axis[-1], h_arm, facing_forward=True)

        self.p = world(exit_local[None, :])[0]
        self.h = float(wrap_angle(self.h + exit_h))


def _ccw(arc: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Return the corner arc ordered so that it begins at the end closest to ``start``."""
    if np.linalg.norm(arc[0] - start) <= np.linalg.norm(arc[-1] - start):
        return arc
    return arc[::-1]


# ---------------------------------------------------------------------------------------------- assembly
def build_layout(pieces: list, road, name: str = "", track: int = 0, seed: int = 0, start_s: float = 3.0,
                 goal_margin: float = 12.0, check_overlap: bool = True) -> TrackLayout:
    """Build a :class:`TrackLayout` from piece specs such as ``[["straight", 40], ["curve", 30, 45]]``."""
    b = LayoutBuilder(road)
    stub = float(road.stub_length)
    first = True
    for i, spec in enumerate(pieces):
        kind = spec[0]
        if kind == "straight":
            b.straight(float(spec[1]), chain=i)
        elif kind == "curve":
            b.curve(float(spec[1]), math.radians(float(spec[2])), chain=i)
        elif kind == "junction":
            if first:
                raise ValueError("a layout must start with a straight or curve piece")
            b.junction(str(spec[1]), chain=i, stub_length=stub)
        else:
            raise ValueError(f"unknown piece '{kind}'")
        first = False
    b._end_wall(b.first_axis_point, b.first_heading, facing_forward=False)
    b._end_wall(b.p, b.h, facing_forward=True)

    # route: concatenate the lane parts and re-sample uniformly
    lengths = [float(arc_length(p)[-1]) for p in b.route_parts]
    joined = np.concatenate([b.route_parts[0]] + [p[1:] for p in b.route_parts[1:]], axis=0)
    route = resample_polyline(joined, b.spacing)
    s = arc_length(route)
    heading = polyline_headings(route)
    curvature = polyline_curvature(route, heading)
    total = float(s[-1])
    # piece boundaries in route arc length (scaled so the piece lengths add up to the re-sampled length)
    bounds = np.concatenate(([0.0], np.cumsum(lengths)))
    bounds *= total / bounds[-1]
    piece_list, junctions = [], []
    j = 0
    for k, kind in enumerate(b.part_kinds):
        p = Piece(kind["kind"], float(bounds[k]), float(bounds[k + 1]))
        if kind["kind"] == "curve":
            p.radius, p.angle = kind["radius"], kind["angle"]
        elif kind["kind"] == "junction":
            p.turn = kind["turn"]
            spec = b.junction_specs[j]
            j += 1
            junctions.append(
                Junction(spec["center"], spec["heading_in"], spec["turn"], spec["half"], p.s_start, p.s_end)
            )
        piece_list.append(p)

    next_m = np.full(len(s), MANEUVER_NONE, dtype=np.int64)
    dist_m = np.full(len(s), np.inf)
    for jn in reversed(junctions):  # nearest upcoming junction wins
        ahead = s <= jn.s_exit
        next_m[ahead] = TURNS.index(jn.turn)
        dist_m[ahead] = np.maximum(jn.s_entry - s[ahead], 0.0)

    goal_s = total - float(goal_margin)
    if goal_s <= start_s + 20.0:
        raise ValueError("route too short")
    axis_samples = np.concatenate([c[0] for c in b.corridors], axis=0)
    layout = TrackLayout(
        name=name,
        track=track,
        seed=seed,
        lane_width=b.lane,
        sidewalk_width=b.sw,
        route=route,
        route_heading=heading,
        route_s=s,
        route_curvature=curvature,
        next_maneuver=next_m,
        dist_to_maneuver=dist_m,
        pieces=piece_list,
        junctions=junctions,
        curbs=b.curbs,
        walls=b.walls,
        strips=b.strips,
        fans=b.fans,
        center_lines=b.center_lines,
        edge_lines=b.edge_lines,
        start_s=float(start_s),
        goal_s=goal_s,
        meta={"pieces": [list(p) for p in pieces], "axis_samples": axis_samples, "corridors": b.corridors},
    )
    if check_overlap:
        problem = overlap_problem(b.corridors, b.W / 2 + b.sw + float(road.wall_thickness), float(road.layout_clearance))
        if problem:
            raise LayoutOverlapError(problem)
    return layout


class LayoutOverlapError(ValueError):
    pass


def overlap_problem(corridors: list[tuple[np.ndarray, int]], half_width: float, clearance: float) -> str | None:
    """Check that road corridors of non-neighbouring pieces keep ``2 * half_width + clearance`` apart."""
    need = 2.0 * half_width + clearance
    for a in range(len(corridors)):
        pa, ca = corridors[a]
        for b in range(a + 1, len(corridors)):
            pb, cb = corridors[b]
            if abs(ca - cb) <= 2:
                continue
            d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min()
            if d < need:
                return f"pieces {ca} and {cb} are {d:.1f} m apart (< {need:.1f} m)"
    return None
