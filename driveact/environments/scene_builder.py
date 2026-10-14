"""USD geometry of a track layout: road surface, lane markings, curbs/sidewalks, walls, start/goal and decoration.

All geometry is generated from the same :class:`TrackLayout` that drives the LiDAR, collision checks and expert, so
what is rendered and simulated matches what is sensed. Static colliders: sidewalk tops + curb faces and walls
(triangle meshes). The road itself is visual; the car drives on the global ground plane (z = 0).

Decoration (buildings, trees, street lamps, start/goal gantries) is placed only outside the wall line, never
inside the drivable area or on the sidewalk, so it cannot change what the planar LiDAR would see at its scan height
(walls occlude everything behind them). It is added in visual mode and optionally headless.

Everything is batched into a few meshes per environment (one per material) to keep the prim count low.
"""

from __future__ import annotations

import math

import numpy as np

from driveact.geometry import left_normal, offset_polyline, polyline_headings, rot2

MATERIALS = {
    "asphalt": dict(color=(0.065, 0.065, 0.07), roughness=0.95),
    "sidewalk": dict(color=(0.36, 0.35, 0.33), roughness=0.9),
    "curb": dict(color=(0.55, 0.54, 0.52), roughness=0.85),
    "wall": dict(color=(0.42, 0.41, 0.38), roughness=0.92),
    "grass": dict(color=(0.13, 0.24, 0.08), roughness=0.95),
    "mark_white": dict(color=(0.92, 0.92, 0.90), roughness=0.6),
    "mark_yellow": dict(color=(0.93, 0.70, 0.08), roughness=0.6),
    "checker_dark": dict(color=(0.05, 0.05, 0.05), roughness=0.6),
    "facade_0": dict(color=(0.55, 0.30, 0.22), roughness=0.85),  # brick
    "facade_1": dict(color=(0.80, 0.74, 0.62), roughness=0.85),  # sandstone
    "facade_2": dict(color=(0.62, 0.64, 0.66), roughness=0.7),  # concrete
    "facade_3": dict(color=(0.86, 0.86, 0.84), roughness=0.75),  # white render
    "facade_4": dict(color=(0.32, 0.40, 0.48), roughness=0.4, metallic=0.3),  # blue cladding
    "window": dict(color=(0.10, 0.14, 0.18), roughness=0.08, metallic=0.7),
    "roof": dict(color=(0.25, 0.25, 0.26), roughness=0.9),
    "trunk": dict(color=(0.28, 0.19, 0.12), roughness=0.95),
    "leaves_0": dict(color=(0.16, 0.36, 0.12), roughness=0.9),
    "leaves_1": dict(color=(0.24, 0.42, 0.14), roughness=0.9),
    "pole": dict(color=(0.30, 0.31, 0.33), roughness=0.5, metallic=0.6),
    "lamp": dict(color=(1.0, 0.95, 0.80), roughness=0.3, emissive=(1.0, 0.92, 0.7)),
    "start": dict(color=(0.10, 0.65, 0.25), roughness=0.5),
    "goal": dict(color=(0.85, 0.12, 0.10), roughness=0.5),
}


# --------------------------------------------------------------------------------------------- mesh batching
class MeshBatch:
    """Accumulates triangles/quads for one material and emits a single UsdGeom.Mesh."""

    def __init__(self):
        self.points: list[np.ndarray] = []
        self.counts: list[np.ndarray] = []
        self.indices: list[np.ndarray] = []
        self.n = 0

    def add(self, points: np.ndarray, faces: list[list[int]]) -> None:
        points = np.asarray(points, dtype=np.float64)
        self.points.append(points)
        self.counts.append(np.array([len(f) for f in faces], dtype=np.int32))
        self.indices.append(np.concatenate([np.asarray(f, dtype=np.int32) for f in faces]) + self.n)
        self.n += len(points)

    def empty(self) -> bool:
        return self.n == 0

    # ---- primitives
    def strip(self, left: np.ndarray, right: np.ndarray, z: float | np.ndarray = 0.0) -> None:
        K = len(left)
        zl = np.broadcast_to(np.asarray(z, dtype=np.float64), (K,))
        pts = np.concatenate((np.column_stack((left, zl)), np.column_stack((right, zl))))
        self.add(pts, [[i, K + i, K + i + 1, i + 1] for i in range(K - 1)])

    def fan(self, center: np.ndarray, ring: np.ndarray, z: float) -> None:
        K = len(ring)
        pts = np.concatenate(([[center[0], center[1], z]], np.column_stack((ring, np.full(K, z)))))
        self.add(pts, [[0, 1 + i, 1 + (i + 1) % K] for i in range(K)])

    def wall_band(self, line: np.ndarray, z0: float, z1: float) -> None:
        """Vertical band along a polyline."""
        K = len(line)
        pts = np.concatenate((np.column_stack((line, np.full(K, z0))), np.column_stack((line, np.full(K, z1)))))
        self.add(pts, [[i, i + 1, K + i + 1, K + i] for i in range(K - 1)])

    def prism(self, inner: np.ndarray, outer: np.ndarray, z0: float, z1: float) -> None:
        """Closed solid between two polylines (same vertex count) from z0 to z1."""
        K = len(inner)
        P = np.concatenate([np.column_stack((inner, np.full(K, z0))), np.column_stack((outer, np.full(K, z0))),
                            np.column_stack((inner, np.full(K, z1))), np.column_stack((outer, np.full(K, z1)))])
        i0, o0, i1, o1 = 0, K, 2 * K, 3 * K
        faces = []
        for k in range(K - 1):
            faces += [[i0 + k, i0 + k + 1, i1 + k + 1, i1 + k], [o0 + k, o1 + k, o1 + k + 1, o0 + k + 1],
                      [i1 + k, i1 + k + 1, o1 + k + 1, o1 + k], [i0 + k, o0 + k, o0 + k + 1, i0 + k + 1]]
        faces += [[i0, i1, o1, o0], [i0 + K - 1, o0 + K - 1, o1 + K - 1, i1 + K - 1]]
        self.add(P, faces)

    def box(self, center, size, yaw: float = 0.0) -> None:
        cx, cy, cz = center
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        R = rot2(yaw)
        xy = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ R.T + np.array([cx, cy])
        pts = np.concatenate((np.column_stack((xy, np.full(4, cz - hz))), np.column_stack((xy, np.full(4, cz + hz)))))
        self.add(pts, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])

    def cylinder(self, center, radius: float, height: float, segments: int = 10) -> None:
        cx, cy, cz = center
        a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring = np.column_stack((cx + radius * np.cos(a), cy + radius * np.sin(a)))
        pts = np.concatenate((np.column_stack((ring, np.full(segments, cz - height / 2))),
                              np.column_stack((ring, np.full(segments, cz + height / 2)))))
        faces = [[k, (k + 1) % segments, segments + (k + 1) % segments, segments + k] for k in range(segments)]
        faces += [list(range(segments - 1, -1, -1)), list(range(segments, 2 * segments))]
        self.add(pts, faces)

    def sphere(self, center, radius: float, rings: int = 5, segments: int = 8, squash: float = 1.0) -> None:
        cx, cy, cz = center
        pts = [[cx, cy, cz + radius * squash]]
        for r in range(1, rings):
            th = np.pi * r / rings
            for s in range(segments):
                ph = 2 * np.pi * s / segments
                pts.append([cx + radius * np.sin(th) * np.cos(ph), cy + radius * np.sin(th) * np.sin(ph),
                            cz + radius * squash * np.cos(th)])
        pts.append([cx, cy, cz - radius * squash])
        faces = []
        bottom = len(pts) - 1
        for s in range(segments):
            faces.append([0, 1 + s, 1 + (s + 1) % segments])
        for r in range(rings - 2):
            a0, b0 = 1 + r * segments, 1 + (r + 1) * segments
            for s in range(segments):
                faces.append([a0 + s, b0 + s, b0 + (s + 1) % segments, a0 + (s + 1) % segments])
        last = 1 + (rings - 2) * segments
        for s in range(segments):
            faces.append([bottom, last + (s + 1) % segments, last + s])
        self.add(np.asarray(pts), faces)

    # ---- USD
    def emit(self, stage, path: str, material, offset=(0.0, 0.0, 0.0), collision: bool = False):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt

        if self.empty():
            return None
        pts = np.concatenate(self.points) + np.asarray(offset, dtype=np.float64)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts.astype(np.float32).tolist()]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(np.concatenate(self.counts).tolist()))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.concatenate(self.indices).tolist()))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        return mesh


def create_materials(stage, root: str = "/World/Looks") -> dict:
    from driveact.vehicle.car_usd import _material

    return {name: _material(stage, f"{root}/{name}", **spec) for name, spec in MATERIALS.items()}


# ------------------------------------------------------------------------------------------ track geometry
def _dash_polylines(line: np.ndarray, dash: float, gap: float, step: float = 0.5) -> list[np.ndarray]:
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
    out, t = [], 0.0
    while t < s[-1] - 0.2:
        ts = np.arange(t, min(t + dash, s[-1]) + 1e-9, step)
        if len(ts) < 2:
            ts = np.array([t, min(t + dash, s[-1])])
        out.append(np.column_stack((np.interp(ts, s, line[:, 0]), np.interp(ts, s, line[:, 1]))))
        t += dash + gap
    return out


def _band(batch: MeshBatch, line: np.ndarray, width: float, z: float) -> None:
    if len(line) < 2:
        return
    h = polyline_headings(line)
    batch.strip(offset_polyline(line, h, width / 2), offset_polyline(line, h, -width / 2), z)


def track_batches(layout, road_cfg) -> dict[str, tuple[MeshBatch, bool]]:
    """{name: (batch, is_collider)} for the static track geometry in layout coordinates."""
    curb_h = float(road_cfg.curb_height)
    wall_h = float(road_cfg.wall_height)
    wall_t = float(road_cfg.wall_thickness)
    mk = road_cfg.marking
    road, side, curb, wall = MeshBatch(), MeshBatch(), MeshBatch(), MeshBatch()
    white, yellow, dark = MeshBatch(), MeshBatch(), MeshBatch()
    for strip in layout.strips:
        if strip.kind == "road":
            road.strip(strip.left, strip.right, 0.004)
        else:
            side.strip(strip.left, strip.right, curb_h)
    for fan in layout.fans:
        road.fan(fan.center, fan.boundary, 0.004)
    for c in layout.curbs:
        curb.wall_band(c, 0.0, curb_h)
    for w in layout.walls:
        h = polyline_headings(w.points)
        outer = offset_polyline(w.points, h, w.side * wall_t)
        wall.prism(w.points, outer, 0.0, wall_h)
    for line in layout.center_lines:
        for dash in _dash_polylines(line, float(mk.dash_length), float(mk.dash_gap)):
            _band(yellow, dash, float(mk.width), 0.010)
    for line in layout.edge_lines:
        _band(white, line, float(mk.width), 0.010)
    _start_goal_markings(layout, white, dark)
    return {"asphalt": (road, False), "sidewalk": (side, True), "curb": (curb, True), "wall": (wall, True),
            "mark_white": (white, False), "mark_yellow": (yellow, False), "checker_dark": (dark, False)}


def _start_goal_markings(layout, white: MeshBatch, dark: MeshBatch) -> None:
    lane = layout.lane_width
    # start line across the right lane, at the front bumper of the start pose
    x, y, h = layout.pose_at(layout.start_s + 2.6, 0.0)
    white.box((x, y, 0.010), (0.4, lane - 0.2, 0.004), h)
    # chequered goal band across the whole road
    n_across, n_along, sq = 10, 3, 2 * lane / 10
    for i in range(n_along):
        for j in range(n_across):
            d = -lane / 2 + (j + 0.5) * sq
            gx, gy, gh = layout.pose_at(layout.goal_s + (i - 1) * sq, d)
            (dark if (i + j) % 2 else white).box((gx, gy, 0.011), (sq, sq, 0.004), gh)


# ---------------------------------------------------------------------------------------------- decoration
def _clear_of_roads(points: np.ndarray, axis: np.ndarray, min_dist: float) -> bool:
    d = np.linalg.norm(points[:, None, :] - axis[None, :, :], axis=-1).min()
    return bool(d >= min_dist)


def decor_batches(layout, road_cfg, rng: np.random.Generator) -> tuple[dict[str, MeshBatch], dict]:
    """Buildings, trees, street lamps and start/goal gantries behind the walls. Returns batches + footprints."""
    half_road = layout.lane_width  # two lanes: the road axis is one lane width from either curb
    wall_line = half_road + layout.sidewalk_width + float(road_cfg.wall_thickness)  # wall outer face from the axis
    axis = layout.axis_samples
    batches = {name: MeshBatch() for name in MATERIALS if name.startswith(("facade", "window", "roof", "trunk", "leaves",
                                                                         "pole", "lamp", "start", "goal"))}
    placed: list[np.ndarray] = []  # building footprints (4, 2)
    info = {"buildings": []}
    for line in layout.center_lines:
        h = polyline_headings(line)
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
        for sign in (1.0, -1.0):
            t = float(rng.uniform(2.0, 10.0))
            while t < s[-1] - 4.0:
                width = float(rng.uniform(12.0, 26.0))
                depth = float(rng.uniform(10.0, 18.0))
                height = float(rng.choice([7.5, 10.5, 13.5, 16.5, 22.5, 28.5]))
                setback = float(rng.uniform(4.0, 8.0))
                tc = min(t + width / 2, s[-1])
                p = np.array([np.interp(tc, s, line[:, 0]), np.interp(tc, s, line[:, 1])])
                hh = float(np.arctan2(np.interp(tc, s, np.sin(h)), np.interp(tc, s, np.cos(h))))
                n = left_normal(hh) * sign
                center = p + n * (wall_line + setback + depth / 2)
                R = rot2(hh)
                corners = np.array([[-width / 2, -depth / 2], [width / 2, -depth / 2], [width / 2, depth / 2],
                                    [-width / 2, depth / 2]]) @ R.T + center
                probe = np.concatenate((corners, [center], 0.5 * (corners + np.roll(corners, 1, axis=0))))
                ok = _clear_of_roads(probe, axis, wall_line + 2.0)
                if ok:
                    for other in placed:
                        if np.linalg.norm(other.mean(0) - center) < 0.5 * (width + depth) + 4.0:
                            ok = False
                            break
                if ok:
                    placed.append(corners)
                    _building(batches, rng, center, width, depth, height, hh, sign)
                    info["buildings"].append({"corners": corners, "height": height})
                    # a tree and a lamp between wall and building
                    if rng.random() < 0.7:
                        tp = p + n * (wall_line + 0.45 * setback)
                        _tree(batches, rng, tp)
                t += width + float(rng.uniform(3.0, 9.0))
            # street lamps directly behind the wall every ~32 m
            for tl in np.arange(float(rng.uniform(5, 20)), s[-1], 32.0):
                p = np.array([np.interp(tl, s, line[:, 0]), np.interp(tl, s, line[:, 1])])
                hh = float(np.arctan2(np.interp(tl, s, np.sin(h)), np.interp(tl, s, np.cos(h))))
                n = left_normal(hh) * sign
                base = p + n * (wall_line + 0.4)
                if _clear_of_roads(base[None], axis, wall_line + 0.2):
                    _lamp(batches, base, -n)
    _gantry(batches["start"], layout, layout.start_s + 2.6, wall_line)
    _gantry(batches["goal"], layout, layout.goal_s, wall_line)
    return batches, info


def _building(batches, rng, center, width, depth, height, yaw, sign):
    facade = f"facade_{int(rng.integers(0, 5))}"
    batches[facade].box((center[0], center[1], height / 2), (width, depth, height), yaw)
    batches["roof"].box((center[0], center[1], height + 0.3), (width - 0.6, depth - 0.6, 0.6), yaw)
    # window bands on all four sides
    floors = int(height // 3.0)
    R = rot2(yaw)
    for f in range(floors):
        z = 1.9 + 3.0 * f
        if z + 0.8 > height - 0.4:
            break
        for nx, ny, length in ((0, 1, width), (0, -1, width), (1, 0, depth), (-1, 0, depth)):
            off = np.array([nx * (width / 2 + 0.03), ny * (depth / 2 + 0.03)]) @ R.T
            size = (length - 1.6, 0.06, 1.4) if ny else (0.06, length - 1.6, 1.4)
            batches["window"].box((center[0] + off[0], center[1] + off[1], z), size, yaw)


def _tree(batches, rng, p):
    trunk_h = float(rng.uniform(2.2, 3.2))
    batches["trunk"].cylinder((p[0], p[1], trunk_h / 2), 0.16, trunk_h, 8)
    r = float(rng.uniform(1.6, 2.6))
    leaves = f"leaves_{int(rng.integers(0, 2))}"
    batches[leaves].sphere((p[0], p[1], trunk_h + 0.7 * r), r, squash=1.15)
    batches[leaves].sphere((p[0] + 0.5, p[1] - 0.3, trunk_h + 1.3 * r), 0.7 * r)


def _lamp(batches, base, towards_road):
    batches["pole"].cylinder((base[0], base[1], 3.5), 0.09, 7.0, 8)
    arm_end = base + 1.8 * towards_road
    mid = 0.5 * (base + arm_end)
    yaw = math.atan2(towards_road[1], towards_road[0])
    batches["pole"].box((mid[0], mid[1], 6.95), (1.8, 0.08, 0.08), yaw)
    batches["lamp"].box((arm_end[0], arm_end[1], 6.85), (0.6, 0.3, 0.12), yaw)


def _gantry(batch: MeshBatch, layout, s: float, wall_line: float):
    """Arch over the whole road at route position ``s``; the posts stand behind the walls."""
    x, y, h = layout.pose_at(s, layout.lane_width / 2)  # on the road axis
    n = left_normal(h)
    for side in (-1.0, 1.0):
        p = np.array([x, y]) + side * (wall_line + 0.5) * n
        batch.box((p[0], p[1], 2.9), (0.4, 0.4, 5.8), h)
    batch.box((x, y, 5.6), (0.6, 2 * wall_line + 1.4, 0.9), h)


def add_lights(stage, cfg) -> None:
    """Sky (HDRI dome) and sun with shadows."""
    import isaaclab.sim as sim_utils

    v = cfg.visual
    tex = str(v.sky_texture) if v.get("sky_texture") else None
    if tex and "{" in tex:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

        tex = tex.format(ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR)
    dome = sim_utils.DomeLightCfg(intensity=float(v.dome_intensity), texture_file=tex, color=(1.0, 1.0, 1.0))
    dome.func("/World/sky", dome)
    sun = sim_utils.DistantLightCfg(intensity=float(v.sun_intensity), angle=float(v.sun_angle), color=tuple(v.sun_color))
    yaw, elev = math.radians(float(v.sun_azimuth_deg)), math.radians(float(v.sun_elevation_deg))
    # distant lights shine along their -z axis: rotate -z to point from the sun towards the ground
    tilt = math.pi / 2 - elev
    qw, qx, qy, qz = _quat_mul(_quat_z(yaw), _quat_y(tilt))
    sun.func("/World/sun", sun, orientation=(qw, qx, qy, qz))


def _quat_z(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def _quat_y(a):
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def build_track(stage, root_path: str, layout, origin, cfg, materials: dict, decor: bool, seed: int = 0) -> dict:
    """Author the static geometry of one environment under ``root_path`` with points in WORLD coordinates
    (layout frame + ``origin``). ``root_path`` must not inherit a transform (do not use the cloned
    ``/World/envs/env_i`` prims, which carry the environment's grid translation)."""
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, root_path)
    UsdGeom.Xform.Define(stage, f"{root_path}/Track")
    offset = (float(origin[0]), float(origin[1]), 0.0)
    for name, (batch, collider) in track_batches(layout, cfg.road).items():
        batch.emit(stage, f"{root_path}/Track/{name}", materials[name], offset, collision=collider)
    # grass under the whole track
    x0, y0, x1, y1 = layout.bounds(margin=float(cfg.sim.env_spacing) / 2)
    grass = MeshBatch()
    grass.strip(np.array([[x0, y1], [x1, y1]]), np.array([[x0, y0], [x1, y0]]), 0.001)  # above the ground plane
    grass.emit(stage, f"{root_path}/Track/grass", materials["grass"], offset)
    info = {}
    if decor:
        batches, info = decor_batches(layout, cfg.road, np.random.default_rng([seed, 4242]))
        for name, batch in batches.items():
            batch.emit(stage, f"{root_path}/Decor/{name}", materials[name], offset)
    return info


def world_wall_bounds(stage, root_path: str) -> tuple[float, float, float, float] | None:
    """World-space x/y bounds of the authored wall mesh (used to verify that physics and analytic geometry agree)."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(f"{root_path}/Track/wall")
    if not prim.IsValid():
        return None
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])
