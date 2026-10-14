"""Procedural USD assets of a full-size passenger car (written with ``pxr``; no external meshes).

* :func:`write_ego_car_usd` - the ego vehicle as a physics articulation:

    chassis (rigid body, 2 box colliders: lower body + cabin)
      |- steer_fl / steer_fr   revolute joints about z (position drives)  -> knuckle_fl / knuckle_fr
      |      '- wheel_fl / wheel_fr   revolute joints about y (velocity drives) -> front wheels
      '- wheel_rl / wheel_rr   revolute joints about y (velocity drives) -> rear wheels

  Wheels are cylinders (collision + visual), all four driven (speed targets from Ackermann kinematics, see
  :mod:`driveact.vehicle.control`). The chassis frame origin is the footprint centre at ground level, so the root
  position is directly the vehicle position used everywhere else.

* :func:`write_parked_car_usd` - the same body as a single rigid body (made kinematic when spawned) for parked cars.
* :func:`write_cone_usd`, :func:`write_barrier_usd` - traffic cone and concrete (jersey) barrier rigid bodies.

Visual shapes are boxes/cylinders/cones with UsdPreviewSurface materials. Collision shapes are invisible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

PAINT_COLORS = {
    "white": (0.85, 0.86, 0.88),
    "red": (0.62, 0.05, 0.06),
    "blue": (0.06, 0.18, 0.52),
    "silver": (0.55, 0.57, 0.60),
    "black": (0.04, 0.04, 0.05),
    "green": (0.10, 0.32, 0.20),
    "yellow": (0.85, 0.62, 0.05),
}
EGO_COLOR = (0.80, 0.18, 0.02)  # orange-red ego car, easy to spot


def _stage(path: str, root: str):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    return stage


def _material(stage, path: str, color, roughness=0.5, metallic=0.0, opacity=1.0, emissive=None):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, static_friction: float, dynamic_friction: float, restitution: float = 0.0):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(float(restitution))
    return mat


def _bind(prim, mat, purpose: str | None = None):
    from pxr import UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose is None:
        api.Bind(mat)
    else:
        api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, purpose)


def _box(stage, path, center, size, mat=None, collision=False, visible=True):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(cube)
    xf.SetTranslate(Gf.Vec3d(*center))
    xf.SetScale(Gf.Vec3f(*size))
    if mat is not None:
        _bind(cube.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not visible:
        cube.MakeInvisible()
    return cube


def _cylinder(stage, path, center, radius, height, axis="Y", mat=None, collision=False, visible=True):
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(axis)
    UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(*center))
    if mat is not None:
        _bind(cyl.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    if not visible:
        cyl.MakeInvisible()
    return cyl


def _mass(prim, mass, inertia, com=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdPhysics

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*com))


def _translate(prim, pos):
    from pxr import Gf, UsdGeom

    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*pos))


# ----------------------------------------------------------------------------------------------- car body
def _body_visuals(stage, root: str, vcfg, paint, mats):
    """Car body made of boxes, relative to the footprint centre at ground level."""
    b = vcfg.body
    L, W, H = float(b.length), float(b.width), float(b.height)
    z0 = float(b.ground_clearance) + 0.07
    belt = 0.86  # top of the lower body
    _box(stage, f"{root}/body_lower", (0.0, 0.0, (z0 + belt) / 2), (L - 0.12, W, belt - z0), mats["paint"])
    _box(stage, f"{root}/hood", (L / 2 - 0.62, 0.0, belt + 0.03), (1.1, W - 0.10, 0.06), mats["paint"])
    _box(stage, f"{root}/trunk", (-L / 2 + 0.45, 0.0, belt + 0.04), (0.8, W - 0.10, 0.08), mats["paint"])
    cab_len, cab_x = 2.25, -0.18
    _box(stage, f"{root}/cabin_glass", (cab_x, 0.0, (belt + H - 0.05) / 2), (cab_len, W - 0.16, H - 0.05 - belt), mats["glass"])
    _box(stage, f"{root}/roof", (cab_x - 0.05, 0.0, H - 0.03), (cab_len - 0.35, W - 0.20, 0.06), mats["paint"])
    for side, y in (("l", 1), ("r", -1)):
        for name, x in (("a", cab_x + cab_len / 2 - 0.06), ("b", cab_x), ("c", cab_x - cab_len / 2 + 0.06)):
            _box(stage, f"{root}/pillar_{name}{side}", (x, y * (W / 2 - 0.09), (belt + H) / 2), (0.08, 0.04, H - belt), mats["paint"])
        _box(stage, f"{root}/mirror_{side}", (cab_x + cab_len / 2 + 0.05, y * (W / 2 + 0.06), belt + 0.12), (0.10, 0.14, 0.10), mats["paint"])
        _box(stage, f"{root}/headlight_{side}", (L / 2 - 0.04, y * (W / 2 - 0.28), 0.72), (0.06, 0.34, 0.12), mats["headlight"])
        _box(stage, f"{root}/taillight_{side}", (-L / 2 + 0.04, y * (W / 2 - 0.24), 0.76), (0.06, 0.30, 0.10), mats["taillight"])
    _box(stage, f"{root}/bumper_front", (L / 2 - 0.05, 0.0, z0 + 0.12), (0.12, W - 0.04, 0.22), mats["trim"])
    _box(stage, f"{root}/bumper_rear", (-L / 2 + 0.05, 0.0, z0 + 0.12), (0.12, W - 0.04, 0.22), mats["trim"])
    _box(stage, f"{root}/grille", (L / 2 - 0.02, 0.0, 0.58), (0.04, 0.70, 0.16), mats["trim"])
    _box(stage, f"{root}/plate_rear", (-L / 2 + 0.01, 0.0, 0.52), (0.02, 0.52, 0.12), mats["plate"])


def _body_colliders(stage, root: str, vcfg):
    b = vcfg.body
    L, W, H = float(b.length), float(b.width), float(b.height)
    z0 = float(b.ground_clearance)
    belt = 0.86
    _box(stage, f"{root}/collision_lower", (0.0, 0.0, (z0 + belt) / 2), (L, W, belt - z0), collision=True, visible=False)
    _box(stage, f"{root}/collision_cabin", (-0.18, 0.0, (belt + H) / 2), (2.25, W - 0.16, H - belt), collision=True, visible=False)


def _materials(stage, root: str, paint):
    return {
        "paint": _material(stage, f"{root}/Looks/paint", paint, roughness=0.32, metallic=0.4),
        "glass": _material(stage, f"{root}/Looks/glass", (0.08, 0.11, 0.14), roughness=0.05, metallic=0.6),
        "trim": _material(stage, f"{root}/Looks/trim", (0.07, 0.07, 0.08), roughness=0.6),
        "tyre": _material(stage, f"{root}/Looks/tyre", (0.03, 0.03, 0.03), roughness=0.95),
        "rim": _material(stage, f"{root}/Looks/rim", (0.70, 0.71, 0.73), roughness=0.25, metallic=0.9),
        "headlight": _material(stage, f"{root}/Looks/headlight", (0.95, 0.95, 0.90), roughness=0.1, emissive=(0.6, 0.6, 0.55)),
        "taillight": _material(stage, f"{root}/Looks/taillight", (0.70, 0.03, 0.03), roughness=0.2, emissive=(0.35, 0.0, 0.0)),
        "plate": _material(stage, f"{root}/Looks/plate", (0.92, 0.92, 0.88), roughness=0.5),
    }


def _wheel_visuals(stage, path: str, vcfg, mats):
    r, w = float(vcfg.body.wheel_radius), float(vcfg.body.wheel_width)
    _cylinder(stage, f"{path}/tyre", (0, 0, 0), r, w, "Y", mats["tyre"])
    _cylinder(stage, f"{path}/rim", (0, 0, 0), 0.62 * r, w + 0.01, "Y", mats["rim"])
    _box(stage, f"{path}/spoke_a", (0, 0, 0), (1.15 * r, w + 0.016, 0.07), mats["trim"])  # makes the spin visible
    _box(stage, f"{path}/spoke_b", (0, 0, 0), (0.07, w + 0.016, 1.15 * r), mats["trim"])


def _content_hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------------------------- ego vehicle
def write_ego_car_usd(path: str, vcfg, color=EGO_COLOR) -> str:
    from pxr import Gf, UsdPhysics

    stage = _stage(path, "/Car")
    b, m, d = vcfg.body, vcfg.mass, vcfg.drive
    L, W = float(b.length), float(b.width)
    r = float(b.wheel_radius)
    front_x = L / 2 - float(b.front_overhang)
    rear_x = front_x - float(b.wheelbase)
    half_track = float(b.track_width) / 2
    mats = _materials(stage, "/Car", color)
    tyre_phys = _physics_material(stage, "/Car/Looks/tyre_physics", vcfg.tyre.static_friction, vcfg.tyre.dynamic_friction)
    root = stage.GetPrimAtPath("/Car")
    UsdPhysics.ArticulationRootAPI.Apply(root)

    chassis = stage.DefinePrim("/Car/chassis", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(chassis)
    mc = float(m.chassis)
    _mass(chassis, mc, (mc * (W**2 + 1.2**2) / 12, mc * (L**2 + 1.2**2) / 12, mc * (L**2 + W**2) / 12),
          com=(0.0, 0.0, float(m.com_height)))
    _body_visuals(stage, "/Car/chassis/visual", vcfg, color, mats)
    _body_colliders(stage, "/Car/chassis", vcfg)

    wheels = {"fl": (front_x, half_track), "fr": (front_x, -half_track), "rl": (rear_x, half_track), "rr": (rear_x, -half_track)}
    mw = float(m.wheel)
    i_axle = 0.5 * mw * r * r
    i_other = mw * (3 * r * r + float(b.wheel_width) ** 2) / 12
    joints = stage.DefinePrim("/Car/joints", "Scope")
    del joints
    for name, (x, y) in wheels.items():
        wheel = stage.DefinePrim(f"/Car/wheel_{name}", "Xform")
        _translate(wheel, (x, y, r))
        UsdPhysics.RigidBodyAPI.Apply(wheel)
        _mass(wheel, mw, (i_other, i_axle, i_other))
        col = _cylinder(stage, f"/Car/wheel_{name}/collision", (0, 0, 0), r, float(b.wheel_width), "Y", collision=True, visible=False)
        _bind(col.GetPrim(), tyre_phys, "physics")
        _wheel_visuals(stage, f"/Car/wheel_{name}/visual", vcfg, mats)
        parent = "/Car/chassis"
        parent_pos = (x, y, r)
        if name.startswith("f"):
            knuckle = stage.DefinePrim(f"/Car/knuckle_{name}", "Xform")
            _translate(knuckle, (x, y, r))
            UsdPhysics.RigidBodyAPI.Apply(knuckle)
            mk = float(m.knuckle)
            _mass(knuckle, mk, (0.05 * mk, 0.05 * mk, 0.05 * mk))
            steer = UsdPhysics.RevoluteJoint.Define(stage, f"/Car/joints/steer_{name}")
            steer.CreateBody0Rel().SetTargets(["/Car/chassis"])
            steer.CreateBody1Rel().SetTargets([f"/Car/knuckle_{name}"])
            steer.CreateLocalPos0Attr(Gf.Vec3f(x, y, r))
            steer.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            steer.CreateAxisAttr("Z")
            lim = float(vcfg.steering.knuckle_limit_deg)
            steer.CreateLowerLimitAttr(-lim)
            steer.CreateUpperLimitAttr(lim)
            drive = UsdPhysics.DriveAPI.Apply(steer.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(d.steer_stiffness) * math.pi / 180.0)  # USD angular gains are per degree
            drive.CreateDampingAttr(float(d.steer_damping) * math.pi / 180.0)
            drive.CreateMaxForceAttr(float(d.steer_effort))
            parent, parent_pos = f"/Car/knuckle_{name}", (0.0, 0.0, 0.0)
        spin = UsdPhysics.RevoluteJoint.Define(stage, f"/Car/joints/wheel_{name}")
        spin.CreateBody0Rel().SetTargets([parent])
        spin.CreateBody1Rel().SetTargets([f"/Car/wheel_{name}"])
        spin.CreateLocalPos0Attr(Gf.Vec3f(*parent_pos))
        spin.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        spin.CreateAxisAttr("Y")
        drive = UsdPhysics.DriveAPI.Apply(spin.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(float(d.wheel_damping) * math.pi / 180.0)
        drive.CreateMaxForceAttr(float(d.wheel_effort))
    stage.GetRootLayer().Save()
    return path


# ------------------------------------------------------------------------------------------- static props
def write_parked_car_usd(path: str, vcfg, color) -> str:
    from pxr import UsdPhysics

    stage = _stage(path, "/ParkedCar")
    b = vcfg.body
    L, r = float(b.length), float(b.wheel_radius)
    front_x = L / 2 - float(b.front_overhang)
    rear_x = front_x - float(b.wheelbase)
    half_track = float(b.track_width) / 2
    root = stage.GetPrimAtPath("/ParkedCar")
    UsdPhysics.RigidBodyAPI.Apply(root)
    _mass(root, float(vcfg.mass.chassis), (1.0, 1.0, 1.0))
    mats = _materials(stage, "/ParkedCar", color)
    _body_visuals(stage, "/ParkedCar/visual", vcfg, color, mats)
    _body_colliders(stage, "/ParkedCar", vcfg)
    for name, (x, y) in {"fl": (front_x, half_track), "fr": (front_x, -half_track), "rl": (rear_x, half_track),
                         "rr": (rear_x, -half_track)}.items():
        wheel = stage.DefinePrim(f"/ParkedCar/visual/wheel_{name}", "Xform")
        _translate(wheel, (x, y, r))
        _wheel_visuals(stage, f"/ParkedCar/visual/wheel_{name}", vcfg, mats)
    stage.GetRootLayer().Save()
    return path


def write_cone_usd(path: str, ocfg) -> str:
    from pxr import Gf, UsdGeom, UsdPhysics

    t = ocfg.cone_taper
    h, rad = float(t.height), float(t.radius)
    stage = _stage(path, "/Cone")
    root = stage.GetPrimAtPath("/Cone")
    UsdPhysics.RigidBodyAPI.Apply(root)
    _mass(root, 4.0, (0.1, 0.1, 0.1))
    orange = _material(stage, "/Cone/Looks/orange", (0.95, 0.30, 0.02), roughness=0.55)
    white = _material(stage, "/Cone/Looks/white", (0.95, 0.95, 0.95), roughness=0.3, emissive=(0.15, 0.15, 0.15))
    black = _material(stage, "/Cone/Looks/black", (0.05, 0.05, 0.05), roughness=0.9)
    _box(stage, "/Cone/base", (0, 0, 0.02), (2.1 * rad, 2.1 * rad, 0.04), black)
    cone = UsdGeom.Cone.Define(stage, "/Cone/body")
    cone.CreateRadiusAttr(0.9 * rad)
    cone.CreateHeightAttr(h - 0.04)
    cone.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(cone).SetTranslate(Gf.Vec3d(0, 0, 0.04 + (h - 0.04) / 2))
    _bind(cone.GetPrim(), orange)
    for k, zf in enumerate((0.45, 0.65)):  # reflective bands
        rr = 0.9 * rad * (1 - zf) + 0.004
        _cylinder(stage, f"/Cone/band_{k}", (0, 0, 0.04 + zf * (h - 0.04)), rr, 0.07, "Z", white)
    _cylinder(stage, "/Cone/collision", (0, 0, h / 2), rad, h, "Z", collision=True, visible=False)
    stage.GetRootLayer().Save()
    return path


def write_barrier_usd(path: str, ocfg) -> str:
    """Concrete jersey barrier: trapezoidal profile extruded along x (``length``), collision box."""
    from pxr import Gf, UsdGeom, UsdPhysics

    t = ocfg.barrier
    L, Wd, H = float(t.length), float(t.width), float(t.height)
    stage = _stage(path, "/Barrier")
    root = stage.GetPrimAtPath("/Barrier")
    UsdPhysics.RigidBodyAPI.Apply(root)
    _mass(root, 900.0, (100.0, 100.0, 100.0))
    concrete = _material(stage, "/Barrier/Looks/concrete", (0.72, 0.71, 0.68), roughness=0.85)
    stripe = _material(stage, "/Barrier/Looks/stripe", (0.95, 0.75, 0.05), roughness=0.4, emissive=(0.2, 0.15, 0.0))
    # profile (y, z): wide base, sloped faces, narrow top
    prof = [(-Wd / 2, 0.0), (Wd / 2, 0.0), (Wd / 2, 0.08), (0.17, 0.33), (0.10, H), (-0.10, H), (-0.17, 0.33), (-Wd / 2, 0.08)]
    n = len(prof)
    pts = [Gf.Vec3f(x, y, z) for x in (-L / 2, L / 2) for (y, z) in prof]
    faces, counts = [], []
    for i in range(n):  # sides
        j = (i + 1) % n
        faces += [i, j, n + j, n + i]
        counts.append(4)
    faces += list(range(n - 1, -1, -1)) + list(range(n, 2 * n))  # end caps
    counts += [n, n]
    mesh = UsdGeom.Mesh.Define(stage, "/Barrier/body")
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh.GetPrim(), concrete)
    _box(stage, "/Barrier/stripe_l", (0.0, 0.135, 0.55), (L * 0.6, 0.02, 0.10), stripe)
    _box(stage, "/Barrier/stripe_r", (0.0, -0.135, 0.55), (L * 0.6, 0.02, 0.10), stripe)
    _box(stage, "/Barrier/collision", (0, 0, H / 2), (L, Wd, H), collision=True, visible=False)
    stage.GetRootLayer().Save()
    return path


# ------------------------------------------------------------------------------------------------- cache
def asset_paths(cfg, cache_dir: str) -> dict:
    """Write (if missing) all procedural assets for the current configuration and return their paths."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _content_hash(cfg.vehicle.to_dict(), cfg.obstacles.to_dict(), EGO_COLOR, 4)
    paths = {"ego": os.path.join(cache_dir, f"ego_car_{key}.usda"),
             "cone": os.path.join(cache_dir, f"cone_{key}.usda"),
             "barrier": os.path.join(cache_dir, f"barrier_{key}.usda"),
             "parked": []}
    if not os.path.exists(paths["ego"]):
        write_ego_car_usd(paths["ego"], cfg.vehicle)
    if not os.path.exists(paths["cone"]):
        write_cone_usd(paths["cone"], cfg.obstacles)
    if not os.path.exists(paths["barrier"]):
        write_barrier_usd(paths["barrier"], cfg.obstacles)
    for name, color in PAINT_COLORS.items():
        p = os.path.join(cache_dir, f"parked_car_{name}_{key}.usda")
        if not os.path.exists(p):
            write_parked_car_usd(p, cfg.vehicle, color)
        paths["parked"].append(p)
    return paths
