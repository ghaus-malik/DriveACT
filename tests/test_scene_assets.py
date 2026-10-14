"""USD generation (needs ``pxr``: Isaac Sim or ``pip install usd-core``; skipped otherwise)."""

import pytest

pxr = pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from driveact.environments.scene_builder import build_track, create_materials  # noqa: E402
from driveact.vehicle.car_usd import asset_paths  # noqa: E402


def test_car_articulation_structure(cfg, tmp_path):
    paths = asset_paths(cfg, str(tmp_path))
    stage = Usd.Stage.Open(paths["ego"])
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    assert set(joints) == {"steer_fl", "steer_fr", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"}
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 7
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    for name in ("steer_fl", "steer_fr"):
        assert joints[name].GetAttribute("physics:axis").Get() == "Z"
    for name in ("wheel_fl", "wheel_rr"):
        assert joints[name].GetAttribute("physics:axis").Get() == "Y"


def test_track_meshes(cfg, layouts):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mats = create_materials(stage)
    info = build_track(stage, "/World/envs/env_0", layouts[4], (0.0, 0.0), cfg, mats, decor=True)
    meshes = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)}
    for name in ("asphalt", "sidewalk", "curb", "wall", "mark_white", "mark_yellow"):
        assert name in meshes
    for name in ("sidewalk", "curb", "wall"):
        assert meshes[name].HasAPI(UsdPhysics.CollisionAPI)
    assert not meshes["asphalt"].HasAPI(UsdPhysics.CollisionAPI)
    assert len(info["buildings"]) > 10


def test_track_geometry_is_authored_in_world_coordinates(cfg, layouts):
    """The wall mesh must sit where the analytic walls are (layout frame + origin), with no inherited transform."""
    import numpy as np

    from driveact.environments.scene_builder import world_wall_bounds

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mats = create_materials(stage)
    origin = np.array([612.5, -305.0])
    layout = layouts[3]
    build_track(stage, "/World/Tracks/env_7", layout, origin, cfg, mats, decor=False)
    got = world_wall_bounds(stage, "/World/Tracks/env_7")
    pts = np.concatenate([w.points for w in layout.walls], axis=0) + origin
    want = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    assert max(abs(g - w) for g, w in zip(got, want)) <= cfg.road.wall_thickness + 0.05
