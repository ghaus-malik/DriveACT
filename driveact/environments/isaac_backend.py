"""Isaac Lab physics backend: articulated full-size car, static track geometry and kinematic obstacles.

Must be imported after the Isaac Sim app has been launched (see :mod:`driveact.utils.launch`).

* The car (``driveact/vehicle/car_usd.py``) is an articulation: steering knuckles with position drives, four wheels
  with velocity drives. Every physics step (100 Hz) the shared actuator model (:mod:`driveact.vehicle.control`)
  converts the 10 Hz action into Ackermann knuckle targets and per-wheel speed targets; the car moves only through
  tyre-ground contact forces computed by PhysX. Nothing is teleported except at episode resets.
* Each environment has its own static track geometry (different layouts per environment are allowed): curbs and
  sidewalks and walls are triangle-mesh colliders, the road is drawn on top of the global ground plane.
* Obstacles are pre-spawned kinematic rigid bodies (a fixed number of slots per type); at every reset the slots
  are moved to the sampled obstacle poses, unused slots are parked below the ground.
* A contact sensor on the chassis reports physical contacts (an additional signal; episode outcomes come from the
  exact geometric checks in :class:`DriveEnv`, which use the same geometry).

Coordinates passed in/out are each environment's local frame (layout frame); env origins are added internally.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

from driveact.environments.drive_env import obstacle_capacity
from driveact.environments.placement import centered_origins, check_no_overlap
from driveact.environments.scene_builder import add_lights, build_track, create_materials, world_wall_bounds
from driveact.vehicle.car_usd import PAINT_COLORS, asset_paths
from driveact.vehicle.control import (
    Actuators,
    VehicleGeometry,
    ackermann_angles,
    actions_to_physical,
    bicycle_angle,
    wheel_angular_speeds,
)

DEFAULT_CACHE = os.path.expanduser("~/.cache/driveact/usd")
STEER_JOINTS = ["steer_fl", "steer_fr"]
WHEEL_JOINTS = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
HIDDEN_Z = -25.0


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw -> (N, 4) quaternion (w, x, y, z)."""
    z = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), dim=-1)


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class IsaacBackend:
    name = "isaac"

    def __init__(self, cfg, num_envs: int, layouts, track_cfg, device: str = "cuda:0", visual: bool = False,
                 decor: bool = False, cache_dir: str = DEFAULT_CACHE, spawn_xy=(12.0, -1.75), build_static: bool = True):
        self.cfg = cfg
        self.num_envs = num_envs
        self.layouts = layouts
        self.visual = visual
        self.g = VehicleGeometry.from_config(cfg.vehicle)
        self.physics_dt = float(cfg.sim.physics_dt)
        self.decimation = int(round(float(cfg.sim.control_dt) / self.physics_dt))
        self.render_every = int(cfg.sim.render_every)
        render_cfg = sim_utils.RenderCfg(antialiasing_mode=str(cfg.visual.antialiasing)) if visual else sim_utils.RenderCfg()
        tyre = cfg.vehicle.tyre
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.physics_dt,
            render_interval=self.render_every if visual else self.decimation,
            device=device,
            physx=sim_utils.PhysxCfg(bounce_threshold_velocity=0.5, enable_stabilization=False),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=float(tyre.static_friction),
                                                            dynamic_friction=float(tyre.dynamic_friction)),
            render=render_cfg,
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.device = torch.device(self.sim.device)
        assets = asset_paths(cfg, cache_dir)
        d = cfg.vehicle.drive

        scene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=float(cfg.sim.env_spacing), replicate_physics=False)
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                size=(5000.0, 5000.0), visible=False,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=float(tyre.static_friction),
                                                                dynamic_friction=float(tyre.dynamic_friction)),
            ),
        )
        scene_cfg.car = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Car",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets["ego"],
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=2.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=4
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(spawn_xy[0], spawn_xy[1], 0.03), joint_pos={".*": 0.0},
                                                       joint_vel={".*": 0.0}),
            actuators={
                "steering": ImplicitActuatorCfg(
                    joint_names_expr=["steer_.*"], stiffness=float(d.steer_stiffness), damping=float(d.steer_damping),
                    effort_limit_sim=float(d.steer_effort), velocity_limit_sim=8.0,
                ),
                "wheels": ImplicitActuatorCfg(
                    joint_names_expr=["wheel_.*"], stiffness=0.0, damping=float(d.wheel_damping),
                    effort_limit_sim=float(d.wheel_effort), velocity_limit_sim=float(d.wheel_velocity_limit),
                ),
            },
        )
        scene_cfg.contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Car/chassis", update_period=0.0, history_length=1)
        self.slots = self._obstacle_slots(track_cfg)
        objects = {}
        kinematic = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        colors = list(PAINT_COLORS)
        for k, (kind, idx) in enumerate(self.slots):
            usd = assets["parked"][idx % len(colors)] if kind == "car" else assets[kind]
            objects[f"{kind}_{idx}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{kind}_{idx}",
                spawn=sim_utils.UsdFileCfg(usd_path=usd, rigid_props=kinematic),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0 * k, 0.0, HIDDEN_Z)),
            )
        if objects:
            scene_cfg.obstacles = RigidObjectCollectionCfg(rigid_objects=objects)
        self.scene = InteractiveScene(scene_cfg)
        # Each layout is anchored at its start (0, 0) and may extend in any direction, so it is shifted to centre
        # its bounding box on the environment's grid point; neighbouring tracks then never overlap.
        origins = centered_origins(self.scene.env_origins.cpu().numpy(), layouts)
        check_no_overlap(layouts, origins, margin=float(cfg.sim.get("env_margin", 30.0)))
        self.origins = torch.tensor(origins, dtype=torch.float32, device=self.device)

        # static per-environment track geometry
        stage = self.sim.stage
        materials = create_materials(stage)
        origins_np = self.origins.cpu().numpy()
        self.decor_info = []
        for i, layout in enumerate(layouts if build_static else []):
            root = f"/World/Tracks/env_{i}"  # NOT under /World/envs/env_i, whose prim carries the grid translation
            self.decor_info.append(build_track(stage, root, layout, origins_np[i, :2], cfg, materials, decor=decor, seed=i))
            # the physical walls must sit exactly where the analytic geometry (LiDAR, collisions, expert) has them
            got = world_wall_bounds(stage, root)
            pts = np.concatenate([w.points for w in layout.walls], axis=0) + origins_np[i, :2]
            want = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            tol = float(cfg.road.wall_thickness) + 0.05
            if got is None or max(abs(g - w) for g, w in zip(got, want)) > tol:
                raise RuntimeError(f"env {i}: wall mesh bounds {got} do not match the layout {want}")
        if visual:
            add_lights(stage, cfg)
        else:
            sim_utils.DomeLightCfg(intensity=1000.0).func("/World/sky", sim_utils.DomeLightCfg(intensity=1000.0))

        self.cameras = None
        if visual:
            from driveact.visualization.isaac_viz import CameraRig

            self.cameras = CameraRig(cfg, self)  # must exist before sim.reset()
        self.sim.reset()
        self.scene.reset()
        self.car = self.scene["car"]
        self.contact = self.scene["contact"]
        self.obstacles = self.scene["obstacles"] if objects else None
        self.steer_ids = self.car.find_joints(STEER_JOINTS, preserve_order=True)[0]
        self.wheel_ids = self.car.find_joints(WHEEL_JOINTS, preserve_order=True)[0]
        self.act = Actuators(num_envs, self.g, self.device)
        self.step_count = 0
        self.render_hooks = []
        if self.cameras is not None:
            self.cameras.initialize()

    # ------------------------------------------------------------------------------------------ obstacle slots
    def _obstacle_slots(self, track_cfg) -> list[tuple[str, int]]:
        groups = track_cfg.obstacles.get("groups") or {}
        o = self.cfg.obstacles
        n_cars = sum(int(hi) for kind, (_, hi) in groups.items() if kind in ("parked_car", "left_parked_car"))
        n_cones = int(groups["cone_taper"][1]) * int(o.cone_taper.count[1]) if "cone_taper" in groups else 0
        n_barriers = int(groups["barrier"][1]) * int(o.barrier.count[1]) if "barrier" in groups else 0
        boxes, circles = obstacle_capacity(track_cfg, o)
        assert n_cars + n_barriers <= boxes and n_cones <= circles
        return [("car", i) for i in range(n_cars)] + [("cone", i) for i in range(n_cones)] + \
               [("barrier", i) for i in range(n_barriers)]

    def set_scene(self, env_ids, layouts, obstacle_sets) -> None:
        """Move the obstacle slots of ``env_ids`` to the sampled obstacle poses (unused slots are hidden)."""
        if self.obstacles is None:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        K = len(self.slots)
        poses = torch.zeros(len(env_ids), K, 7, device=self.device)
        for row, (i, obs) in enumerate(zip(env_ids.tolist(), obstacle_sets)):
            pools = {"car": [o for o in obs.items if o.kind in ("parked_car", "left_parked_car")],
                     "cone": [o for o in obs.items if o.kind == "cone"],
                     "barrier": [o for o in obs.items if o.kind == "barrier"]}
            origin = self.origins[i]
            for k, (kind, idx) in enumerate(self.slots):
                if idx < len(pools[kind]):
                    ob = pools[kind][idx]
                    x, y, z, yaw = ob.x, ob.y, 0.0, ob.yaw
                else:
                    x, y, z, yaw = 4.0 * k, 0.0, HIDDEN_Z, 0.0
                poses[row, k, :3] = torch.tensor([x, y, z], device=self.device) + origin
                poses[row, k, 3:] = yaw_to_quat(torch.tensor([yaw], device=self.device))[0]
            for kind, items in pools.items():
                if len(items) > sum(1 for s in self.slots if s[0] == kind):
                    raise RuntimeError(f"not enough {kind} slots")
        self.obstacles.write_object_pose_to_sim(poses, env_ids=env_ids)
        self.obstacles.write_object_velocity_to_sim(torch.zeros(len(env_ids), K, 6, device=self.device), env_ids=env_ids)

    # --------------------------------------------------------------------------------------------- interface
    def reset(self, env_ids, poses: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = len(env_ids)
        root = torch.zeros(n, 13, device=self.device)
        root[:, :2] = poses[:, :2].to(self.device) + self.origins[env_ids, :2]
        root[:, 2] = self.origins[env_ids, 2] + 0.03
        root[:, 3:7] = yaw_to_quat(poses[:, 2].to(self.device))
        self.car.write_root_pose_to_sim(root[:, :7], env_ids)
        self.car.write_root_velocity_to_sim(root[:, 7:], env_ids)
        J = self.car.num_joints
        zeros = torch.zeros(n, J, device=self.device)
        self.car.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        self.car.set_joint_position_target(torch.zeros(n, 2, device=self.device), joint_ids=self.steer_ids, env_ids=env_ids)
        self.car.set_joint_velocity_target(torch.zeros(n, 4, device=self.device), joint_ids=self.wheel_ids, env_ids=env_ids)
        self.car.reset(env_ids)
        self.contact.reset(env_ids)
        self.act.reset(env_ids)

    def step(self, actions: torch.Tensor) -> None:
        steer_t, accel = actions_to_physical(actions.to(self.device), self.g)
        for _ in range(self.decimation):
            v = self.car.data.root_lin_vel_b[:, 0]
            self.act.step(self.physics_dt, steer_t, accel, v)
            left, right = ackermann_angles(self.act.steer, self.g)
            self.car.set_joint_position_target(torch.stack((left, right), dim=-1), joint_ids=self.steer_ids)
            self.car.set_joint_velocity_target(wheel_angular_speeds(self.act.v_cmd, self.act.steer, self.g),
                                               joint_ids=self.wheel_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            self.step_count += 1
            if self.visual and self.step_count % self.render_every == 0:
                self.sim.render()
                for hook in self.render_hooks:
                    hook(self)

    def state(self) -> dict:
        data = self.car.data
        pos = data.root_pos_w[:, :2] - self.origins[:, :2]
        q = data.root_quat_w
        jp = data.joint_pos
        steer = bicycle_angle(jp[:, self.steer_ids[0]], jp[:, self.steer_ids[1]], self.g)
        forces = self.contact.data.net_forces_w  # (N, bodies, 3)
        contact = forces.norm(dim=-1).max(dim=1).values > 100.0
        # z axis of the chassis in the world frame (roll-over check)
        w, x, y, z = q.unbind(-1)
        up_z = 1 - 2 * (x * x + y * y)
        return {
            "pos": pos.clone(),
            "yaw": quat_to_yaw(q),
            "v_long": data.root_lin_vel_b[:, 0].clone(),
            "v_lat": data.root_lin_vel_b[:, 1].clone(),
            "yaw_rate": data.root_ang_vel_b[:, 2].clone(),
            "steer": steer,
            "contact": contact,
            "up_z": up_z,
            "z": data.root_pos_w[:, 2] - self.origins[:, 2],
            "wheel_speed": data.joint_vel[:, self.wheel_ids].clone(),
            "v_cmd": self.act.v_cmd.clone(),
        }

    def render(self) -> None:
        self.sim.render()

    def close(self) -> None:
        pass


def overview_view(layout, origin: np.ndarray, margin: float, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Eye / target of a straight-down camera that frames the whole layout."""
    x0, y0, x1, y1 = layout.bounds(margin)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, (y1 - y0) * 16 / 9)
    height = half / math.tan(math.radians(fov_deg) / 2)
    eye = np.array([cx + origin[0], cy + origin[1] - 0.01, height])
    return eye, np.array([cx + origin[0], cy + origin[1], 0.0])
