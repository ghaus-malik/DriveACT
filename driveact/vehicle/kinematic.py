"""CPU-capable kinematic bicycle "test double" with the same control interface as the Isaac articulation.

Purpose: unit tests and end-to-end pipeline dry runs (collection -> training -> evaluation) on machines without
Isaac Sim. It applies the same actuator model (steering slew-rate limit, acceleration -> speed command with no
reverse) and adds a first-order lag of the speed behind the speed command. It has no tyre slip, no load transfer
and no physical contact; results obtained with it are NOT simulation results and are never reported as such.
"""

from __future__ import annotations

import torch

from driveact.geometry import wrap_angle
from driveact.vehicle.control import Actuators, VehicleGeometry, actions_to_physical


class KinematicBackend:
    name = "kinematic"

    def __init__(self, cfg, num_envs: int, device="cpu"):
        self.cfg = cfg
        self.g = VehicleGeometry.from_config(cfg.vehicle)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.control_dt = float(cfg.sim.control_dt)
        kd = cfg.vehicle.kinematic_double
        self.substeps = int(kd.substeps)
        self.tau = float(kd.speed_time_constant)
        self.act = Actuators(num_envs, self.g, self.device)
        z = lambda: torch.zeros(num_envs, device=self.device)  # noqa: E731
        self.rear = torch.zeros(num_envs, 2, device=self.device)
        self.yaw, self.v, self.steer, self.yaw_rate = z(), z(), z(), z()

    # ------------------------------------------------------------------------------------------------ interface
    def set_scene(self, env_ids, layouts, obstacle_sets) -> None:
        """The test double has no scene; obstacles only exist as geometry in the environment."""

    def reset(self, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
        """poses (n, 3): footprint-centre x, y, yaw in each environment's local frame. Cars start at rest."""
        yaw = poses[:, 2]
        self.yaw[env_ids] = yaw
        self.rear[env_ids] = poses[:, :2] + self.g.rear_axle_x * torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
        self.v[env_ids] = 0.0
        self.steer[env_ids] = 0.0
        self.yaw_rate[env_ids] = 0.0
        self.act.reset(env_ids)

    def step(self, actions: torch.Tensor) -> None:
        steer_t, accel = actions_to_physical(actions, self.g)
        dt = self.control_dt / self.substeps
        for _ in range(self.substeps):
            self.act.step(dt, steer_t, accel, self.v)
            self.steer = self.act.steer.clone()
            self.v = self.v + (self.act.v_cmd - self.v) * min(1.0, dt / self.tau)
            self.v = self.v.clamp(min=0.0)
            self.yaw_rate = self.v * torch.tan(self.steer) / self.g.wheelbase
            self.rear = self.rear + dt * self.v[:, None] * torch.stack((torch.cos(self.yaw), torch.sin(self.yaw)), -1)
            self.yaw = wrap_angle(self.yaw + dt * self.yaw_rate)

    def state(self) -> dict:
        fwd = torch.stack((torch.cos(self.yaw), torch.sin(self.yaw)), -1)
        pos = self.rear - self.g.rear_axle_x * fwd
        return {
            "pos": pos,
            "yaw": self.yaw.clone(),
            "v_long": self.v.clone(),
            "v_lat": -self.g.rear_axle_x * self.yaw_rate,  # lateral velocity of the footprint centre
            "yaw_rate": self.yaw_rate.clone(),
            "steer": self.steer.clone(),
            "contact": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
