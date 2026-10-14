"""Vehicle control interface shared by all backends.

Policy / expert action (normalised, both in [-1, 1]):
    action[0] = steering     -> bicycle steering angle  delta = action[0] * max_angle   (+ = left)
    action[1] = longitudinal -> acceleration a = action[1] * max_accel  if >= 0
                                             a = action[1] * max_brake  if <  0   (deceleration)

Actuation (identical for the Isaac articulation and the CPU test double):
  * the steering angle follows the command with a slew-rate limit (``max_rate``);
  * the acceleration command is integrated into a speed command v_cmd in [0, max_speed] (no reverse driving),
    with anti-windup that keeps v_cmd within ``command_slack`` of the measured speed;
  * the bicycle angle is split into left/right knuckle angles with Ackermann geometry and v_cmd into per-wheel
    angular velocities consistent with the turning radius (no tyre scrub), which drive the wheel joints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class VehicleGeometry:
    length: float
    width: float
    wheelbase: float
    track_width: float
    wheel_radius: float
    front_axle_x: float  # in the vehicle frame (origin = footprint centre)
    rear_axle_x: float
    max_steer: float  # rad
    max_steer_rate: float  # rad/s
    knuckle_limit: float  # rad
    max_accel: float
    max_brake: float
    max_speed: float
    command_slack: float

    @staticmethod
    def from_config(vcfg) -> "VehicleGeometry":
        b = vcfg.body
        front_axle_x = float(b.length) / 2 - float(b.front_overhang)
        return VehicleGeometry(
            length=float(b.length),
            width=float(b.width),
            wheelbase=float(b.wheelbase),
            track_width=float(b.track_width),
            wheel_radius=float(b.wheel_radius),
            front_axle_x=front_axle_x,
            rear_axle_x=front_axle_x - float(b.wheelbase),
            max_steer=math.radians(float(vcfg.steering.max_angle_deg)),
            max_steer_rate=math.radians(float(vcfg.steering.max_rate_deg_s)),
            knuckle_limit=math.radians(float(vcfg.steering.knuckle_limit_deg)),
            max_accel=float(vcfg.longitudinal.max_accel),
            max_brake=float(vcfg.longitudinal.max_brake),
            max_speed=float(vcfg.longitudinal.max_speed),
            command_slack=float(vcfg.longitudinal.command_slack),
        )

    @property
    def min_turn_radius(self) -> float:
        """Rear-axle turning radius at full steering."""
        return self.wheelbase / math.tan(self.max_steer)


def actions_to_physical(actions: torch.Tensor, g: VehicleGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 2) normalised actions -> steering angle target (rad) and acceleration (m/s^2)."""
    a = actions.clamp(-1.0, 1.0)
    steer = a[:, 0] * g.max_steer
    accel = torch.where(a[:, 1] >= 0, a[:, 1] * g.max_accel, a[:, 1] * g.max_brake)
    return steer, accel


def physical_to_actions(steer: torch.Tensor, accel: torch.Tensor, g: VehicleGeometry) -> torch.Tensor:
    """Inverse of :func:`actions_to_physical` (with clipping)."""
    s = (steer / g.max_steer).clamp(-1.0, 1.0)
    a = torch.where(accel >= 0, accel / g.max_accel, accel / g.max_brake).clamp(-1.0, 1.0)
    return torch.stack((s, a), dim=-1)


class Actuators:
    """Steering slew-rate limiter and speed-command integrator for N vehicles."""

    def __init__(self, num_envs: int, g: VehicleGeometry, device):
        self.g = g
        self.steer = torch.zeros(num_envs, device=device)  # commanded bicycle angle after the rate limit
        self.v_cmd = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor, speed: torch.Tensor | None = None) -> None:
        self.steer[env_ids] = 0.0
        self.v_cmd[env_ids] = 0.0 if speed is None else speed

    def step(self, dt: float, steer_target: torch.Tensor, accel: torch.Tensor, v_measured: torch.Tensor) -> None:
        g = self.g
        max_delta = g.max_steer_rate * dt
        self.steer = self.steer + (steer_target.clamp(-g.max_steer, g.max_steer) - self.steer).clamp(-max_delta, max_delta)
        v = self.v_cmd + accel * dt
        v = torch.minimum(torch.maximum(v, v_measured - g.command_slack), v_measured + g.command_slack)
        self.v_cmd = v.clamp(0.0, g.max_speed)

    def knuckle_angles(self) -> tuple[torch.Tensor, torch.Tensor]:
        return ackermann_angles(self.steer, self.g)

    def wheel_speeds(self) -> torch.Tensor:
        return wheel_angular_speeds(self.v_cmd, self.steer, self.g)


def ackermann_angles(steer: torch.Tensor, g: VehicleGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Left and right front wheel angles for a bicycle angle (same turning centre on the rear axle line)."""
    t = torch.tan(steer)
    half = g.track_width / 2
    left = torch.atan2(g.wheelbase * t, g.wheelbase - half * t)
    right = torch.atan2(g.wheelbase * t, g.wheelbase + half * t)
    lim = g.knuckle_limit
    return left.clamp(-lim, lim), right.clamp(-lim, lim)


def bicycle_angle(left: torch.Tensor, right: torch.Tensor, g: VehicleGeometry) -> torch.Tensor:
    """Equivalent bicycle angle from measured knuckle angles (inverse of :func:`ackermann_angles`, averaged)."""
    half = g.track_width / 2
    tl, tr = torch.tan(left), torch.tan(right)
    from_left = torch.atan(tl * g.wheelbase / (g.wheelbase + half * tl))
    from_right = torch.atan(tr * g.wheelbase / (g.wheelbase - half * tr))
    return 0.5 * (from_left + from_right)


def wheel_angular_speeds(v: torch.Tensor, steer: torch.Tensor, g: VehicleGeometry) -> torch.Tensor:
    """(N, 4) wheel spin rates [front-left, front-right, rear-left, rear-right] in rad/s for rear-axle speed ``v``."""
    omega = v * torch.tan(steer) / g.wheelbase  # yaw rate
    half = g.track_width / 2
    rear_l = v - omega * half
    rear_r = v + omega * half
    front_l = torch.sqrt(rear_l**2 + (omega * g.wheelbase) ** 2)
    front_r = torch.sqrt(rear_r**2 + (omega * g.wheelbase) ** 2)
    return torch.stack((front_l, front_r, rear_l, rear_r), dim=-1) / g.wheel_radius
