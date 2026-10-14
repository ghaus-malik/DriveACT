"""Scripted expert driver (privileged): pure pursuit on the planned offset path + speed-profile tracking.

It reads the route, the planned lateral offsets d(s) and the speed profile v(s) that :class:`DriveEnv` computed at
reset from the true obstacle positions, plus the true vehicle state. The learned policies never see any of this.
"""

from __future__ import annotations

import torch

from driveact.geometry import to_frame
from driveact.vehicle.control import physical_to_actions


class ScriptedExpert:
    name = "expert"

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, env_ids) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: torch.Tensor, env) -> torch.Tensor:
        e, g = self.cfg.expert, env.g
        st = env.state
        pos, yaw, v = st["pos"], st["yaw"], st["v_long"]
        # ---- steering: pure pursuit from the rear axle towards the offset path
        fwd = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        rear = pos + g.rear_axle_x * fwd
        pp = e.pure_pursuit
        ld = (float(pp.lookahead_time) * v).clamp(float(pp.lookahead_min), float(pp.lookahead_max))
        s_target = (env.s + g.rear_axle_x + ld).clamp(max=env.route.length)
        d_target = env.route.interp(env.offsets, s_target)
        target = env.route.point_at(s_target[:, None], d_target[:, None])[:, 0]
        local = to_frame(target[:, None, :], rear, yaw)[:, 0]
        dist = local.norm(dim=-1).clamp(min=1.0)
        alpha = torch.atan2(local[:, 1], local[:, 0])
        steer = torch.atan(2.0 * g.wheelbase * torch.sin(alpha) / dist)
        # ---- speed: track the speed profile slightly ahead of the car
        sp = e.speed
        s_prev = (env.s + v * float(sp.preview_time)).clamp(max=env.route.length)
        v_target = env.route.interp(env.v_ref, s_prev)
        accel = float(sp.gain) * (v_target - v)
        actions = physical_to_actions(steer, accel, g)
        to_goal = env.route.goal_s - env.s
        hold = (to_goal < 0.5) | ((to_goal < float(self.cfg.sim.goal.tolerance)) & (v < 0.6))
        actions[:, 1] = torch.where(hold, torch.full_like(actions[:, 1], float(sp.hold_brake)), actions[:, 1])
        return actions
