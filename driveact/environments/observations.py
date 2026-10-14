"""Policy observation vector (all features scaled to roughly [-1, 1]; the model additionally z-normalises them).

    lidar  (num_sectors)  min-pooled planar LiDAR ranges / max_range, sector num_sectors//2 starts straight ahead
    state  (5)            speed / max_speed, steering angle / max_steer, yaw rate (rad/s),
                          previous steering command, previous longitudinal command (normalised actions)
    route  (2 + 1 + 2K + 4 + 1 + 1)
                          sin / cos of the heading error to the route, lateral offset from the right-lane centre / lane
                          width, K look-ahead route points in the vehicle frame / lookahead_scale,
                          one-hot next intersection manoeuvre (left, straight, right, none), distance to it / horizon,
                          remaining route distance to the goal / goal_horizon

The expert's planned offsets, obstacle positions and future actions are never part of the observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

STATE_NAMES = ["speed", "steer", "yaw_rate", "prev_steer_cmd", "prev_accel_cmd"]


@dataclass
class ObsSpec:
    num_sectors: int
    num_lookahead: int

    @property
    def state_dim(self) -> int:
        return len(STATE_NAMES)

    @property
    def route_dim(self) -> int:
        return 3 + 2 * self.num_lookahead + 4 + 1 + 1

    @property
    def dim(self) -> int:
        return self.num_sectors + self.state_dim + self.route_dim

    @property
    def lidar(self) -> slice:
        return slice(0, self.num_sectors)

    @property
    def state(self) -> slice:
        return slice(self.num_sectors, self.num_sectors + self.state_dim)

    @property
    def prev_actions(self) -> slice:
        """Previous steering and longitudinal commands (inside the state block)."""
        return slice(self.state.start + 3, self.state.start + 5)

    @property
    def route(self) -> slice:
        return slice(self.num_sectors + self.state_dim, self.dim)

    def to_dict(self) -> dict:
        return {"num_sectors": self.num_sectors, "num_lookahead": self.num_lookahead, "dim": self.dim,
                "lidar": [self.lidar.start, self.lidar.stop], "state": [self.state.start, self.state.stop],
                "route": [self.route.start, self.route.stop], "state_names": STATE_NAMES}

    @staticmethod
    def from_config(cfg) -> "ObsSpec":
        return ObsSpec(int(cfg.sensors.lidar.num_sectors), len(cfg.sensors.route.lookahead_m))

    @staticmethod
    def from_dict(d: dict) -> "ObsSpec":
        return ObsSpec(int(d["num_sectors"]), int(d["num_lookahead"]))


def build_observation(spec: ObsSpec, sectors, speed, steer, yaw_rate, prev_action, heading_err, lateral, lookahead_local,
                      maneuver, dist_maneuver, dist_goal, cfg, max_speed: float, max_steer: float) -> torch.Tensor:
    rc = cfg.sensors.route
    lane = float(cfg.road.lane_width)
    lidar = sectors / float(cfg.sensors.lidar.max_range)
    state = torch.stack(
        (speed / max_speed, steer / max_steer, yaw_rate.clamp(-2.0, 2.0), prev_action[:, 0], prev_action[:, 1]), dim=-1
    )
    horizon = float(rc.maneuver_horizon)
    far = dist_maneuver > horizon
    m = torch.where(far, torch.full_like(maneuver, 3), maneuver)
    onehot = torch.nn.functional.one_hot(m, 4).float()
    route = torch.cat(
        (
            torch.sin(heading_err)[:, None],
            torch.cos(heading_err)[:, None],
            (lateral / lane)[:, None],
            (lookahead_local / float(rc.lookahead_scale)).flatten(1),
            onehot,
            (dist_maneuver / horizon).clamp(0.0, 1.0)[:, None],
            (dist_goal / float(rc.goal_horizon)).clamp(-0.2, 1.0)[:, None],
        ),
        dim=-1,
    )
    obs = torch.cat((lidar, state, route), dim=-1)
    assert obs.shape[1] == spec.dim
    return obs
