"""Backend-agnostic driving environment: episodes, sensing, observations, events and metrics for N parallel cars.

The physics backend (Isaac Lab articulation, or the CPU kinematic test double) only moves the car. Everything else
is computed here from the backend state at every control step (10 Hz), so sensor readings are synchronised with the
control timestep and both backends share one definition of observations, collisions and success.

Per control step the caller does::

    obs = env.observe()             # observation at time t (uses the action executed at t-1)
    info = env.step(actions)        # executes the action for control_dt and evaluates events at t+1
    env.reset(done_ids, next_episode_ids)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from driveact.config import load_track_config
from driveact.environments.collision import footprint_box_hits, footprint_circle_hits, footprint_segment_hits
from driveact.environments.observations import ObsSpec, build_observation
from driveact.environments.route import RouteTensors, heading_error
from driveact.experts.planner import plan_lateral_offsets, plan_speed
from driveact.geometry import to_frame
from driveact.sensors.raycast import PlanarLidar
from driveact.tracks.obstacles import KIND_IDS, ObstacleSet, sample_obstacles
from driveact.utils.seeding import episode_rng
from driveact.vehicle.control import VehicleGeometry

OUTCOMES = [
    "running",
    "success",
    "collision_wall",
    "collision_vehicle",
    "collision_cone",
    "collision_barrier",
    "off_route",
    "overshoot",
    "timeout",
    "stuck",
    "goal_after_curb_strike",
]
OUTCOME_ID = {name: i for i, name in enumerate(OUTCOMES)}
_BOX_KIND_TO_OUTCOME = {
    KIND_IDS["parked_car"]: "collision_vehicle",
    KIND_IDS["left_parked_car"]: "collision_vehicle",
    KIND_IDS["barrier"]: "collision_barrier",
}


def _pad_segments(seg_list: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seg_list)
    segs = torch.zeros(len(seg_list), n, 4, device=device)
    mask = torch.zeros(len(seg_list), n, dtype=torch.bool, device=device)
    for i, s in enumerate(seg_list):
        segs[i, : len(s)] = torch.as_tensor(s, dtype=torch.float32)
        mask[i, : len(s)] = True
    return segs, mask


def obstacle_capacity(track_cfg, ocfg) -> tuple[int, int]:
    """Maximum number of box and circle obstacles a track can place in one episode."""
    boxes = circles = 0
    for kind, (_, hi) in (track_cfg.obstacles.get("groups") or {}).items():
        if kind == "cone_taper":
            circles += int(hi) * int(ocfg.cone_taper.count[1])
        elif kind == "barrier":
            boxes += int(hi) * int(ocfg.barrier.count[1])
        else:
            boxes += int(hi)
    return max(boxes, 1), max(circles, 1)


@dataclass
class EpisodeRecord:
    env: int
    episode: int
    track: int
    layout_seed: int
    outcome: str
    success: bool
    reached_goal: bool
    curb_strike: bool
    steps: int
    time_s: float
    route_length: float
    progress: float  # fraction of start->goal distance covered
    mean_track_error: float
    max_track_error: float
    lane_departure: bool
    lane_departure_fraction: float
    mean_abs_steer_rate: float  # mean |delta steering command| per step (normalised units)
    mean_abs_accel_rate: float
    mean_abs_steer_jerk: float  # mean |second difference| of the steering command
    min_clearance: float
    obstacle_groups: int
    obstacle_groups_passed: int
    mean_speed: float
    noisy: bool = False
    extra: dict = field(default_factory=dict)


class DriveEnv:
    def __init__(self, cfg, track: int, layouts: list, backend, obstacle_stream: int = 0, track_cfg=None):
        if len(layouts) != backend.num_envs:
            raise ValueError("one layout per environment is required")
        self.cfg = cfg
        self.track = int(track)
        self.track_cfg = track_cfg or load_track_config(track)
        self.layouts = layouts
        self.backend = backend
        self.device = backend.device
        self.N = backend.num_envs
        self.stream = int(obstacle_stream)
        self.g = VehicleGeometry.from_config(cfg.vehicle)
        self.dt = float(cfg.sim.control_dt)
        self.spec = ObsSpec.from_config(cfg)
        self.lidar = PlanarLidar(cfg.sensors.lidar, self.device)
        margin = float(cfg.sim.collision.footprint_margin)
        self.half = torch.tensor([self.g.length / 2 + margin, self.g.width / 2 + margin], device=self.device)
        self.route = RouteTensors(layouts, self.device)
        self.walls, self.wall_mask = _pad_segments([l.wall_segments() for l in layouts], self.device)
        self.curbs, self.curb_mask = _pad_segments([l.curb_segments() for l in layouts], self.device)
        kb, kc = obstacle_capacity(self.track_cfg, cfg.obstacles)
        self.boxes = torch.zeros(self.N, kb, 6, device=self.device)
        self.box_mask = torch.zeros(self.N, kb, dtype=torch.bool, device=self.device)
        self.circles = torch.zeros(self.N, kc, 4, device=self.device)
        self.circle_mask = torch.zeros(self.N, kc, dtype=torch.bool, device=self.device)
        self.obstacles: list[ObstacleSet] = [ObstacleSet() for _ in range(self.N)]
        P = self.route.xy.shape[1]
        self.offsets = torch.zeros(self.N, P, device=self.device)  # expert plan (privileged)
        self.v_ref = torch.zeros(self.N, P, device=self.device)
        self.lookahead = torch.tensor(list(cfg.sensors.route.lookahead_m), device=self.device, dtype=torch.float32)
        # body boundary distance along each LiDAR ray (for the clearance metric)
        a = self.lidar.ray_angles
        self.body_ray = torch.minimum(
            (self.g.length / 2) / a.cos().abs().clamp(min=1e-6), (self.g.width / 2) / a.sin().abs().clamp(min=1e-6)
        )
        self._alloc_episode_buffers()
        self.state: dict = {}
        self.ranges = None

    # ------------------------------------------------------------------------------------------------ buffers
    def _alloc_episode_buffers(self):
        N, d = self.N, self.device
        zf = lambda: torch.zeros(N, device=d)  # noqa: E731
        zb = lambda: torch.zeros(N, dtype=torch.bool, device=d)  # noqa: E731
        self.episode_id = torch.full((N,), -1, dtype=torch.long, device=d)
        self.steps = torch.zeros(N, dtype=torch.long, device=d)
        self.max_steps = torch.zeros(N, dtype=torch.long, device=d)
        self.idx = torch.zeros(N, dtype=torch.long, device=d)
        self.s = zf()
        self.lateral = zf()
        self.prev_action = torch.zeros(N, 2, device=d)
        self.prev_prev_action = torch.zeros(N, 2, device=d)
        self.outcome = torch.zeros(N, dtype=torch.long, device=d)
        self.curb_strike = zb()
        self.lane_departure = zb()
        self.noisy = zb()
        self.sum_err, self.max_err, self.dep_steps = zf(), zf(), zf()
        self.sum_dsteer, self.sum_daccel, self.sum_jerk, self.sum_speed = zf(), zf(), zf(), zf()
        self.min_clear = torch.full((N,), math.inf, device=d)
        self.progress_s, self.progress_step = zf(), torch.zeros(N, dtype=torch.long, device=d)
        self.start_pose = torch.zeros(N, 3, device=d)

    # --------------------------------------------------------------------------------------------------- reset
    def reset(self, env_ids, episode_ids, noisy=None) -> None:
        """Start episode ``episode_ids[k]`` in environment ``env_ids[k]`` (obstacles and start pose are drawn
        deterministically from (stream, track, layout seed, episode id))."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        episode_ids = [int(e) for e in episode_ids]
        poses = torch.zeros(len(env_ids), 3, device=self.device)
        sc = self.track_cfg.start
        speed_limit = float(self.track_cfg.speed_limit)
        for k, (i, ep) in enumerate(zip(env_ids.tolist(), episode_ids)):
            layout = self.layouts[i]
            rng = episode_rng(self.stream, self.track, layout.seed, ep)
            obs = sample_obstacles(layout, self.track_cfg.obstacles, self.cfg, rng)
            self.obstacles[i] = obs
            offsets = plan_lateral_offsets(layout, obs, self.cfg)
            vref = plan_speed(layout, offsets, self.cfg, speed_limit)
            P = len(offsets)
            self.offsets[i, :P] = torch.as_tensor(offsets, dtype=torch.float32)
            self.offsets[i, P:] = float(offsets[-1])
            self.v_ref[i, :P] = torch.as_tensor(vref, dtype=torch.float32)
            self.v_ref[i, P:] = 0.0
            boxes, circles = obs.boxes(), obs.circles()
            if len(boxes) > self.boxes.shape[1] or len(circles) > self.circles.shape[1]:
                raise RuntimeError("obstacle capacity exceeded")
            self.boxes[i].zero_()
            self.box_mask[i] = False
            self.circles[i].zero_()
            self.circle_mask[i] = False
            if len(boxes):
                self.boxes[i, : len(boxes)] = torch.as_tensor(boxes, dtype=torch.float32)
                self.box_mask[i, : len(boxes)] = True
            if len(circles):
                self.circles[i, : len(circles)] = torch.as_tensor(circles, dtype=torch.float32)
                self.circle_mask[i, : len(circles)] = True
            lat = float(rng.uniform(-1, 1) * float(sc.lateral_jitter))
            x, y, h = layout.pose_at(layout.start_s, lat)
            h += math.radians(float(rng.uniform(-1, 1) * float(sc.heading_jitter_deg)))
            poses[k] = torch.tensor([x, y, h])
            em = self.cfg.sim.episode
            limit = (layout.goal_s - layout.start_s) / float(em.min_speed_for_timeout) + float(em.time_margin_s)
            self.max_steps[i] = int(math.ceil(limit / self.dt))
        self.backend.set_scene(env_ids, [self.layouts[i] for i in env_ids.tolist()], [self.obstacles[i] for i in env_ids.tolist()])
        self.backend.reset(env_ids, poses)
        self.start_pose[env_ids] = poses
        self.episode_id[env_ids] = torch.as_tensor(episode_ids, dtype=torch.long, device=self.device)
        for buf in (self.steps, self.outcome, self.progress_step):
            buf[env_ids] = 0
        for buf in (self.sum_err, self.max_err, self.dep_steps, self.sum_dsteer, self.sum_daccel, self.sum_jerk, self.sum_speed):
            buf[env_ids] = 0.0
        self.min_clear[env_ids] = math.inf
        self.curb_strike[env_ids] = False
        self.lane_departure[env_ids] = False
        self.noisy[env_ids] = False if noisy is None else torch.as_tensor(noisy, dtype=torch.bool, device=self.device)
        self.prev_action[env_ids] = 0.0
        self.prev_prev_action[env_ids] = 0.0
        # initialise the route projection around the start
        start_idx = torch.searchsorted(self.route.s[env_ids], self.route.start_s[env_ids, None]).squeeze(1)
        self.idx[env_ids] = start_idx.clamp(max=self.route.xy.shape[1] - 1)
        self._update_state()
        self.progress_s[env_ids] = self.s[env_ids]

    # ------------------------------------------------------------------------------------------------- sensing
    def _update_state(self):
        st = self.backend.state()
        self.state = st
        idx, s, lat, h_route = self.route.project(st["pos"], self.idx)
        self.idx, self.s, self.lateral, self.route_heading = idx, s, lat, h_route
        self.ranges = self.lidar.scan(st["pos"], st["yaw"], self.walls, self.wall_mask, self.boxes, self.box_mask,
                                      self.circles, self.circle_mask)

    def observe(self) -> torch.Tensor:
        st = self.state
        sectors = self.lidar.sectors(self.ranges)
        s_ahead = (self.s[:, None] + self.lookahead[None, :]).clamp(max=self.route.length[:, None])
        pts = self.route.point_at(s_ahead)
        local = to_frame(pts, st["pos"], st["yaw"])
        maneuver, dist_m = self.route.maneuver_at(self.idx)
        return build_observation(
            self.spec, sectors, st["v_long"], st["steer"], st["yaw_rate"], self.prev_action,
            heading_error(st["yaw"], self.route_heading), self.lateral, local, maneuver, dist_m,
            self.route.goal_s - self.s, self.cfg, self.g.max_speed, self.g.max_steer,
        )

    def reference_offset(self) -> torch.Tensor:
        """Expert's planned lateral offset at the current route position (privileged; metrics and plots only)."""
        return self.route.interp(self.offsets, self.s)

    # ---------------------------------------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor) -> dict:
        actions = actions.clamp(-1.0, 1.0)
        active = self.outcome == 0
        self.backend.step(actions)
        self._update_state()
        st = self.state
        self.steps += active.long()

        # ---- events
        pos, yaw = st["pos"], st["yaw"]
        wall_hit = footprint_segment_hits(pos, yaw, self.half, self.walls, self.wall_mask).any(dim=1)
        box_hits = footprint_box_hits(pos, yaw, self.half, self.boxes, self.box_mask)
        circle_hit = footprint_circle_hits(pos, yaw, self.half, self.circles, self.circle_mask).any(dim=1)
        curb_hit = footprint_segment_hits(pos, yaw, self.half, self.curbs, self.curb_mask).any(dim=1)
        self.curb_strike |= curb_hit & active

        outcome = torch.zeros_like(self.outcome)
        # obstacles first (more specific), then walls
        box_kind = torch.where(box_hits, self.boxes[..., 5].long(), torch.full_like(box_hits, -1, dtype=torch.long))
        for kind_id, name in _BOX_KIND_TO_OUTCOME.items():
            hit = (box_kind == kind_id).any(dim=1)
            outcome = torch.where((outcome == 0) & hit, torch.full_like(outcome, OUTCOME_ID[name]), outcome)
        outcome = torch.where((outcome == 0) & circle_hit, torch.full_like(outcome, OUTCOME_ID["collision_cone"]), outcome)
        outcome = torch.where((outcome == 0) & wall_hit, torch.full_like(outcome, OUTCOME_ID["collision_wall"]), outcome)

        gcfg = self.cfg.sim.goal
        speed = st["v_long"]
        to_goal = self.route.goal_s - self.s
        at_goal = (to_goal.abs() <= float(gcfg.tolerance)) & (speed.abs() < float(gcfg.stop_speed))
        goal_code = torch.where(self.curb_strike, torch.full_like(outcome, OUTCOME_ID["goal_after_curb_strike"]),
                                torch.full_like(outcome, OUTCOME_ID["success"]))
        outcome = torch.where((outcome == 0) & at_goal, goal_code, outcome)
        off_route = self.lateral.abs() > 3.0 * float(self.cfg.road.lane_width)
        outcome = torch.where((outcome == 0) & off_route, torch.full_like(outcome, OUTCOME_ID["off_route"]), outcome)
        overshoot = to_goal < -float(gcfg.overshoot)
        outcome = torch.where((outcome == 0) & overshoot, torch.full_like(outcome, OUTCOME_ID["overshoot"]), outcome)
        moved = (self.s - self.progress_s) > 1.0
        self.progress_s = torch.where(moved, self.s, self.progress_s)
        self.progress_step = torch.where(moved, self.steps, self.progress_step)
        stuck_steps = int(round(float(self.cfg.sim.episode.stuck_time_s) / self.dt))
        stuck = ((self.steps - self.progress_step) > stuck_steps) & (to_goal > float(gcfg.tolerance))
        outcome = torch.where((outcome == 0) & stuck, torch.full_like(outcome, OUTCOME_ID["stuck"]), outcome)
        timeout = self.steps >= self.max_steps
        outcome = torch.where((outcome == 0) & timeout, torch.full_like(outcome, OUTCOME_ID["timeout"]), outcome)
        newly_done = active & (outcome > 0)
        self.outcome = torch.where(newly_done, outcome, self.outcome)

        # ---- metrics (only for steps of active episodes)
        a = active.float()
        err = (self.lateral - self.reference_offset()).abs()
        self.sum_err += a * err
        self.max_err = torch.where(active, torch.maximum(self.max_err, err), self.max_err)
        dep = err > float(self.cfg.eval.lane_departure_margin)
        self.lane_departure |= dep & active
        self.dep_steps += a * dep.float()
        first = self.steps <= 1
        d1 = actions - self.prev_action
        d2 = actions - 2 * self.prev_action + self.prev_prev_action
        self.sum_dsteer += a * torch.where(first, torch.zeros_like(d1[:, 0]), d1[:, 0].abs())
        self.sum_daccel += a * torch.where(first, torch.zeros_like(d1[:, 1]), d1[:, 1].abs())
        self.sum_jerk += a * torch.where(self.steps <= 2, torch.zeros_like(d2[:, 0]), d2[:, 0].abs())
        self.sum_speed += a * speed
        clear = (self.ranges - self.body_ray[None, :]).min(dim=1).values
        self.min_clear = torch.where(active, torch.minimum(self.min_clear, clear), self.min_clear)
        self.prev_prev_action = torch.where(active[:, None], self.prev_action, self.prev_prev_action)
        self.prev_action = torch.where(active[:, None], actions, self.prev_action)
        return {"done": self.outcome > 0, "newly_done": newly_done, "outcome": self.outcome.clone(),
                "contact": st["contact"], "lidar_overflow": self.lidar.last_overflow}

    # ------------------------------------------------------------------------------------------------- records
    def episode_record(self, i: int) -> EpisodeRecord:
        layout = self.layouts[i]
        steps = max(1, int(self.steps[i]))
        outcome = OUTCOMES[int(self.outcome[i])]
        obs = self.obstacles[i]
        margin = float(self.cfg.eval.obstacle_pass_margin)
        s_final = float(self.s[i])
        passed = sum(1 for g in obs.groups if s_final > g.s_max + margin)  # episodes end at the first collision
        span = max(1e-6, layout.goal_s - layout.start_s)
        return EpisodeRecord(
            env=i,
            episode=int(self.episode_id[i]),
            track=self.track,
            layout_seed=int(layout.seed),
            outcome=outcome,
            success=outcome == "success",
            reached_goal=outcome in ("success", "goal_after_curb_strike"),
            curb_strike=bool(self.curb_strike[i]),
            steps=int(self.steps[i]),
            time_s=float(self.steps[i]) * self.dt,
            route_length=span,
            progress=float(np.clip((s_final - layout.start_s) / span, 0.0, 1.0)),
            mean_track_error=float(self.sum_err[i]) / steps,
            max_track_error=float(self.max_err[i]),
            lane_departure=bool(self.lane_departure[i]),
            lane_departure_fraction=float(self.dep_steps[i]) / steps,
            mean_abs_steer_rate=float(self.sum_dsteer[i]) / steps,
            mean_abs_accel_rate=float(self.sum_daccel[i]) / steps,
            mean_abs_steer_jerk=float(self.sum_jerk[i]) / steps,
            min_clearance=float(self.min_clear[i]),
            obstacle_groups=len(obs.groups),
            obstacle_groups_passed=passed,
            mean_speed=float(self.sum_speed[i]) / steps,
            noisy=bool(self.noisy[i]),
        )
