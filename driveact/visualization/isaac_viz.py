"""Visual demonstration mode (Isaac Sim, RTX rendering): three cameras, in-world markers and video recording.

Cameras (environment 0):
  top    follows the car from above (``top.follow_height``), or frames the whole track (``overview``)
  front  rigidly mounted behind the windscreen, looking along the vehicle x axis
  chase  behind and above the car, position smoothed between frames

In-world markers (visible in all cameras):
  green  ACT's predicted chunk rolled out through the kinematic bicycle model from the current state (this is
         NOT a position output of ACT; ACT predicts controls, the path is what those controls would produce)
  blue   the expert's reference path (debug option)
  orange the car's past trajectory
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera, CameraCfg

from driveact.environments.isaac_backend import overview_view


def _pitch_quat(pitch_deg: float) -> tuple[float, float, float, float]:
    a = math.radians(pitch_deg) / 2  # rotation about +y: positive = nose down in the "world" camera convention
    return (math.cos(a), 0.0, math.sin(a), 0.0)


class CameraRig:
    VIEWS = ("top", "front", "chase")

    def __init__(self, cfg, backend):
        v = cfg.visual.cameras
        self.cfg = cfg
        self.backend = backend
        W, H = int(v.width), int(v.height)
        common = dict(update_period=0.0, height=H, width=W, data_types=["rgb"])
        self.front = Camera(CameraCfg(
            prim_path="/World/envs/env_0/Car/chassis/front_cam",
            spawn=sim_utils.PinholeCameraCfg(focal_length=float(v.front.focal_length),
                                             horizontal_aperture=float(v.front.horizontal_aperture), clipping_range=(0.1, 600.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(v.front.pos), rot=_pitch_quat(float(v.front.pitch_deg)), convention="world"),
            **common))
        self.chase = Camera(CameraCfg(prim_path="/World/chase_cam", spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(v.chase.focal_length), clipping_range=(0.1, 800.0)), **common))
        self.top = Camera(CameraCfg(prim_path="/World/top_cam", spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(v.top.focal_length), clipping_range=(1.0, 3000.0)), **common))
        self._chase_eye = None
        self.top_mode = "follow"
        markers = VisualizationMarkersCfg(prim_path="/Visuals/driveact", markers={
            "pred": sim_utils.SphereCfg(radius=0.22, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.95, 0.3), emissive_color=(0.05, 0.6, 0.15))),
            "ref": sim_utils.SphereCfg(radius=0.15, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.4, 1.0), emissive_color=(0.05, 0.15, 0.6))),
            "trail": sim_utils.SphereCfg(radius=0.10, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.30, 0.02), emissive_color=(0.35, 0.08, 0.0))),
        })
        self.markers = VisualizationMarkers(markers)

    def initialize(self) -> None:
        self.update_views(force=True)

    # ------------------------------------------------------------------------------------------------ poses
    def update_views(self, force: bool = False) -> None:
        b = self.backend
        v = self.cfg.visual.cameras
        st = b.car.data
        p = st.root_pos_w[0].detach().cpu().numpy()
        q = st.root_quat_w[0].detach().cpu().numpy()
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        eye = p - float(v.chase.distance) * f + np.array([0.0, 0.0, float(v.chase.height)])
        if self._chase_eye is None or force:
            self._chase_eye = eye
        else:
            a = float(v.chase.smoothing)
            self._chase_eye = (1 - a) * self._chase_eye + a * eye
        target = p + float(v.chase.look_ahead) * f + np.array([0.0, 0.0, 1.0])
        dev = b.device
        self.chase.set_world_poses_from_view(torch.tensor(self._chase_eye, device=dev, dtype=torch.float32)[None],
                                             torch.tensor(target, device=dev, dtype=torch.float32)[None])
        if self.top_mode == "overview":
            eye_t, tgt_t = overview_view(b.layouts[0], b.origins[0].cpu().numpy(), float(v.top.overview_margin),
                                         self._fov(float(v.top.focal_length)))
        else:
            c = p + float(v.top.follow_ahead) * f
            eye_t = np.array([c[0] - 0.01 * f[0], c[1] - 0.01 * f[1], float(v.top.follow_height)])
            tgt_t = np.array([c[0], c[1], 0.0])
        self.top.set_world_poses_from_view(torch.tensor(eye_t, device=dev, dtype=torch.float32)[None],
                                           torch.tensor(tgt_t, device=dev, dtype=torch.float32)[None])

    @staticmethod
    def _fov(focal_length: float, aperture: float = 20.955) -> float:
        return math.degrees(2 * math.atan(aperture / (2 * focal_length)))

    # ---------------------------------------------------------------------------------------------- capture
    def capture(self) -> dict[str, np.ndarray]:
        dt = self.backend.physics_dt
        out = {}
        for name in self.VIEWS:
            cam = getattr(self, name)
            cam.update(dt)
            out[name] = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        return out

    def set_markers(self, pred_xy=None, ref_xy=None, trail_xy=None) -> None:
        """World-frame (local to env 0) points; z is fixed slightly above the road."""
        origin = self.backend.origins[0].cpu().numpy()
        pts, idx = [], []
        for k, (xy, z) in enumerate(((pred_xy, 0.35), (ref_xy, 0.2), (trail_xy, 0.15))):
            if xy is None or len(xy) == 0:
                continue
            xy = np.asarray(xy, dtype=np.float32)
            pts.append(np.column_stack((xy[:, 0] + origin[0], xy[:, 1] + origin[1], np.full(len(xy), z + origin[2]))))
            idx.append(np.full(len(xy), k))
        if not pts:
            self.markers.set_visibility(False)
            return
        self.markers.set_visibility(True)
        self.markers.visualize(translations=torch.tensor(np.concatenate(pts), dtype=torch.float32),
                               marker_indices=torch.tensor(np.concatenate(idx), dtype=torch.long))
