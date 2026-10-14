"""Exact planar ray casting against wall segments, oriented boxes and circles (pure PyTorch, batched over envs).

The LiDAR scan plane (``sensors.yaml: lidar.scan_height``) is crossed by the concrete walls, parked cars, barriers
and cones, so their 2D cross-sections are what a horizontal LiDAR sees. Because obstacles change every episode and
differ between parallel environments, the geometry is passed in as padded tensors instead of a static mesh. The
same tensors drive the geometric collision checks, so sensing and collision use one source of truth.
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-9
INF = float("inf")


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _safe(d: torch.Tensor) -> torch.Tensor:
    return torch.where(d.abs() < _EPS, torch.full_like(d, _EPS), d)


def ray_segments(origin: torch.Tensor, dirs: torch.Tensor, segs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Nearest hit with any active segment (inf if none).

    origin (N, 2), dirs (N, R, 2), segs (N, S, 4) = (x0, y0, x1, y1), mask (N, S) -> (N, R)
    """
    ax = segs[..., 0] - origin[:, 0:1]  # (N, S)
    ay = segs[..., 1] - origin[:, 1:2]
    ex = segs[..., 2] - segs[..., 0]
    ey = segs[..., 3] - segs[..., 1]
    dx, dy = dirs[..., 0:1], dirs[..., 1:2]  # (N, R, 1)
    denom = _cross(dx, dy, ex[:, None, :], ey[:, None, :])  # (N, R, S)
    safe = torch.where(denom.abs() < _EPS, torch.full_like(denom, _EPS), denom)
    t = _cross(ax, ay, ex, ey)[:, None, :] / safe
    u = _cross(ax[:, None, :], ay[:, None, :], dx, dy) / safe
    hit = (denom.abs() >= _EPS) & (t >= 0) & (u >= 0) & (u <= 1) & mask[:, None, :]
    t = torch.where(hit, t, torch.full_like(t, INF))
    return t.min(dim=-1).values if t.shape[-1] else torch.full(dirs.shape[:2], INF, device=dirs.device)


def ray_boxes(origin: torch.Tensor, dirs: torch.Tensor, boxes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Nearest hit with any active rotated box (slab method in the box frame); 0 if the origin is inside.

    boxes (N, K, >=5) = (cx, cy, half_length, half_width, yaw, ...)
    """
    if boxes.shape[1] == 0:
        return torch.full(dirs.shape[:2], INF, device=dirs.device)
    c, s = torch.cos(boxes[..., 4]), torch.sin(boxes[..., 4])
    rel = origin[:, None, :] - boxes[..., :2]
    o_x = c * rel[..., 0] + s * rel[..., 1]
    o_y = -s * rel[..., 0] + c * rel[..., 1]
    d_x = _safe(c[:, None] * dirs[..., 0:1] + s[:, None] * dirs[..., 1:2])
    d_y = _safe(-s[:, None] * dirs[..., 0:1] + c[:, None] * dirs[..., 1:2])
    hx, hy = boxes[:, None, :, 2], boxes[:, None, :, 3]
    tx1, tx2 = (-hx - o_x[:, None]) / d_x, (hx - o_x[:, None]) / d_x
    ty1, ty2 = (-hy - o_y[:, None]) / d_y, (hy - o_y[:, None]) / d_y
    t_near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
    t_far = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
    hit = (t_near <= t_far) & (t_far >= 0) & mask[:, None, :]
    t = torch.where(hit, t_near.clamp(min=0.0), torch.full_like(t_near, INF))
    return t.min(dim=-1).values


def ray_circles(origin: torch.Tensor, dirs: torch.Tensor, circles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Nearest hit with any active circle; 0 if the origin is inside. circles (N, K, >=3) = (cx, cy, r, ...)"""
    if circles.shape[1] == 0:
        return torch.full(dirs.shape[:2], INF, device=dirs.device)
    oc = circles[:, None, :, :2] - origin[:, None, None, :]
    proj = (oc * dirs[:, :, None, :]).sum(-1)
    perp2 = (oc * oc).sum(-1) - proj**2
    disc = circles[:, None, :, 2] ** 2 - perp2
    root = disc.clamp(min=0.0).sqrt()
    t_in, t_out = proj - root, proj + root
    hit = (disc >= 0) & (t_out >= 0) & mask[:, None, :]
    t = torch.where(hit, t_in.clamp(min=0.0), torch.full_like(t_in, INF))
    return t.min(dim=-1).values


def point_segment_distance(p: torch.Tensor, segs: torch.Tensor) -> torch.Tensor:
    """p (N, 2), segs (N, S, 4) -> (N, S)."""
    a, b = segs[..., :2], segs[..., 2:]
    ab = b - a
    ap = p[:, None, :] - a
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)).clamp(0.0, 1.0)
    return (p[:, None, :] - (a + t[..., None] * ab)).norm(dim=-1)


def nearest_segments(origin: torch.Tensor, segs: torch.Tensor, mask: torch.Tensor, k: int, max_range: float):
    """Reduce the segment set to the ``k`` closest segments that lie within ``max_range`` of ``origin``.

    Segments farther than ``max_range`` cannot produce a return inside the sensor range, so dropping them is exact.
    The result is exact as long as at most ``k`` segments are within range; the number that were dropped beyond
    that (should be 0) is returned so callers can detect a too-small ``k``.
    """
    dist = point_segment_distance(origin, segs)
    in_range = mask & (dist <= max_range)
    overflow = (in_range.sum(dim=1) - k).clamp(min=0)
    if segs.shape[1] <= k:
        return segs, in_range, overflow
    dist = torch.where(in_range, dist, torch.full_like(dist, INF))
    idx = dist.topk(k, dim=1, largest=False).indices
    segs_k = torch.gather(segs, 1, idx[..., None].expand(-1, -1, 4))
    return segs_k, torch.gather(in_range, 1, idx), overflow


class PlanarLidar:
    """360-degree planar LiDAR: ``num_rays`` exact rays min-pooled into ``num_sectors`` sectors.

    Ray ``i`` points at body-frame azimuth ``-pi + (i + 0.5) * 2 pi / num_rays`` (counter-clockwise), so sector
    ``num_sectors // 2`` starts straight ahead and the scan begins and ends behind the car.
    """

    def __init__(self, lidar_cfg, device: str | torch.device = "cpu", max_segments: int = 160):
        self.num_rays = int(lidar_cfg.num_rays)
        self.num_sectors = int(lidar_cfg.num_sectors)
        if self.num_rays % self.num_sectors:
            raise ValueError("num_rays must be a multiple of num_sectors")
        self.max_range = float(lidar_cfg.max_range)
        self.mount_x = float(lidar_cfg.mount_x)
        self.max_segments = max_segments
        self.device = torch.device(device)
        i = torch.arange(self.num_rays, dtype=torch.float32, device=self.device)
        self.ray_angles = -math.pi + (i + 0.5) * (2 * math.pi / self.num_rays)
        self.sector_angles = self.ray_angles.view(self.num_sectors, -1).mean(dim=1)

    def origin(self, pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        return pos + self.mount_x * torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)

    def scan(self, pos, yaw, segs, seg_mask, boxes, box_mask, circles, circle_mask) -> torch.Tensor:
        """Ranges (N, num_rays) in metres, clipped to ``max_range``."""
        origin = self.origin(pos, yaw)
        angles = yaw[:, None] + self.ray_angles[None, :]
        dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        segs, seg_mask, overflow = nearest_segments(origin, segs, seg_mask, self.max_segments, self.max_range)
        self.last_overflow = overflow  # (N,) segments in range beyond max_segments; checked by the environment
        dist = ray_segments(origin, dirs, segs, seg_mask)
        dist = torch.minimum(dist, ray_boxes(origin, dirs, boxes, box_mask))
        dist = torch.minimum(dist, ray_circles(origin, dirs, circles, circle_mask))
        return dist.clamp(max=self.max_range)

    def sectors(self, ranges: torch.Tensor) -> torch.Tensor:
        """Min-pool rays into sectors -> (N, num_sectors)."""
        return ranges.view(ranges.shape[0], self.num_sectors, -1).min(dim=-1).values
