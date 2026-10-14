"""Planar geometry helpers (NumPy for track construction, PyTorch for batched per-step computations).

Conventions: x/y in metres, angles in radians, yaw measured counter-clockwise from +x, left normal of a heading
``h`` is ``(-sin h, cos h)``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


# ----------------------------------------------------------------------------------------------- scalars / angles
def wrap_angle(a):
    """Wrap to [-pi, pi). Works for floats, NumPy arrays and tensors."""
    if isinstance(a, torch.Tensor):
        return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
    return np.mod(np.asarray(a) + np.pi, 2 * np.pi) - np.pi


def unit(h):
    if isinstance(h, torch.Tensor):
        return torch.stack((torch.cos(h), torch.sin(h)), dim=-1)
    h = np.asarray(h, dtype=np.float64)
    return np.stack((np.cos(h), np.sin(h)), axis=-1)


def left_normal(h):
    if isinstance(h, torch.Tensor):
        return torch.stack((-torch.sin(h), torch.cos(h)), dim=-1)
    h = np.asarray(h, dtype=np.float64)
    return np.stack((-np.sin(h), np.cos(h)), axis=-1)


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def to_frame(points: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """World points (N, ..., 2) -> frame at ``origin`` (N, 2) with heading ``yaw`` (N,)."""
    shape = (points.shape[0],) + (1,) * (points.dim() - 2)
    c, s = torch.cos(yaw).view(shape), torch.sin(yaw).view(shape)
    d = points - origin.view(shape + (2,))
    return torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), dim=-1)


def from_frame(points: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_frame`."""
    shape = (points.shape[0],) + (1,) * (points.dim() - 2)
    c, s = torch.cos(yaw).view(shape), torch.sin(yaw).view(shape)
    x, y = points[..., 0], points[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1) + origin.view(shape + (2,))


# ------------------------------------------------------------------------------------------------------ polylines
def arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative length along a (K, 2) polyline, starting at 0."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Uniformly re-sample a polyline (duplicate points removed). The last point is kept exactly."""
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9))
    points = points[keep]
    s = arc_length(points)
    targets = np.arange(0.0, s[-1], spacing)
    if s[-1] - targets[-1] < 0.25 * spacing and len(targets) > 1:
        targets[-1] = s[-1]  # avoid a near-duplicate last point
    else:
        targets = np.append(targets, s[-1])
    return np.stack((np.interp(targets, s, points[:, 0]), np.interp(targets, s, points[:, 1])), axis=1)


def polyline_headings(points: np.ndarray) -> np.ndarray:
    """Heading at every vertex (central differences, one-sided at the ends)."""
    d = np.zeros_like(points)
    d[1:-1] = points[2:] - points[:-2]
    d[0] = points[1] - points[0]
    d[-1] = points[-1] - points[-2]
    return np.arctan2(d[:, 1], d[:, 0])


def polyline_curvature(points: np.ndarray, headings: np.ndarray | None = None) -> np.ndarray:
    """Signed curvature (1/m, + = turning left) from heading changes over arc length."""
    h = polyline_headings(points) if headings is None else headings
    s = arc_length(points)
    dh = np.zeros_like(h)
    ds = np.ones_like(h)
    dh[1:-1] = wrap_angle(h[2:] - h[:-2])
    ds[1:-1] = s[2:] - s[:-2]
    dh[0], ds[0] = wrap_angle(h[1] - h[0]), s[1] - s[0]
    dh[-1], ds[-1] = wrap_angle(h[-1] - h[-2]), s[-1] - s[-2]
    return dh / np.maximum(ds, 1e-9)


def offset_polyline(points: np.ndarray, headings: np.ndarray, offset: float) -> np.ndarray:
    """Shift every vertex by ``offset`` along its left normal (+ = left)."""
    return points + offset * left_normal(headings)


def polyline_segments(points: np.ndarray) -> np.ndarray:
    """(K, 2) polyline -> (K-1, 4) segments (x0, y0, x1, y1)."""
    return np.concatenate((points[:-1], points[1:]), axis=1)


def point_segment_distance(p: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Distance from points (M, 2) to segments (S, 4) -> (M, S)."""
    a, b = seg[None, :, :2], seg[None, :, 2:]
    ab = b - a
    ap = p[:, None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p[:, None, :] - closest, axis=-1)


# ------------------------------------------------------------------------------------------------ boxes (OBB)
def box_corners(cx, cy, length, width, yaw) -> np.ndarray:
    """Corners (4, 2) of an oriented rectangle, counter-clockwise from front-left."""
    hl, hw = length / 2, width / 2
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return local @ rot2(yaw).T + np.array([cx, cy])


def box_corners_t(center: torch.Tensor, half: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Batched corners: center (..., 2), half (..., 2) = (half length, half width), yaw (...) -> (..., 4, 2)."""
    signs = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=center.device, dtype=center.dtype)
    local = signs * half[..., None, :]
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    x = c * local[..., 0] - s * local[..., 1]
    y = s * local[..., 0] + c * local[..., 1]
    return torch.stack((x, y), dim=-1) + center[..., None, :]


def smoothstep_cos(x: np.ndarray) -> np.ndarray:
    """0 -> 1 cosine ramp on [0, 1] (C1-continuous, zero slope at both ends)."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)
