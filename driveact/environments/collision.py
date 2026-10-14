"""Geometric collision checks of the car footprint (oriented rectangle) against the track and obstacles.

Used by both backends at every control step: walls and obstacles end the episode, crossing a curb line is
recorded as a curb strike (road departure). In the Isaac backend the chassis contact sensor is reported as an
additional signal, but the episode outcome comes from these exact checks so both backends label events identically.
"""

from __future__ import annotations

import torch

_EPS = 1e-9


def _to_car(points: torch.Tensor, center: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """(N, K, 2) world -> car frame."""
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    d = points - center[:, None, :]
    return torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), dim=-1)


def footprint_segment_hits(center, yaw, half, segs, mask) -> torch.Tensor:
    """Does each segment intersect the footprint rectangle? (Liang-Barsky clipping in the car frame)

    center (N, 2), yaw (N,), half (2,) = (half length, half width), segs (N, S, 4), mask (N, S) -> (N, S) bool
    """
    if segs.shape[1] == 0:
        return torch.zeros(segs.shape[:2], dtype=torch.bool, device=segs.device)
    p0 = _to_car(segs[..., :2], center, yaw)
    p1 = _to_car(segs[..., 2:], center, yaw)
    d = p1 - p0
    t_enter = torch.zeros(segs.shape[:2], device=segs.device)
    t_exit = torch.ones(segs.shape[:2], device=segs.device)
    ok = mask.clone()
    for k in range(2):
        h = half[k]
        dk, pk = d[..., k], p0[..., k]
        parallel = dk.abs() < _EPS
        ok &= ~(parallel & (pk.abs() > h))
        safe = torch.where(parallel, torch.ones_like(dk), dk)
        ta, tb = (-h - pk) / safe, (h - pk) / safe
        lo = torch.where(parallel, torch.full_like(ta, -1e9), torch.minimum(ta, tb))
        hi = torch.where(parallel, torch.full_like(ta, 1e9), torch.maximum(ta, tb))
        t_enter = torch.maximum(t_enter, lo)
        t_exit = torch.minimum(t_exit, hi)
    return ok & (t_enter <= t_exit)


def footprint_box_hits(center, yaw, half, boxes, mask) -> torch.Tensor:
    """Separating-axis test footprint vs oriented boxes (N, K, >=5) = (cx, cy, hl, hw, yaw, ...) -> (N, K) bool."""
    if boxes.shape[1] == 0:
        return torch.zeros(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
    ua = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)[:, None, :]  # (N, 1, 2)
    va = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)[:, None, :]
    ub = torch.stack((torch.cos(boxes[..., 4]), torch.sin(boxes[..., 4])), dim=-1)  # (N, K, 2)
    vb = torch.stack((-torch.sin(boxes[..., 4]), torch.cos(boxes[..., 4])), dim=-1)
    t = boxes[..., :2] - center[:, None, :]
    ha, wa = half[0], half[1]
    hb, wb = boxes[..., 2], boxes[..., 3]
    separated = torch.zeros(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
    for axis in (ua.expand_as(ub), va.expand_as(ub), ub, vb):
        dist = (t * axis).sum(-1).abs()
        ra = ha * (ua * axis).sum(-1).abs() + wa * (va * axis).sum(-1).abs()
        rb = hb * (ub * axis).sum(-1).abs() + wb * (vb * axis).sum(-1).abs()
        separated |= dist > ra + rb
    return ~separated & mask


def footprint_circle_hits(center, yaw, half, circles, mask) -> torch.Tensor:
    """Footprint vs circles (N, K, >=3) = (cx, cy, r, ...) -> (N, K) bool."""
    if circles.shape[1] == 0:
        return torch.zeros(circles.shape[:2], dtype=torch.bool, device=circles.device)
    p = _to_car(circles[..., :2], center, yaw)
    q = torch.stack((p[..., 0].clamp(-half[0], half[0]), p[..., 1].clamp(-half[1], half[1])), dim=-1)
    return ((p - q).norm(dim=-1) <= circles[..., 2]) & mask
