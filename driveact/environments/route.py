"""Batched route representation: padded per-environment route polylines with projection and look-up by arc length."""

from __future__ import annotations

import torch

from driveact.geometry import wrap_angle


def _pad(arrays, value_last: bool, dtype, device) -> torch.Tensor:
    """Stack 1-D/2-D arrays of different lengths, padding by repeating the last row."""
    n = max(len(a) for a in arrays)
    out = []
    for a in arrays:
        t = torch.as_tensor(a, dtype=dtype)
        if len(t) < n:
            pad = t[-1:].expand((n - len(t),) + tuple(t.shape[1:])) if value_last else torch.zeros((n - len(t),) + tuple(t.shape[1:]), dtype=dtype)
            t = torch.cat((t, pad), dim=0)
        out.append(t)
    return torch.stack(out).to(device)


class RouteTensors:
    """Routes of N environments (possibly different layouts), padded to the longest one."""

    def __init__(self, layouts, device):
        self.device = torch.device(device)
        f = torch.float32
        self.xy = _pad([l.route for l in layouts], True, f, device)  # (N, P, 2)
        self.heading = _pad([l.route_heading for l in layouts], True, f, device)
        # arc length keeps increasing in the padding so searchsorted stays well defined
        s_list = []
        for l in layouts:
            s = torch.as_tensor(l.route_s, dtype=torch.float64)
            s_list.append(s)
        n = self.xy.shape[1]
        s_pad = []
        for s in s_list:
            if len(s) < n:
                s = torch.cat((s, s[-1] + 1e-3 * torch.arange(1, n - len(s) + 1, dtype=torch.float64)))
            s_pad.append(s)
        self.s = torch.stack(s_pad).to(device=device, dtype=f)
        self.curvature = _pad([l.route_curvature for l in layouts], True, f, device)
        self.next_maneuver = _pad([l.next_maneuver for l in layouts], True, torch.long, device)
        self.dist_to_maneuver = _pad([l.dist_to_maneuver.clip(max=1e4) for l in layouts], True, f, device)
        self.num_points = torch.tensor([len(l.route) for l in layouts], device=device)
        self.length = torch.tensor([l.length for l in layouts], device=device, dtype=f)
        self.start_s = torch.tensor([l.start_s for l in layouts], device=device, dtype=f)
        self.goal_s = torch.tensor([l.goal_s for l in layouts], device=device, dtype=f)
        self.spacing = float(layouts[0].spacing)
        self.n = len(layouts)

    def per_env_profile(self, arrays) -> torch.Tensor:
        """Pad per-route-point profiles (e.g. planned offsets / speeds) to (N, P)."""
        return _pad(arrays, True, torch.float32, self.device)

    # ------------------------------------------------------------------------------------------- projection
    def project(self, xy: torch.Tensor, idx_prev: torch.Tensor, back: int = 20, ahead: int = 100):
        """Project points (N, 2) onto the route, searching indices [idx_prev - back, idx_prev + ahead].

        Returns idx (N,) nearest vertex, s (N,), lateral offset (N,) (+ = left), route heading at s (N,).
        """
        N, P = self.xy.shape[:2]
        ar = torch.arange(-back, ahead + 1, device=self.device)
        cand = (idx_prev[:, None] + ar[None, :]).clamp(0, P - 1)
        cand = torch.minimum(cand, (self.num_points - 1)[:, None])
        pts = torch.gather(self.xy, 1, cand[..., None].expand(-1, -1, 2))
        d2 = ((pts - xy[:, None, :]) ** 2).sum(-1)
        k = d2.argmin(dim=1)
        idx = torch.gather(cand, 1, k[:, None]).squeeze(1)
        # refine on the segment towards the neighbour on the side of the point
        rows = torch.arange(N, device=self.device)
        h = self.heading[rows, idx]
        p = self.xy[rows, idx]
        rel = xy - p
        along = rel[:, 0] * torch.cos(h) + rel[:, 1] * torch.sin(h)
        lateral = -rel[:, 0] * torch.sin(h) + rel[:, 1] * torch.cos(h)
        s = self.s[rows, idx] + along
        s = torch.minimum(s.clamp(min=0.0), self.length)
        return idx, s, lateral, self.heading_at(s)

    # ------------------------------------------------------------------------------------------------ look-up
    def _interp_index(self, s_query: torch.Tensor):
        """s_query (N, K) -> lower index (N, K) and fraction (N, K)."""
        P = self.s.shape[1]
        i = torch.searchsorted(self.s, s_query.contiguous(), right=True) - 1
        i = i.clamp(0, P - 2)
        s0 = torch.gather(self.s, 1, i)
        s1 = torch.gather(self.s, 1, i + 1)
        w = ((s_query - s0) / (s1 - s0).clamp(min=1e-6)).clamp(0.0, 1.0)
        return i, w

    def interp(self, values: torch.Tensor, s_query: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of per-point values (N, P) or (N, P, D) at arc lengths (N, K)."""
        squeeze = s_query.dim() == 1
        sq = s_query[:, None] if squeeze else s_query
        i, w = self._interp_index(sq)
        if values.dim() == 2:
            v0, v1 = torch.gather(values, 1, i), torch.gather(values, 1, i + 1)
            out = v0 + w * (v1 - v0)
        else:
            D = values.shape[-1]
            v0 = torch.gather(values, 1, i[..., None].expand(-1, -1, D))
            v1 = torch.gather(values, 1, (i + 1)[..., None].expand(-1, -1, D))
            out = v0 + w[..., None] * (v1 - v0)
        return out[:, 0] if squeeze else out

    def heading_at(self, s_query: torch.Tensor) -> torch.Tensor:
        c = self.interp(torch.cos(self.heading), s_query)
        si = self.interp(torch.sin(self.heading), s_query)
        return torch.atan2(si, c)

    def point_at(self, s_query: torch.Tensor, lateral: torch.Tensor | None = None) -> torch.Tensor:
        """Route point at arc length (N, K), optionally shifted by ``lateral`` (N, K) to the left -> (N, K, 2)."""
        xy = self.interp(self.xy, s_query)
        if lateral is not None:
            h = self.heading_at(s_query)
            xy = xy + lateral[..., None] * torch.stack((-torch.sin(h), torch.cos(h)), dim=-1)
        return xy

    def maneuver_at(self, idx: torch.Tensor):
        rows = torch.arange(self.n, device=self.device)
        return self.next_maneuver[rows, idx], self.dist_to_maneuver[rows, idx]


def heading_error(yaw: torch.Tensor, route_heading: torch.Tensor) -> torch.Tensor:
    return wrap_angle(yaw - route_heading)
