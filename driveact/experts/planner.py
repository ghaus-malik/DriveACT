"""Privileged planning for the scripted expert: lateral offset profile around obstacles and a speed profile.

Lateral profile d(s) (metres left of the right-lane centre, on the route grid):
  * every obstacle group that intrudes into the corridor the car occupies at d = 0 (half width + clearance)
    requires a pass on the left: d >= D = max(pass_offset, d_max + half_width + clearance) on
    [s_min - buffer_before, s_max + buffer_after];
  * passes closer than 2 * ramp + merge_gap are merged (the car stays in the left lane in between);
  * each plateau is entered and left with a cosine ramp of length ``lane_change_length`` (C1-smooth);
  * the result is checked against every obstacle (including ones in the left lane that the car must not swerve
    into) and against both curbs. If any check fails the configuration is infeasible.

Speed profile v(s): min(speed limit, sqrt(a_lat / |kappa_eff|)), a pass-speed cap while shifted, a stop at the goal,
then backward (deceleration) and forward (acceleration) passes.
"""

from __future__ import annotations

import numpy as np

from driveact.geometry import smoothstep_cos


class InfeasibleError(RuntimeError):
    pass


def vehicle_half_extents(vcfg) -> tuple[float, float]:
    return float(vcfg.body.length) / 2, float(vcfg.body.width) / 2


def plan_lateral_offsets(layout, obstacles, cfg) -> np.ndarray:
    e = cfg.expert.lateral
    lane = layout.lane_width
    half_len, half_w = vehicle_half_extents(cfg.vehicle)
    s = layout.route_s
    clear = float(e.clearance)
    ramp = float(e.lane_change_length)
    d_lo_road = -lane / 2 + half_w + float(e.curb_clearance)
    d_hi_road = 1.5 * lane - half_w - float(e.curb_clearance)

    plateaus = []
    for g in obstacles.groups:
        if g.d_max > -(half_w + clear) and g.d_min < half_w + clear:
            need = max(float(e.pass_offset), g.d_max + half_w + clear)
            plateaus.append([g.s_min - float(e.buffer_before), g.s_max + float(e.buffer_after), need])
    plateaus.sort(key=lambda p: p[0])
    merged: list[list[float]] = []
    for p in plateaus:
        if merged and p[0] - merged[-1][1] < 2 * ramp + float(e.merge_gap):
            merged[-1][1] = max(merged[-1][1], p[1])
            merged[-1][2] = max(merged[-1][2], p[2])
        else:
            merged.append(list(p))

    d = np.zeros_like(s)
    for a, b, D in merged:
        up = smoothstep_cos((s - (a - ramp)) / ramp)
        down = 1.0 - smoothstep_cos((s - b) / ramp)
        d = np.maximum(d, D * np.minimum(up, down))

    # ---- checks
    if merged and (merged[0][0] - ramp < layout.start_s or merged[-1][1] + ramp > layout.goal_s):
        raise InfeasibleError("lane change would start before the start or end after the goal")
    if d.max(initial=0.0) > d_hi_road + 1e-6 or d.min(initial=0.0) < d_lo_road - 1e-6:
        raise InfeasibleError("offset profile leaves the road")
    for g in obstacles.groups:
        near = (s >= g.s_min - half_len - 1.0) & (s <= g.s_max + half_len + 1.0)
        if not near.any():
            continue
        lo, hi = d[near] - half_w, d[near] + half_w
        # car interval must stay clear of the obstacle interval by (clearance - tolerance)
        margin = clear - 0.15
        overlap = (lo < g.d_max + margin) & (hi > g.d_min - margin)
        if overlap.any():
            raise InfeasibleError(f"{g.kind} at s={g.s_min:.1f} cannot be passed with clearance")
    return d


def plan_speed(layout, offsets: np.ndarray, cfg, speed_limit: float) -> np.ndarray:
    sp = cfg.expert.speed
    s = layout.route_s
    ds = np.diff(s)
    # effective path curvature: route curvature + second derivative of the offset profile, lightly smoothed
    d2 = np.gradient(np.gradient(offsets, s), s)
    kappa = np.abs(layout.route_curvature) + np.abs(d2)
    win = max(1, int(round(3.0 / layout.spacing)))
    kappa = np.convolve(np.pad(kappa, win, mode="edge"), np.ones(2 * win + 1) / (2 * win + 1), mode="same")[win:-win]
    v = np.minimum(speed_limit, np.sqrt(float(sp.lateral_accel) / np.maximum(kappa, 1e-6)))
    shifted = np.abs(offsets) > 0.05
    v[shifted] = np.minimum(v[shifted], float(sp.pass_speed))
    v = np.maximum(v, float(sp.min_speed))
    v[s >= layout.goal_s] = 0.0
    a_dec, a_acc = float(sp.plan_decel), float(sp.plan_accel)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_dec * ds[i]))
    start = int(np.searchsorted(s, layout.start_s))
    v[start] = min(v[start], float(sp.min_speed))
    for i in range(start + 1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_acc * ds[i - 1]))
    v[:start] = v[start]
    return v
