"""Matplotlib top-down plots of layouts, obstacles and driven trajectories (no simulator needed)."""

from __future__ import annotations

import numpy as np

ROAD = "#3b3d42"
SIDEWALK = "#c9c5bb"
WALL = "#7a7468"
GRASS = "#9bb37e"
CURB = "#f2f0ea"
CENTER = "#f2c230"
EDGE = "#ffffff"
ROUTE = "#2f7de1"
OBSTACLE_COLORS = {"parked_car": "#c0392b", "left_parked_car": "#8e44ad", "cone": "#ff7f0e", "barrier": "#95a5a6"}


def _dash_segments(points: np.ndarray, dash: float, gap: float) -> list[np.ndarray]:
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    out = []
    t = 0.0
    while t < s[-1]:
        ts = np.linspace(t, min(t + dash, s[-1]), 6)
        out.append(np.stack((np.interp(ts, s, points[:, 0]), np.interp(ts, s, points[:, 1])), axis=1))
        t += dash + gap
    return out


def draw_layout(ax, layout, obstacles=None, show_route: bool = True, markings: bool = True, decor=None):
    from matplotlib.patches import Polygon

    ax.set_facecolor(GRASS)
    for strip in layout.strips:
        poly = np.concatenate((strip.left, strip.right[::-1]), axis=0)
        ax.add_patch(Polygon(poly, closed=True, fc=ROAD if strip.kind == "road" else SIDEWALK, ec="none", zorder=1))
    for fan in layout.fans:
        ax.add_patch(Polygon(fan.boundary, closed=True, fc=ROAD, ec="none", zorder=1))
    if decor is not None:
        for b in decor.get("buildings", []):
            ax.add_patch(Polygon(b["corners"], closed=True, fc="#b8aea0", ec="#8c8273", lw=0.4, zorder=1))
    for curb in layout.curbs:
        ax.plot(curb[:, 0], curb[:, 1], color=CURB, lw=0.6, zorder=2)
    for wall in layout.walls:
        ax.plot(wall.points[:, 0], wall.points[:, 1], color=WALL, lw=1.4, zorder=2)
    if markings:
        for line in layout.center_lines:
            for d in _dash_segments(line, 3.0, 5.0):
                ax.plot(d[:, 0], d[:, 1], color=CENTER, lw=0.5, zorder=2)
        for line in layout.edge_lines:
            ax.plot(line[:, 0], line[:, 1], color=EDGE, lw=0.3, zorder=2)
    if show_route:
        ax.plot(layout.route[:, 0], layout.route[:, 1], color=ROUTE, lw=0.8, ls="--", zorder=3, label="route (right lane)")
    sx, sy, _ = layout.pose_at(layout.start_s)
    gx, gy, _ = layout.pose_at(layout.goal_s)
    ax.plot(sx, sy, marker="o", color="#2ecc71", ms=7, mec="k", zorder=6, label="start")
    ax.plot(gx, gy, marker="*", color="#f1c40f", ms=13, mec="k", zorder=6, label="goal")
    if obstacles is not None:
        draw_obstacles(ax, obstacles)
    ax.set_aspect("equal")
    x0, y0, x1, y1 = layout.bounds(margin=8.0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def draw_obstacles(ax, obstacles):
    from matplotlib.patches import Circle, Polygon

    for ob in obstacles.items:
        color = OBSTACLE_COLORS.get(ob.kind, "k")
        if ob.shape == "box":
            ax.add_patch(Polygon(ob.corners(), closed=True, fc=color, ec="k", lw=0.5, zorder=5))
        else:
            ax.add_patch(Circle((ob.x, ob.y), ob.radius * 1.6, fc=color, ec="k", lw=0.4, zorder=5))


def draw_trajectory(ax, xy: np.ndarray, color: str = "#e74c3c", label: str | None = None, lw: float = 1.4):
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, zorder=7, label=label)


def plot_layout_png(path: str, layout, obstacles=None, trajectories: dict | None = None, title: str | None = None,
                    offsets=None, dpi: int = 160):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0, x1, y1 = layout.bounds(margin=8.0)
    w = 9.0
    h = max(3.0, w * (y1 - y0) / max(x1 - x0, 1e-6))
    fig, ax = plt.subplots(figsize=(w, min(h, 14.0)))
    draw_layout(ax, layout, obstacles)
    if offsets is not None:  # reference path of the expert (route shifted by the planned lateral offset)
        from driveact.geometry import left_normal

        ref = layout.route + offsets[:, None] * left_normal(layout.route_heading)
        ax.plot(ref[:, 0], ref[:, 1], color="#16a085", lw=1.0, zorder=4, label="expert reference")
    for name, xy in (trajectories or {}).items():
        draw_trajectory(ax, xy, label=name)
    ax.set_title(title or layout.name, fontsize=10)
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
