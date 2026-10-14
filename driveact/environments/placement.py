"""Placement of per-environment tracks in one simulation world.

Each layout is anchored at its start point (0, 0) and may extend in any direction, so environments are not simply
put at grid points: every layout's bounding box is centred on its environment's grid point, and the padded boxes
of all environments must be disjoint (otherwise a car could touch another environment's walls or cars).
"""

from __future__ import annotations

import numpy as np


def centered_origins(grid_points: np.ndarray, layouts) -> np.ndarray:
    """(N, 3) world origins of the layout frames so that each bounding-box centre lies on its grid point."""
    centers = np.array([[0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), 0.0] for b in (l.bounds() for l in layouts)])
    return np.asarray(grid_points, dtype=np.float64) - centers


def check_no_overlap(layouts, origins: np.ndarray, margin: float) -> None:
    """Raise if the (margin-padded) bounding boxes of any two environments' tracks intersect in the world."""
    boxes = []
    for layout, o in zip(layouts, origins):
        x0, y0, x1, y1 = layout.bounds(margin)
        boxes.append((x0 + o[0], y0 + o[1], x1 + o[0], y1 + o[1]))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                raise RuntimeError(f"tracks of environments {i} and {j} overlap; increase sim.env_spacing")


def grid_points(num_envs: int, spacing: float) -> np.ndarray:
    """A square-ish grid of environment points (same spacing rule as Isaac Lab's grid cloner)."""
    per_row = int(np.sqrt(num_envs))
    rows = int(np.ceil(num_envs / per_row))
    cols = int(np.ceil(num_envs / rows))
    pts = []
    for i in range(num_envs):
        r, c = divmod(i, cols)
        pts.append([0.5 * spacing * (rows - 1) - r * spacing, c * spacing - 0.5 * spacing * (cols - 1), 0.0])
    return np.asarray(pts)
