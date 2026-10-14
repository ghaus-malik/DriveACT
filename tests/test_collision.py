import math

import torch

from driveact.environments.collision import footprint_box_hits, footprint_circle_hits, footprint_segment_hits

HALF = torch.tensor([2.2, 0.9])


def test_segment_hits():
    c, y = torch.zeros(1, 2), torch.zeros(1)
    segs = torch.tensor([[[2.0, -3.0, 2.0, 3.0], [2.3, -3.0, 2.3, 3.0], [-3.0, 0.5, 3.0, 0.5], [0, 1.0, 1, 1.0]]])
    hits = footprint_segment_hits(c, y, HALF, segs, torch.ones(1, 4, dtype=torch.bool))
    assert hits.tolist() == [[True, False, True, False]]
    # rotated car: segment at x = 1 crosses it, segment at x = 1.0 beyond the rotated half-width does not
    hits = footprint_segment_hits(c, torch.tensor([math.pi / 2]), HALF, segs[:, :2], torch.ones(1, 2, dtype=torch.bool))
    assert hits.tolist() == [[False, False]]


def test_box_and_circle_hits():
    c, y = torch.zeros(1, 2), torch.zeros(1)
    # box 0 spans y in [0.95, 2.65] (clear), box 1 spans y in [0.85, 2.55] (overlaps), box 2 reaches x = 1.94 (overlaps)
    boxes = torch.tensor([[[0.0, 1.8, 2.2, 0.85, 0.0, 0], [0.0, 1.7, 2.2, 0.85, 0.0, 0], [2.9, 0.0, 1.0, 0.3, 0.7, 3]]])
    hits = footprint_box_hits(c, y, HALF, boxes, torch.ones(1, 3, dtype=torch.bool))
    assert hits.tolist() == [[False, True, True]]
    circles = torch.tensor([[[2.3, 0.0, 0.18, 2], [2.5, 0.0, 0.18, 2]]])  # 0.10 m and 0.30 m from the front bumper
    hits = footprint_circle_hits(c, y, HALF, circles, torch.ones(1, 2, dtype=torch.bool))
    assert hits.tolist() == [[True, False]]
