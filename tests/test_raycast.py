import math

import torch

from driveact.sensors.raycast import PlanarLidar, nearest_segments, ray_boxes, ray_circles, ray_segments


def _dirs(angles):
    a = torch.tensor(angles, dtype=torch.float32)
    return torch.stack((torch.cos(a), torch.sin(a)), -1)[None]


def test_segment_box_circle_distances():
    o = torch.zeros(1, 2)
    d = _dirs([0.0, math.pi / 2, math.pi])
    seg = torch.tensor([[[5.0, -1.0, 5.0, 1.0]]])
    r = ray_segments(o, d, seg, torch.ones(1, 1, dtype=torch.bool))
    assert torch.isclose(r[0, 0], torch.tensor(5.0)) and torch.isinf(r[0, 1]) and torch.isinf(r[0, 2])
    box = torch.tensor([[[0.0, 4.0, 1.0, 0.5, 0.0]]])  # centre (0, 4), half sizes 1 x 0.5
    rb = ray_boxes(o, d, box, torch.ones(1, 1, dtype=torch.bool))
    assert torch.isclose(rb[0, 1], torch.tensor(3.5))
    circ = torch.tensor([[[-6.0, 0.0, 1.0]]])
    rc = ray_circles(o, d, circ, torch.ones(1, 1, dtype=torch.bool))
    assert torch.isclose(rc[0, 2], torch.tensor(5.0))


def test_masked_geometry_is_ignored_and_nearest_wins():
    o = torch.zeros(1, 2)
    d = _dirs([0.0])
    segs = torch.tensor([[[5.0, -1.0, 5.0, 1.0], [3.0, -1.0, 3.0, 1.0]]])
    r = ray_segments(o, d, segs, torch.tensor([[True, True]]))
    assert torch.isclose(r[0, 0], torch.tensor(3.0))
    r = ray_segments(o, d, segs, torch.tensor([[True, False]]))
    assert torch.isclose(r[0, 0], torch.tensor(5.0))


def test_lidar_scan_rotation_and_range(cfg):
    lidar = PlanarLidar(cfg.sensors.lidar)
    # square room 10 x 10 around the origin
    pts = [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]
    segs = torch.tensor([[[*pts[i], *pts[i + 1]] for i in range(4)]], dtype=torch.float32)
    m = torch.ones(1, 4, dtype=torch.bool)
    eb, em = torch.zeros(1, 0, 6), torch.zeros(1, 0, dtype=torch.bool)
    ec, ecm = torch.zeros(1, 0, 4), torch.zeros(1, 0, dtype=torch.bool)
    r0 = lidar.scan(torch.zeros(1, 2), torch.zeros(1), segs, m, eb, em, ec, ecm)
    assert r0.shape == (1, lidar.num_rays)
    assert r0.min() >= 5.0 - 1e-4 and r0.max() <= 5.0 * math.sqrt(2) + 1e-3
    # rotating the car by 90 deg in a square room gives the same scan shifted by a quarter turn
    r90 = lidar.scan(torch.zeros(1, 2), torch.tensor([math.pi / 2]), segs, m, eb, em, ec, ecm)
    assert torch.allclose(torch.roll(r0, -lidar.num_rays // 4, dims=1), r90, atol=1e-3)
    # sector 60 (front) of a car facing +x sees the wall at 5 m
    s = lidar.sectors(r0)
    assert s.shape == (1, lidar.num_sectors)
    assert abs(float(s[0, lidar.num_sectors // 2]) - 5.0) < 0.02
    far = lidar.scan(torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.bool), eb, em, ec, ecm)
    assert torch.all(far == lidar.max_range)


def test_nearest_segments_is_exact_when_k_large_enough():
    torch.manual_seed(0)
    segs = torch.rand(2, 50, 4) * 40 - 20
    mask = torch.ones(2, 50, dtype=torch.bool)
    o = torch.zeros(2, 2)
    d = _dirs(list(torch.linspace(-3.1, 3.1, 64).tolist())).expand(2, -1, -1)
    full = ray_segments(o, d, segs, mask).clamp(max=30.0)
    sk, mk, overflow = nearest_segments(o, segs, mask, 50, 30.0)
    assert torch.allclose(full, ray_segments(o, d, sk, mk).clamp(max=30.0))
    assert int(overflow.max()) == 0
