import math

import torch

from driveact.vehicle.control import (
    Actuators,
    VehicleGeometry,
    ackermann_angles,
    actions_to_physical,
    bicycle_angle,
    physical_to_actions,
    wheel_angular_speeds,
)
from driveact.vehicle.kinematic import KinematicBackend


def test_action_mapping_round_trip(cfg):
    g = VehicleGeometry.from_config(cfg.vehicle)
    a = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [2.0, -3.0]])
    steer, accel = actions_to_physical(a, g)
    assert torch.allclose(steer[:2], torch.tensor([g.max_steer, -g.max_steer]))
    assert torch.allclose(accel[:3], torch.tensor([g.max_accel, -g.max_brake, -0.5 * g.max_brake]))
    assert torch.allclose(physical_to_actions(steer, accel, g)[:3], a[:3])


def test_ackermann_geometry(cfg):
    g = VehicleGeometry.from_config(cfg.vehicle)
    s = torch.tensor([0.3, -0.3, 1e-4])
    left, right = ackermann_angles(s, g)
    assert left[0] > s[0] > right[0] > 0  # left turn: inner (left) wheel turns more
    assert right[1] < s[1] < left[1] < 0
    assert torch.allclose(bicycle_angle(left, right, g), s, atol=1e-5)
    # both wheels point at the same turning centre on the rear-axle line
    R = g.wheelbase / math.tan(0.3)
    assert math.isclose(math.tan(float(left[0])), g.wheelbase / (R - g.track_width / 2), rel_tol=1e-4)


def test_wheel_speeds(cfg):
    g = VehicleGeometry.from_config(cfg.vehicle)
    w = wheel_angular_speeds(torch.tensor([10.0, 10.0]), torch.tensor([0.0, 0.3]), g)
    assert torch.allclose(w[0], torch.full((4,), 10.0 / g.wheel_radius))
    fl, fr, rl, rr = w[1]
    assert rl < rr and fl < fr and fl > rl  # left turn: left (inner) side slower, front faster than rear


def test_actuators_slew_and_no_reverse(cfg):
    g = VehicleGeometry.from_config(cfg.vehicle)
    act = Actuators(1, g, "cpu")
    act.step(0.1, torch.tensor([g.max_steer]), torch.tensor([-6.0]), torch.tensor([0.0]))
    assert math.isclose(float(act.steer[0]), g.max_steer_rate * 0.1, rel_tol=1e-5)
    assert float(act.v_cmd[0]) == 0.0
    for _ in range(100):
        act.step(0.1, torch.tensor([0.0]), torch.tensor([3.0]), act.v_cmd.clone())
    assert float(act.v_cmd[0]) == g.max_speed


def test_kinematic_double_turning_radius(cfg):
    be = KinematicBackend(cfg, 1)
    be.reset(torch.tensor([0]), torch.zeros(1, 3))
    for _ in range(80):
        st = be.state()
        a = 1.5 * (4.0 - float(st["v_long"][0])) / be.g.max_accel
        be.step(torch.tensor([[0.5 if _ > 20 else 0.0, a]]))
    st = be.state()
    R = float(st["v_long"][0] / st["yaw_rate"][0])
    assert math.isclose(R, be.g.wheelbase / math.tan(0.5 * be.g.max_steer), rel_tol=0.02)
