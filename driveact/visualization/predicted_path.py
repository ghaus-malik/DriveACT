"""Turn a predicted action chunk into the path it would produce (for display only).

ACT predicts controls, not positions. To show its plan, the chunk is executed open-loop on the kinematic bicycle
model with the same actuator limits as the car, starting from the current measured state. The result is labelled
as such in videos; it is not a position prediction of the network.
"""

from __future__ import annotations

import torch

from driveact.vehicle.control import Actuators, VehicleGeometry, actions_to_physical


@torch.no_grad()
def rollout_chunk(chunk: torch.Tensor, pos: torch.Tensor, yaw: torch.Tensor, speed: torch.Tensor, steer: torch.Tensor,
                  g: VehicleGeometry, dt: float, substeps: int = 5) -> torch.Tensor:
    """chunk (N, k, 2) normalised actions; state (N, ...) -> footprint-centre path (N, k, 2)."""
    N, k, _ = chunk.shape
    act = Actuators(N, g, chunk.device)
    act.steer = steer.clone()
    act.v_cmd = speed.clone()
    v = speed.clone()
    fwd = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
    rear = pos + g.rear_axle_x * fwd
    yaw = yaw.clone()
    out = []
    h = dt / substeps
    for t in range(k):
        s_t, a_t = actions_to_physical(chunk[:, t], g)
        for _ in range(substeps):
            act.step(h, s_t, a_t, v)
            v = act.v_cmd.clone()
            rear = rear + h * v[:, None] * torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
            yaw = yaw + h * v * torch.tan(act.steer) / g.wheelbase
        out.append(rear - g.rear_axle_x * torch.stack((torch.cos(yaw), torch.sin(yaw)), -1))
    return torch.stack(out, dim=1)
