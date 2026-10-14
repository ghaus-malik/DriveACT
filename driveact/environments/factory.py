"""Create layouts, a physics backend and a :class:`DriveEnv` from command-line style options."""

from __future__ import annotations

from driveact.config import load_track_config
from driveact.environments.drive_env import DriveEnv
from driveact.tracks.generators import make_layout, split_seeds


def env_layouts(cfg, track: int, layout_seeds: list[int], num_envs: int) -> list:
    """Assign layouts round-robin to environments (every environment keeps its layout for the whole run)."""
    cache = {s: make_layout(track, s, cfg) for s in dict.fromkeys(layout_seeds)}
    return [cache[layout_seeds[i % len(layout_seeds)]] for i in range(num_envs)]


def resolve_layout_seeds(track: int, split: str | None, seeds: list[int] | None) -> list[int]:
    if seeds:
        return [int(s) for s in seeds]
    return split_seeds(track, split or "train")


def make_backend(name: str, cfg, num_envs: int, device: str, layouts=None, track_cfg=None, visual: bool = False,
                 decor: bool = False, **kw):
    if name == "kinematic":
        from driveact.vehicle.kinematic import KinematicBackend

        return KinematicBackend(cfg, num_envs, device)
    if name == "isaac":
        from driveact.environments.isaac_backend import IsaacBackend  # requires a running Isaac Sim app

        return IsaacBackend(cfg, num_envs, layouts, track_cfg, device=device, visual=visual, decor=decor, **kw)
    raise ValueError(f"unknown backend '{name}'")


def make_env(cfg, track: int, layout_seeds: list[int], num_envs: int, backend: str = "kinematic", device: str = "cpu",
             obstacle_stream: int = 0, visual: bool = False, decor: bool = False, **kw) -> DriveEnv:
    track_cfg = load_track_config(track)
    layouts = env_layouts(cfg, track, layout_seeds, num_envs)
    be = make_backend(backend, cfg, num_envs, device, layouts=layouts, track_cfg=track_cfg, visual=visual, decor=decor, **kw)
    return DriveEnv(cfg, track, layouts, be, obstacle_stream=obstacle_stream, track_cfg=track_cfg)
