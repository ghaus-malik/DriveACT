"""Demonstration storage in HDF5.

File layout (format ``driveact-demos-v1``)::

    /                         attrs: format, created, track, backend, control_dt, obs_spec (json), action_names (json),
                                     config (json of the full configuration used for collection)
    /episodes/000000/obs       (T, D) float32   observation at control step t
                    /actions   (T, 2) float32   expert action computed from obs[t]'s state (training label, clean)
                    /executed  (T, 2) float32   action actually applied at step t (label + DART noise if noisy)
                    /state     (T, 7) float32   x, y, yaw, speed, steering angle, route s, lateral offset (analysis)
                    /time      (T,)   float32   t * control_dt
                    attrs: episode, layout_seed, outcome, success, noisy, steps, obstacles (json), record (json)

Alignment: obs[t] is measured before action t is applied, and obs[t + 1] contains executed[t] as its "previous
command" features (checked by ``tests/test_dataset.py`` and ``verify_alignment``).
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict

import h5py
import numpy as np
import torch

FORMAT = "driveact-demos-v1"
ACTION_NAMES = ["steering", "longitudinal_acceleration"]
STATE_COLUMNS = ["x", "y", "yaw", "speed", "steer", "s", "lateral"]


class DemoRecorder:
    """Collects per-environment step buffers during :func:`run_episodes` and writes finished episodes."""

    def __init__(self, path: str, cfg, track: int, backend: str, obs_spec, keep_failures: bool = False):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = h5py.File(path, "w")
        self.f.attrs["format"] = FORMAT
        self.f.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.f.attrs["track"] = int(track)
        self.f.attrs["backend"] = backend
        self.f.attrs["control_dt"] = float(cfg.sim.control_dt)
        self.f.attrs["obs_spec"] = json.dumps(obs_spec.to_dict())
        self.f.attrs["action_names"] = json.dumps(ACTION_NAMES)
        self.f.attrs["config"] = json.dumps(cfg.to_dict())
        self.group = self.f.create_group("episodes")
        self.keep_failures = keep_failures
        self.buffers: dict[int, dict[str, list]] = {}
        self.count = 0
        self.dropped = 0

    def add_step(self, obs, labels, executed, env, active) -> None:
        st = env.state
        state = torch.stack((st["pos"][:, 0], st["pos"][:, 1], st["yaw"], st["v_long"], st["steer"], env.s, env.lateral), -1)
        obs, labels, executed, state = (t.detach().float().cpu().numpy() for t in (obs, labels, executed, state))
        for i in torch.nonzero(active).flatten().tolist():
            b = self.buffers.setdefault(i, {"obs": [], "actions": [], "executed": [], "state": []})
            b["obs"].append(obs[i])
            b["actions"].append(labels[i])
            b["executed"].append(executed[i])
            b["state"].append(state[i])

    def finish_episode(self, i: int, record, obstacles_json: str | None = None) -> None:
        b = self.buffers.pop(i, None)
        if b is None or not b["obs"]:
            return
        if not (record.success or self.keep_failures):
            self.dropped += 1
            return
        g = self.group.create_group(f"{self.count:06d}")
        T = len(b["obs"])
        g.create_dataset("obs", data=np.stack(b["obs"]).astype(np.float32), compression="gzip", compression_opts=4)
        g.create_dataset("actions", data=np.stack(b["actions"]).astype(np.float32))
        g.create_dataset("executed", data=np.stack(b["executed"]).astype(np.float32))
        g.create_dataset("state", data=np.stack(b["state"]).astype(np.float32))
        g.create_dataset("time", data=(np.arange(T) * float(self.f.attrs["control_dt"])).astype(np.float32))
        rec = asdict(record)
        rec.pop("extra", None)
        g.attrs["episode"] = int(record.episode)
        g.attrs["layout_seed"] = int(record.layout_seed)
        g.attrs["outcome"] = record.outcome
        g.attrs["success"] = bool(record.success)
        g.attrs["noisy"] = bool(record.noisy)
        g.attrs["steps"] = T
        g.attrs["record"] = json.dumps(rec)
        if obstacles_json is not None:
            g.attrs["obstacles"] = obstacles_json
        self.count += 1

    def close(self) -> None:
        self.f.attrs["num_episodes"] = self.count
        self.f.attrs["dropped_episodes"] = self.dropped
        self.f.close()


class EnvDemoRecorder(DemoRecorder):
    """Recorder that also stores each episode's obstacle configuration (needs the env at finish time)."""

    def __init__(self, *args, env=None, **kw):
        super().__init__(*args, **kw)
        self.env = env

    def finish_episode(self, i: int, record, obstacles_json: str | None = None) -> None:
        super().finish_episode(i, record, self.env.obstacles[i].to_json() if self.env is not None else None)


def load_episodes(paths: list[str] | str) -> tuple[list[dict], dict]:
    """Read all episodes (into memory) from one or more files -> (episodes, attrs of the first file)."""
    if isinstance(paths, str):
        paths = [paths]
    episodes, attrs = [], None
    for path in paths:
        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file")
            if attrs is None:
                attrs = {k: f.attrs[k] for k in f.attrs}
            elif json.loads(f.attrs["obs_spec"]) != json.loads(attrs["obs_spec"]):
                raise ValueError(f"{path}: observation layout differs from {paths[0]}")
            for name in sorted(f["episodes"].keys()):
                g = f["episodes"][name]
                episodes.append({
                    "obs": g["obs"][()],
                    "actions": g["actions"][()],
                    "executed": g["executed"][()],
                    "state": g["state"][()],
                    "file": os.path.basename(path),
                    "name": name,
                    "track": int(f.attrs["track"]),
                    **{k: (g.attrs[k].item() if hasattr(g.attrs[k], "item") else g.attrs[k]) for k in g.attrs},
                })
    return episodes, attrs


def verify_alignment(episode: dict, obs_spec) -> float:
    """Max |obs[t+1] previous-command features - executed[t]| (0 when observations and actions are aligned)."""
    prev = episode["obs"][1:, obs_spec.state.start + 3 : obs_spec.state.start + 5]
    return float(np.abs(prev - episode["executed"][:-1]).max()) if len(prev) else 0.0
