"""Load the YAML configuration files into one nested attribute-style object.

Every ``configs/<name>.yaml`` becomes the section ``cfg.<name>``; track recipes live in
``configs/tracks/<name>.yaml`` and are loaded on demand with :func:`load_track_config`.

    cfg = load_config(overrides=["act.model.chunk_size=50"])
    cfg.vehicle.body.wheelbase  # 2.7
"""

from __future__ import annotations

import copy
import glob
import os
from typing import Any

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
TRACK_CONFIG_DIR = os.path.join(CONFIG_DIR, "tracks")


class Node(dict):
    """A dict whose keys are also attributes (recursively)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"config has no key '{name}'") from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __deepcopy__(self, memo) -> "Node":
        return _wrap(copy.deepcopy(dict(self), memo))

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Node) else copy.deepcopy(v) for k, v in self.items()}


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Node({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def apply_override(cfg: Node, override: str) -> None:
    """``"a.b.c=value"``; the value is parsed as YAML (numbers, lists, booleans)."""
    if "=" not in override:
        raise ValueError(f"Override '{override}' must look like key.path=value")
    path, raw = override.split("=", 1)
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node:
            raise KeyError(f"Unknown config section '{key}' in override '{override}'")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{path}'")
    node[keys[-1]] = _wrap(yaml.safe_load(raw))


def load_config(config_dir: str = CONFIG_DIR, overrides: list[str] | None = None) -> Node:
    cfg = Node()
    for path in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            cfg[name] = _wrap(yaml.safe_load(f) or {})
    for override in overrides or []:
        apply_override(cfg, override)
    return cfg


def load_track_config(track: str | int, config_dir: str = TRACK_CONFIG_DIR) -> Node:
    """``track`` is a file stem (``"track1"``) or a track number (``1``)."""
    name = f"track{track}" if isinstance(track, int) or str(track).isdigit() else str(track)
    path = os.path.join(config_dir, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No track config '{path}'")
    with open(path) as f:
        return _wrap(yaml.safe_load(f))


def track_names(config_dir: str = TRACK_CONFIG_DIR) -> list[str]:
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(config_dir, "*.yaml")))
