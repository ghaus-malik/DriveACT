"""Heads-up display, multi-view composition and MP4 writing for demonstration videos (OpenCV + imageio)."""

from __future__ import annotations

import os

import numpy as np

GREEN = (40, 210, 90)
RED = (230, 50, 40)
WHITE = (245, 245, 245)
AMBER = (255, 180, 30)


def _text(img, text, org, scale=0.6, color=WHITE, thickness=1):
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _bar(img, x, y, w, h, value, lo, hi, color, label):
    import cv2

    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    zero = int(x + w * (0 - lo) / (hi - lo))
    v = int(x + w * (np.clip(value, lo, hi) - lo) / (hi - lo))
    cv2.rectangle(img, (min(zero, v), y), (max(zero, v), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    _text(img, label, (x, y - 6), 0.5)


def draw_hud(img: np.ndarray, hud: dict) -> np.ndarray:
    """Speed, steering and acceleration commands, goal progress, collision indicator and labels."""
    import cv2

    img = np.ascontiguousarray(img).copy()
    H, W = img.shape[:2]
    panel = img.copy()
    cv2.rectangle(panel, (10, 10), (330, 200), (0, 0, 0), -1)
    img = cv2.addWeighted(panel, 0.35, img, 0.65, 0)
    _text(img, hud.get("title", ""), (20, 38), 0.7, WHITE, 2)
    _text(img, f"speed {hud['speed'] * 3.6:5.1f} km/h", (20, 68), 0.65)
    _bar(img, 20, 92, 290, 12, hud["steer_cmd"], -1, 1, AMBER, f"steering cmd {hud['steer_cmd']:+.2f}")
    _bar(img, 20, 128, 290, 12, hud["accel_cmd"], -1, 1, GREEN if hud["accel_cmd"] >= 0 else RED,
         f"accel/brake cmd {hud['accel_cmd']:+.2f}")
    _bar(img, 20, 164, 290, 12, hud["progress"], 0, 1, (80, 160, 255), f"route progress {100 * hud['progress']:.0f} %")
    status = hud.get("status", "")
    color = RED if hud.get("collision") else (GREEN if hud.get("success") else WHITE)
    if hud.get("collision"):
        cv2.rectangle(img, (W - 250, 15), (W - 15, 55), RED, -1)
        _text(img, "COLLISION", (W - 235, 45), 0.9, WHITE, 2)
    elif status:
        _text(img, status, (W - 330, 45), 0.8, color, 2)
    if hud.get("legend"):
        _text(img, hud["legend"], (20, H - 18), 0.5)
    return img


def compose(main: np.ndarray, insets: list[np.ndarray], scale: float = 0.3) -> np.ndarray:
    """Main view with smaller views stacked in the bottom-right corner."""
    import cv2

    out = main.copy()
    H, W = out.shape[:2]
    y = H - 10
    for inset in insets:
        h, w = int(H * scale), int(W * scale)
        small = cv2.resize(inset, (w, h), interpolation=cv2.INTER_AREA)
        y0 = y - h
        out[y0:y, W - 10 - w:W - 10] = small
        cv2.rectangle(out, (W - 10 - w, y0), (W - 10, y), (230, 230, 230), 2)
        y = y0 - 8
    return out


class VideoWriter:
    def __init__(self, path: str, fps: int):
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=6, macro_block_size=8)
        self.frames = 0

    def add(self, frame: np.ndarray) -> None:
        self.w.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        self.w.close()
