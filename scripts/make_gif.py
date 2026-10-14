"""Make a small, sped-up GIF from an MP4 (for the README).

    python scripts/make_gif.py videos/act_track4_showcase/composite.mp4 docs/media/act_track4.gif --speed 3 --width 560
"""

from __future__ import annotations

import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed-up (frames are dropped)")
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--fps", type=float, default=12.5)
    p.add_argument("--max_seconds", type=float, default=None, help="limit the source duration")
    args = p.parse_args()

    import cv2
    import imageio.v2 as imageio

    reader = imageio.get_reader(args.src)
    src_fps = float(reader.get_meta_data().get("fps", 25.0))
    step = max(1, int(round(src_fps * args.speed / args.fps)))
    frames = []
    for i, frame in enumerate(reader):
        if args.max_seconds is not None and i / src_fps > args.max_seconds:
            break
        if i % step:
            continue
        h, w = frame.shape[:2]
        frames.append(cv2.resize(frame, (args.width, int(round(h * args.width / w))), interpolation=cv2.INTER_AREA))
    reader.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    imageio.mimsave(args.dst, frames, duration=1.0 / args.fps, loop=0)
    print(f"wrote {args.dst}: {len(frames)} frames, {os.path.getsize(args.dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
