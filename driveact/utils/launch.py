"""Command-line plumbing shared by the scripts: backend selection, GPU choice and Isaac Sim start-up.

With ``--backend isaac`` the Isaac Sim application must be started (``AppLauncher``) before any ``isaaclab`` /
``omni`` / ``pxr`` import; :func:`parse_and_launch` does that and returns the app. With ``--backend kinematic``
(CPU test double) nothing is launched.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading


def base_parser(description: str, default_backend: str = "isaac") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--backend", choices=("isaac", "kinematic"), default=default_backend,
                   help="isaac = Isaac Lab physics; kinematic = CPU test double (tests and dry runs only)")
    p.add_argument("--gpu", default="auto",
                   help="'auto' picks a GPU with spare capacity (refuses to start if none), or an index, or 'keep' "
                        "to use CUDA_VISIBLE_DEVICES as set")
    p.add_argument("--min_free_gb", type=float, default=10.0, help="free GPU memory required by --gpu auto")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. act.model.chunk_size=50")
    return p


def select_gpu(args) -> None:
    """Set CUDA_VISIBLE_DEVICES before CUDA / Isaac Sim is initialised."""
    if args.gpu == "keep":
        return
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[driveact] using GPU {args.gpu}", flush=True)
        return
    from driveact.utils.gpu import describe, pick_gpu

    idx, gpus = pick_gpu(min_free_gb=args.min_free_gb)
    if not gpus:
        return
    print("[driveact] GPU status:\n" + describe(gpus), flush=True)
    if idx is None:
        sys.exit(f"[driveact] no GPU has >= {args.min_free_gb:.0f} GB free memory and <= 90 % average utilisation; "
                 "not starting (use --gpu N to force)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    print(f"[driveact] using GPU {idx}", flush=True)


def parse_and_launch(parser: argparse.ArgumentParser, enable_cameras: bool = False):
    """Parse arguments; for the Isaac backend add the AppLauncher options and start the app. -> (args, app | None)"""
    pre, _ = parser.parse_known_args()
    if pre.backend == "isaac":
        from isaaclab.app import AppLauncher  # noqa: WPS433 (deferred: only available inside Isaac Lab)

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        select_gpu(args)
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            args.device = "cuda:0"  # the chosen GPU is the only one visible
        if enable_cameras or getattr(args, "visual", False):
            args.enable_cameras = True
        app = AppLauncher(args).app
        return args, app
    parser.add_argument("--device", default="cpu", help="torch device for the kinematic backend")
    args = parser.parse_args()
    return args, None


def shutdown(app, code: int = 0, timeout_s: float = 60.0) -> None:
    """Close Isaac Sim without waiting for the replicator orchestrator (which can hang after headless runs) and
    force the process to exit with ``code`` if closing takes longer than ``timeout_s``."""
    sys.stdout.flush()
    sys.stderr.flush()
    if app is None:
        return
    timer = threading.Timer(timeout_s, lambda: os._exit(code))
    timer.daemon = True
    timer.start()
    try:
        app.close(wait_for_replicator=False)
    except TypeError:
        app.close()
    os._exit(code)


def run_main(main, app) -> None:
    """Run ``main()`` and always shut Isaac Sim down; exceptions are printed before the forced exit."""
    code = 1
    try:
        result = main()
        code = int(result) if isinstance(result, int) else 0
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code if isinstance(e.code, str) else f"exit {e.code}", file=sys.stderr)
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        shutdown(app, code)
    sys.exit(code)
