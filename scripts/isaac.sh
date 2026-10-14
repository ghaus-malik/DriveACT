#!/usr/bin/env bash
# Run a DriveACT script inside the Isaac Lab environment, headless, on a GPU that has spare capacity.
#
#   scripts/isaac.sh scripts/test_vehicle.py
#   scripts/isaac.sh scripts/collect_demos.py --track 1 --episodes 200 --num_envs 16 --out data/track1_train.h5
#
# ISAACLAB_VENV (default ~/envs/isaaclab) is activated if it exists. GPU selection is done by the script itself
# (--gpu auto: the GPU with the most free memory, refusing to start if none has --min_free_gb free).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${ISAACLAB_VENV:-$HOME/envs/isaaclab}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
script="$1"
shift
exec python "$script" --headless "$@"
