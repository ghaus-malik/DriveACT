#!/usr/bin/env bash
# Stage 8: final ACT (all tracks) demonstration videos, two renders in parallel (one per GPU).
set -x
cd "$(dirname "$0")/../.."
source "${ISAACLAB_VENV:-$HOME/envs/isaaclab}/bin/activate"
CK=checkpoints/act_all/best.pt
( scripts/isaac.sh scripts/visualize.py --gpu 0 --policy act --checkpoint $CK --track 4 --layout_seeds -1 --episode 0 --out videos/act_track4_showcase
  scripts/isaac.sh scripts/visualize.py --gpu 0 --policy act --checkpoint $CK --track 2 --episode 0 --out videos/act_track2 ) &
( scripts/isaac.sh scripts/visualize.py --gpu 1 --policy act --checkpoint $CK --track 3 --episode 0 --out videos/act_track3
  scripts/isaac.sh scripts/visualize.py --gpu 1 --policy act --checkpoint $CK --track 4 --layout_seeds 1000 --episode 1 --top overview --out videos/act_track4_layout1000_overview ) &
wait
echo "stage8 finished"
