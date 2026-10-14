#!/usr/bin/env bash
# Stage 8: demonstration videos of the final ACT policy (RTX rendering, one episode each).
#   nohup bash scripts/jobs/stage8_videos.sh > logs/stage8.log 2>&1 &
set -x
cd "$(dirname "$0")/../.."
source "${ISAACLAB_VENV:-$HOME/envs/isaaclab}/bin/activate"
CK="${CK:-checkpoints/act_all/best.pt}"
scripts/isaac.sh scripts/visualize.py --policy act --checkpoint "$CK" --track 4 --layout_seeds -1 --episode 0 --out videos/act_track4_showcase
scripts/isaac.sh scripts/visualize.py --policy act --checkpoint "$CK" --track 4 --layout_seeds -1 --episode 0 --top overview --out videos/act_track4_showcase_overview
scripts/isaac.sh scripts/visualize.py --policy act --checkpoint "$CK" --track 3 --episode 0 --out videos/act_track3
scripts/isaac.sh scripts/visualize.py --policy act --checkpoint "$CK" --track 2 --episode 0 --out videos/act_track2
scripts/isaac.sh scripts/visualize.py --policy act --checkpoint "$CK" --track 1 --episode 0 --out videos/act_track1
mkdir -p docs/media
for v in act_track4_showcase act_track3 act_track2; do
  python scripts/make_gif.py "videos/$v/composite.mp4" "docs/media/$v.gif" --speed 3 --width 560
done
