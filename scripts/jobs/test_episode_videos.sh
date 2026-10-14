#!/usr/bin/env bash
# Videos of individual held-out test episodes of the final ACT evaluation (stage 7): same test layouts, obstacle
# placements (evaluation stream) and start poses as in results/stage7/act_track*/episodes.csv.
set -x
cd "$(dirname "$0")/../.."
source "${ISAACLAB_VENV:-$HOME/envs/isaaclab}/bin/activate"
CK=checkpoints/act_all/best.pt
render() {  # render <gpu> <track> <layout> <episode>
  local out="videos/test_episodes/track$2_layout$3_episode$4"
  [ -f "$out/episode.json" ] || scripts/isaac.sh scripts/visualize.py --gpu "$1" --policy act --checkpoint $CK \
    --track "$2" --split test --layout_seeds "$3" --episode "$4" --out "$out"
}
( render 0 4 1001 2; render 0 4 1005 6; render 0 4 1000 23; render 0 1 1001 1; render 0 2 1001 1 ) &
( render 1 3 1001 1; render 1 3 1002 6; render 1 3 1003 3; render 1 3 1003 11; render 1 4 1002 27 ) &
wait
echo "test episode videos finished"
