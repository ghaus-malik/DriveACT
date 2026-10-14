#!/usr/bin/env bash
# Full experimental pipeline (stages 2-8). Every Isaac step picks a GPU with spare capacity (--gpu auto) and refuses
# to start otherwise; run the steps one at a time on a shared machine:
#
#   scripts/run_experiments.sh expert       # stage 2: scripted expert on every track (train layouts)
#   scripts/run_experiments.sh collect      # demonstrations for tracks 1-4 (train layouts, collection stream)
#   scripts/run_experiments.sh train        # ACT per stage + final ACT/BC on all tracks + chunk-size ablation
#   scripts/run_experiments.sh evaluate     # closed-loop evaluation on held-out layouts / unseen obstacle placements
#   scripts/run_experiments.sh videos       # stage 8: demonstration videos (RTX rendering)
#
# Episode counts can be overridden: N1=200 N2=200 N3=300 N4=400 NEVAL=100 scripts/run_experiments.sh collect
set -euo pipefail
cd "$(dirname "$0")/.."
N1="${N1:-200}"; N2="${N2:-200}"; N3="${N3:-300}"; N4="${N4:-400}"; NEVAL="${NEVAL:-100}"; ENVS="${ENVS:-16}"
ISAAC=(scripts/isaac.sh)
mkdir -p data checkpoints results/eval logs

case "${1:-}" in
expert)
  for t in 1 2 3 4; do
    "${ISAAC[@]}" scripts/evaluate.py --policy expert --track "$t" --split train --episodes 32 --num_envs "$ENVS" \
      --stream 0 --out "results/stage2/expert_track$t"
  done
  ;;
collect)
  declare -A N=([1]=$N1 [2]=$N2 [3]=$N3 [4]=$N4)
  for t in 1 2 3 4; do
    "${ISAAC[@]}" scripts/collect_demos.py --track "$t" --split train --episodes "${N[$t]}" --num_envs "$ENVS" \
      --out "data/track${t}_train.h5"
  done
  ;;
train)
  # stage 4-6: one ACT per curriculum stage (track 1; tracks 1-2; tracks 1-3)
  python scripts/train_act.py --data data/track1_train.h5 --out checkpoints/act_s4_track1
  python scripts/train_act.py --data data/track1_train.h5 data/track2_train.h5 --out checkpoints/act_s5_track12
  python scripts/train_act.py --data data/track1_train.h5 data/track2_train.h5 data/track3_train.h5 --out checkpoints/act_s6_track123
  # final models on all training data (track 4 train layouts only)
  python scripts/train_act.py --data "data/track*_train.h5" --out checkpoints/act_all
  python scripts/train_act.py --data "data/track*_train.h5" --policy bc --out checkpoints/bc_all
  # chunk-size ablation
  for k in 10 50; do
    python scripts/train_act.py --data "data/track*_train.h5" --out "checkpoints/act_all_k$k" --set "act.model.chunk_size=$k"
  done
  ;;
evaluate)
  for t in 1 2 3 4; do
    "${ISAAC[@]}" scripts/evaluate.py --policy expert --track "$t" --split test --episodes "$NEVAL" --num_envs "$ENVS" --out "results/eval/expert_track$t"
    "${ISAAC[@]}" scripts/evaluate.py --policy bc --checkpoint checkpoints/bc_all/best.pt --track "$t" --split test --episodes "$NEVAL" --num_envs "$ENVS" --out "results/eval/bc_track$t"
    "${ISAAC[@]}" scripts/evaluate.py --policy act --checkpoint checkpoints/act_all/best.pt --track "$t" --split test --episodes "$NEVAL" --num_envs "$ENVS" --out "results/eval/act_track$t"
    "${ISAAC[@]}" scripts/evaluate.py --policy act --checkpoint checkpoints/act_all/best.pt --track "$t" --split test --episodes "$NEVAL" --num_envs "$ENVS" --temporal_ensemble --out "results/eval/act_te_track$t"
    for k in 10 50; do
      "${ISAAC[@]}" scripts/evaluate.py --policy act --checkpoint "checkpoints/act_all_k$k/best.pt" --track "$t" --split test --episodes "$NEVAL" --num_envs "$ENVS" --out "results/eval/act_k${k}_track$t"
    done
  done
  # training-distribution performance (train layouts, unseen obstacle placements) vs generalisation (test layouts)
  "${ISAAC[@]}" scripts/evaluate.py --policy act --checkpoint checkpoints/act_all/best.pt --track 4 --split train --episodes "$NEVAL" --num_envs "$ENVS" --out results/eval/act_track4_trainlayouts
  python scripts/summarize_results.py --root results/eval --out results/eval/summary.md
  ;;
videos)
  "${ISAAC[@]}" scripts/visualize.py --policy act --checkpoint checkpoints/act_all/best.pt --track 4 --layout_seeds -1 --episode 0 --out videos/act_track4_showcase
  "${ISAAC[@]}" scripts/visualize.py --policy act --checkpoint checkpoints/act_all/best.pt --track 2 --episode 0 --out videos/act_track2
  "${ISAAC[@]}" scripts/visualize.py --policy act --checkpoint checkpoints/act_all/best.pt --track 3 --episode 0 --out videos/act_track3
  "${ISAAC[@]}" scripts/visualize.py --policy act --checkpoint checkpoints/act_all/best.pt --track 4 --layout_seeds -1 --episode 0 --top overview --out videos/act_track4_overview
  ;;
*)
  sed -n '2,13p' "$0"
  exit 1
  ;;
esac
