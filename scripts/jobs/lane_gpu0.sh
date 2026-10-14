#!/usr/bin/env bash
# Stages 4-7, ACT lane (GPU 0). Run together with lane_gpu1.sh. Finished steps are skipped on restart.
set -x
cd "$(dirname "$0")/../.."
source "${ISAACLAB_VENV:-$HOME/envs/isaaclab}/bin/activate"
G=0; ENVS="${ENVS:-16}"; NEVAL="${NEVAL:-48}"
D1=data/track1_train.h5; D2=data/track2_train.h5; D3=data/track3_train.h5; D4=data/track4_train.h5
train() { local out="$1"; shift; [ -f "$out/best.pt" ] || python scripts/train_act.py --device cuda:$G --min_free_gb 2 "$@" --out "$out"; }
evaluate() { local out="$1"; shift; [ -f "$out/summary.json" ] || scripts/isaac.sh scripts/evaluate.py --gpu $G "$@" --num_envs "$ENVS" --out "$out"; }

train checkpoints/act_s4_track1 --data $D1 --epochs 30
evaluate results/stage4/act_track1 --policy act --checkpoint checkpoints/act_s4_track1/best.pt --track 1 --split test --episodes "$NEVAL"
train checkpoints/act_s5_track12 --data $D1 $D2 --epochs 25
evaluate results/stage5/act_track2 --policy act --checkpoint checkpoints/act_s5_track12/best.pt --track 2 --split test --episodes "$NEVAL"
train checkpoints/act_s6_track123 --data $D1 $D2 $D3 --epochs 25
evaluate results/stage6/act_track3 --policy act --checkpoint checkpoints/act_s6_track123/best.pt --track 3 --split test --episodes "$NEVAL"
train checkpoints/act_all --data $D1 $D2 $D3 $D4 --epochs 15
for t in 4 3 2 1; do
  evaluate results/stage7/act_track$t --policy act --checkpoint checkpoints/act_all/best.pt --track $t --split test --episodes "$NEVAL"
done
for t in 4 3; do
  evaluate results/stage7/act_te_track$t --policy act --checkpoint checkpoints/act_all/best.pt --temporal_ensemble --track $t --split test --episodes "$NEVAL"
done
evaluate results/stage7/act_track4_trainlayouts --policy act --checkpoint checkpoints/act_all/best.pt --track 4 --split train --episodes "$NEVAL"
echo "lane gpu0 finished"
