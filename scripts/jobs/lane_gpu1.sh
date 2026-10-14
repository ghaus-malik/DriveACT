#!/usr/bin/env bash
# Stages 4-7, baseline/ablation lane (GPU 1): BC, expert, ACT without previous commands.
set -x
cd "$(dirname "$0")/../.."
source "${ISAACLAB_VENV:-$HOME/envs/isaaclab}/bin/activate"
G=1; ENVS="${ENVS:-16}"; NEVAL="${NEVAL:-48}"
D1=data/track1_train.h5; D2=data/track2_train.h5; D3=data/track3_train.h5; D4=data/track4_train.h5
train() { local out="$1"; shift; [ -f "$out/best.pt" ] || python scripts/train_act.py --device cuda:$G --min_free_gb 2 "$@" --out "$out"; }
evaluate() { local out="$1"; shift; [ -f "$out/summary.json" ] || scripts/isaac.sh scripts/evaluate.py --gpu $G "$@" --num_envs "$ENVS" --out "$out"; }

train checkpoints/bc_s4_track1 --data $D1 --policy bc --epochs 30
evaluate results/stage4/bc_track1 --policy bc --checkpoint checkpoints/bc_s4_track1/best.pt --track 1 --split test --episodes "$NEVAL"
for t in 1 2 3 4; do
  evaluate results/stage7/expert_track$t --policy expert --track $t --split test --episodes "$NEVAL"
done
train checkpoints/bc_all --data $D1 $D2 $D3 $D4 --policy bc --epochs 15
for t in 4 3 2 1; do
  evaluate results/stage7/bc_track$t --policy bc --checkpoint checkpoints/bc_all/best.pt --track $t --split test --episodes "$NEVAL"
done
train checkpoints/act_noprev_all --data $D1 $D2 $D3 $D4 --epochs 15 --set act.model.use_prev_actions=false act.train.prev_action_dropout=0.0 act.train.prev_action_noise=0.0
for t in 4 3; do
  evaluate results/stage7/act_noprev_track$t --policy act --checkpoint checkpoints/act_noprev_all/best.pt --track $t --split test --episodes "$NEVAL"
done
echo "lane gpu1 finished"
