#!/usr/bin/env bash
# Stage 5 diagnosis: ACT on tracks 1-2 with (A) previous-command dropout/noise and (B) without previous commands.
# Trains both in parallel (one per GPU), then evaluates both on held-out Track 2 layouts.
set -x
cd "$(dirname "$0")/../.."
source "${ISAACLAB_VENV:-$HOME/envs/isaaclab}/bin/activate"
D="data/track1_train.h5 data/track2_train.h5"
python scripts/train_act.py --data $D --epochs 25 --device cuda:0 --out checkpoints/dbg_A_prevdrop \
  --set act.train.prev_action_dropout=0.5 act.train.prev_action_noise=0.5 > logs/dbg_A_train.log 2>&1 &
python scripts/train_act.py --data $D --epochs 25 --device cuda:1 --out checkpoints/dbg_B_noprev \
  --set act.model.use_prev_actions=false > logs/dbg_B_train.log 2>&1 &
wait
scripts/isaac.sh scripts/evaluate.py --gpu 0 --policy act --checkpoint checkpoints/dbg_A_prevdrop/best.pt --track 2 --split test \
  --episodes 48 --num_envs 16 --out results/debug_s5/act_A_prevdrop_track2 > logs/dbg_A_eval.log 2>&1 &
scripts/isaac.sh scripts/evaluate.py --gpu 1 --policy act --checkpoint checkpoints/dbg_B_noprev/best.pt --track 2 --split test \
  --episodes 48 --num_envs 16 --out results/debug_s5/act_B_noprev_track2 > logs/dbg_B_eval.log 2>&1 &
wait
echo "debug_copycat finished"
