#!/usr/bin/env bash
# Stages 2-7 as one sequential job (one GPU process at a time; every Isaac step re-checks GPU capacity).
#   nohup bash scripts/jobs/full_pipeline.sh > logs/full_pipeline.log 2>&1 &
# Existing data files and finished evaluations are skipped, so the job can be restarted after an interruption.
set -x
cd "$(dirname "$0")/../.."
source "${ISAACLAB_VENV:-$HOME/envs/isaaclab}/bin/activate"
ENVS="${ENVS:-16}"
NEVAL="${NEVAL:-48}"

isaac() { scripts/isaac.sh "$@"; }
evaluate() {  # evaluate <out> <args...>  (skipped if <out>/summary.json exists)
  local out="$1"; shift
  [ -f "$out/summary.json" ] || isaac scripts/evaluate.py "$@" --num_envs "$ENVS" --out "$out"
}
train() {  # train <out> <args...>  (skipped if <out>/best.pt exists)
  local out="$1"; shift
  [ -f "$out/best.pt" ] || python scripts/train_act.py "$@" --out "$out"
}

# ---- stage 2: scripted expert on every track (training layouts, collection stream)
for t in 1 2 3 4; do
  evaluate "results/stage2/expert_track$t" --policy expert --track "$t" --split train --episodes 16 --stream 0 --plots 3
done

# ---- demonstrations
declare -A N=([1]=200 [2]=200 [3]=300 [4]=400)
for t in 1 2 3 4; do
  [ -f "data/track${t}_train.h5" ] || isaac scripts/collect_demos.py --track "$t" --episodes "${N[$t]}" --num_envs "$ENVS" --out "data/track${t}_train.h5"
done
D1="data/track1_train.h5"; D2="data/track2_train.h5"; D3="data/track3_train.h5"; D4="data/track4_train.h5"

# ---- stage 4: track 1
train checkpoints/act_s4_track1 --data "$D1" --epochs 30
train checkpoints/bc_s4_track1 --data "$D1" --policy bc --epochs 30
for p in act bc; do
  evaluate "results/stage4/${p}_track1" --policy "$p" --checkpoint "checkpoints/${p}_s4_track1/best.pt" --track 1 --split test --episodes "$NEVAL"
done
evaluate results/stage4/expert_track1 --policy expert --track 1 --split test --episodes "$NEVAL"

# ---- stage 5: tracks 1-2 (parked vehicle)
train checkpoints/act_s5_track12 --data "$D1" "$D2" --epochs 25
evaluate results/stage5/act_track2 --policy act --checkpoint checkpoints/act_s5_track12/best.pt --track 2 --split test --episodes "$NEVAL"

# ---- stage 6: tracks 1-3 (multiple obstacles)
train checkpoints/act_s6_track123 --data "$D1" "$D2" "$D3" --epochs 25
evaluate results/stage6/act_track3 --policy act --checkpoint checkpoints/act_s6_track123/best.pt --track 3 --split test --episodes "$NEVAL"

# ---- stage 7: all tracks (track 4 training layouts only); ACT vs BC vs expert on held-out layouts
train checkpoints/act_all --data "$D1" "$D2" "$D3" "$D4" --epochs 15
train checkpoints/bc_all --data "$D1" "$D2" "$D3" "$D4" --policy bc --epochs 15
for t in 4 3 2 1; do
  evaluate "results/stage7/expert_track$t" --policy expert --track "$t" --split test --episodes "$NEVAL"
  evaluate "results/stage7/act_track$t" --policy act --checkpoint checkpoints/act_all/best.pt --track "$t" --split test --episodes "$NEVAL"
  evaluate "results/stage7/bc_track$t" --policy bc --checkpoint checkpoints/bc_all/best.pt --track "$t" --split test --episodes "$NEVAL"
  evaluate "results/stage7/act_te_track$t" --policy act --checkpoint checkpoints/act_all/best.pt --temporal_ensemble --track "$t" --split test --episodes "$NEVAL"
done
# training layouts with unseen obstacle placements (training distribution) vs held-out layouts (generalisation)
evaluate results/stage7/act_track4_trainlayouts --policy act --checkpoint checkpoints/act_all/best.pt --track 4 --split train --episodes "$NEVAL"
python scripts/summarize_results.py --root results --out results/summary.md
echo "full pipeline finished"
