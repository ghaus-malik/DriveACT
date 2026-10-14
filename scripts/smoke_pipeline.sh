#!/usr/bin/env bash
# End-to-end smoke test of the pipeline: collect a few demonstrations -> train ACT and BC for a few epochs ->
# evaluate in closed loop. It checks that every stage runs; the policies are far too undertrained to drive well.
#
#   scripts/smoke_pipeline.sh kinematic        # CPU test double (no simulator, ~5 min on a laptop)
#   scripts/smoke_pipeline.sh isaac            # Isaac Lab physics on a GPU with spare capacity
#
# Outputs go to results/smoke_<backend>/ (data and checkpoints included; delete afterwards).
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${1:-kinematic}"
OUT="results/smoke_${BACKEND}"
EPOCHS="${EPOCHS:-3}"
mkdir -p "$OUT"
if [ "$BACKEND" = "isaac" ]; then
  RUN=(scripts/isaac.sh)
  ENVS=8
else
  RUN=(python)
  ENVS=4
  EXTRA=(--backend kinematic)
fi
EXTRA=("${EXTRA[@]+"${EXTRA[@]}"}")

echo "== collect (track 2, 16 episodes)"
"${RUN[@]}" scripts/collect_demos.py "${EXTRA[@]}" --track 2 --episodes 16 --num_envs "$ENVS" --out "$OUT/track2.h5"
echo "== train ACT ($EPOCHS epochs)"
python scripts/train_act.py --data "$OUT/track2.h5" --policy act --out "$OUT/act" --epochs "$EPOCHS" --threads 4
echo "== train BC ($EPOCHS epochs)"
python scripts/train_act.py --data "$OUT/track2.h5" --policy bc --out "$OUT/bc" --epochs "$EPOCHS" --threads 4
for policy in act bc; do
  echo "== evaluate $policy"
  "${RUN[@]}" scripts/evaluate.py "${EXTRA[@]}" --policy "$policy" --checkpoint "$OUT/$policy/best.pt" --track 2 \
    --split test --episodes 4 --num_envs 4 --plots 2 --out "$OUT/eval_$policy"
done
echo "== evaluate expert"
"${RUN[@]}" scripts/evaluate.py "${EXTRA[@]}" --policy expert --track 2 --split test --episodes 4 --num_envs 4 --plots 2 \
  --out "$OUT/eval_expert"
echo "smoke pipeline finished: $OUT"
