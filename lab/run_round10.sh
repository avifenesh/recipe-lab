#!/bin/bash
# Round 10 — seed verification of the round-9 scale flip (the no-single-seed
# rule that killed the SSM headline applies to good news too).
#
# H10: C768 best-val < A768 best-val on seeds 13 and 29 as well.
# 3/3 seeds -> the flip is real and the campaign has its scale-over recipe.
# Metric is best val over the run (early-stop), not final.
set -e
for seed in 13 29; do
  for arm in A768 C768; do
    ~/venv/bin/python train.py --arm $arm --steps 20000 --seed $seed \
      --batch-size 12 \
      --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
      --out results/epoch39_${arm}_s${seed}.json 2>&1 | tail -1
  done
done
echo ROUND10_DONE
