#!/bin/bash
# Round 3c — LR grid extension. Round 3b left every arm right-censored (all
# preferred the top rung 1.2e-3) and at that rung the looped arms nearly
# matched vanilla (B 4.7588 vs A 4.7608 at 2000 steps) — meaning the round-3
# fresh-data verdict (run at 6e-4) may be an artifact of a globally
# suboptimal LR. Two questions:
#  (1) Where do the optima actually sit? Add 2.4e-3 and 4.8e-3 rungs.
#      Proof 4 predicts the unscaled arms (B, E1) hit their edge first.
#  (2) Does the ranking at 4000 steps change at 1.2e-3? Rerun the five arms
#      at 4000 steps / 1.2e-3 to compare against the 6e-4 smoke table.
set -e
for lr in 2.4e-3 4.8e-3; do
  for arm in A B C E1 E3; do
    ~/venv/bin/python train.py --arm $arm --steps 2000 --seed 7 --lr $lr \
      --out results/lr2k_${arm}_${lr}.json 2>&1 | tail -1
  done
done
for arm in A B C E1 E3; do
  ~/venv/bin/python train.py --arm $arm --steps 4000 --seed 7 --lr 1.2e-3 \
    --out results/hi4k_${arm}_s7.json 2>&1 | tail -1
done
echo ROUND3C_DONE
