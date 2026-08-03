#!/bin/bash
# Round 6 — SSM-mixer arms (proof 5 on GPU): MA/MB/MC at N=2, MB4/MC4 at N=4.
# 4000-step smokes, seed 7, same data/protocol as round 1.
# Prediction (pre-registered): MC < MB, MC4 < MB4, and the scaled-vs-unscaled
# gap at each N exceeds the attention-mixer gap at the same N (proof 5:
# state recurrence compounds the loop correlation).
set -e
for arm in MA MB MC MB4 MC4; do
  ~/venv/bin/python train.py --arm $arm --steps 4000 --seed 7 \
    --out results/ssm_${arm}_s7.json 2>&1 | tail -1
done
echo ROUND6_DONE
