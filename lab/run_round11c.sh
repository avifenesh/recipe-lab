#!/bin/bash
# Round 11c — final hybrid variant + seed-13 legs for the live results.
#
# State after seed 7:
#   pure SSM:  MC768 4.9651... wait, MC 5.0034 < MA 5.0297 (-0.0263) FLIP HOLDS
#   hybrid:    HC768 (layer-loop) +0.28 FAIL; HD768 (model-loop, placement-
#              fair) +0.25 FAIL -> looping ATTENTION through shared weights
#              is what hurts, not placement.
#   HE768 (ssmloop): loop only SSM blocks, attention applied once each.
#              2 attn + 10 ssm apps, attn at 0,7. 82.5M vs HA768 112.2M.
#
# Runs (seed 7 HE, then seed 13 for MA/MC/HE/HA), skipping existing files
# (MC768 s13 may already be done by the killed 11b orchestrator):
set -e
while pgrep -f "train.py --arm" > /dev/null; do sleep 60; done
[ -f results/epoch39_HE768_s7.json ] || \
  ~/venv/bin/python train.py --arm HE768 --steps 20000 --seed 7 \
    --batch-size 12 \
    --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
    --out results/epoch39_HE768_s7.json 2>&1 | tail -1
for arm in MA768 MC768 HE768 HA768; do
  [ -f results/epoch39_${arm}_s13.json ] && continue
  ~/venv/bin/python train.py --arm $arm --steps 20000 --seed 13 \
    --batch-size 12 \
    --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
    --out results/epoch39_${arm}_s13.json 2>&1 | tail -1
done
echo ROUND11C_DONE
