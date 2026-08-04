#!/bin/bash
# Round 8 — the SSM surprise: verify before believing.
#
# Round 6 (seed 7, 98M tokens): MC 5.1035 < MB 5.1156 < MA 5.1374 — the
# looped+scaled SSM BEATS vanilla SSM at matched FLOPs with 24% fewer params,
# on fresh data. Exactly what attention arms failed to do (C-A +0.0755).
# If real, this is the recipe headline: recurrence substitutes for params in
# SSM backbones, and eps=1/N compounds it.
#
# (1) Seed stability: MA/MB/MC at seeds 13, 29.
# (2) Trajectory: 20K-step MA vs MC on fresh 500M — does the MC lead grow,
#     hold, or decay with tokens? (Attention C-A grew against us; if MC-MA
#     grows in our favor, scaling bet is live.)
# Also MB4/MC4 at seed 13: is the N=4 inversion (-0.0068) noise or real?
set -e
for seed in 13 29; do
  for arm in MA MB MC; do
    ~/venv/bin/python train.py --arm $arm --steps 4000 --seed $seed \
      --out results/ssm_${arm}_s${seed}.json 2>&1 | tail -1
  done
done
for arm in MB4 MC4; do
  ~/venv/bin/python train.py --arm $arm --steps 4000 --seed 13 \
    --out results/ssm_${arm}_s13.json 2>&1 | tail -1
done
for arm in MA MC; do
  ~/venv/bin/python train.py --arm $arm --steps 20000 --seed 7 \
    --data data500/fineweb_train.bin --val-data data500/fineweb_val.bin \
    --out results/long_${arm}_s7.json 2>&1 | tail -1
done
echo ROUND8_DONE
