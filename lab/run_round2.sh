#!/bin/bash
# Round 2:
# (1) N=4 eps ladder — Thm 1 predicts monotone: eps=1/N > eps=1/sqrt(N) > eps=1
# (2) N=2 multi-seed — is the C<B gap real across seeds?
# (3) LR probe at 2x base LR — scaled arm should tolerate, unscaled degrade
set -e
for arm in E1 E2 E3; do
  ~/venv/bin/python train.py --arm $arm --steps 4000 --seed 7 \
    --out results/n4_${arm}_s7.json 2>&1 | tail -2
done
for seed in 13 29; do
  for arm in B C; do
    ~/venv/bin/python train.py --arm $arm --steps 4000 --seed $seed \
      --out results/smoke_${arm}_s${seed}.json 2>&1 | tail -2
  done
done
for arm in B C; do
  ~/venv/bin/python train.py --arm $arm --steps 2000 --seed 7 --lr 1.2e-3 \
    --out results/hilr_${arm}_s7.json 2>&1 | tail -2
done
echo ROUND2_DONE
