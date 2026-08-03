#!/bin/bash
# Round 4 — N-sweep + learnable lambda, 4000-step smokes, seed 7.
#
# (1) I arms complete the val-loss-vs-N curve at fixed 12 applications/token:
#     N=6 (2 stored blocks) and N=12 (1 stored block), scaled vs unscaled.
#     Theory: the eps advantage grows with N (round 2 showed 6x from N=2 to
#     N=4); param starvation also grows with N. Where the curve turns tells
#     us the recipe's operating point.
# (2) H arms: does learning lambda beat fixed lambda=1 at N=2 and N=4?
set -e
for arm in I6 I6u I12 I12u H H4; do
  ~/venv/bin/python train.py --arm $arm --steps 4000 --seed 7 \
    --out results/n_sweep_${arm}_s7.json 2>&1 | tail -1
done
echo ROUND4_DONE
