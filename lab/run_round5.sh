#!/bin/bash
# Round 5 — data-constrained regime: does looping win when data, not params,
# is the binding constraint?
#
# Round 3 (fresh data) refuted the crossover: C-A gap GROWS with tokens
# (+0.044 @98M -> +0.076 @490M). Params win when every token is new.
# But today's frontier constraint is the data wall: training repeats data.
# Fewer stored params + more compute per token = less memorization pressure.
# Muennighoff 2023: epochs <=4 nearly free, beyond that params saturate —
# exactly where a 30M-param C should hold up better than a 40.6M-param A.
#
# Design: 50M-token train subset (truncated from the 500M bin), same 20K
# steps / 491M tokens seen = ~9.8 epochs. Val = fresh held-out from data500.
# Compare end-state val loss AND overfit dynamics (val curve turning up).
# H5: (C-A)|repeated < (C-A)|fresh = +0.0755, ideally < 0.
set -e
mkdir -p data50
[ -f data50/fineweb_train.bin ] || head -c 100000000 data500/fineweb_train.bin > data50/fineweb_train.bin
[ -f data50/fineweb_val.bin ]   || cp data500/fineweb_val.bin data50/fineweb_val.bin
for arm in A C E3; do
  ~/venv/bin/python train.py --arm $arm --steps 20000 --seed 7 \
    --data data50/fineweb_train.bin --val-data data50/fineweb_val.bin \
    --out results/epoch_${arm}_s7.json 2>&1 | tail -1
done
echo ROUND5_DONE
