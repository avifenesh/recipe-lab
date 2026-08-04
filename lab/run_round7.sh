#!/bin/bash
# Round 7 — push the epoch axis to the flip point.
#
# Round 5 at 9.8 epochs: repeat tax monotone in params (A +0.133, C +0.102,
# E3 +0.097), C-A gap falling through the back half of training. Muennighoff:
# repeated-token value halves around ~15 epochs and params saturate hard
# beyond. 12.5M unique tokens x 20K steps = ~39 epochs, deep in the
# saturation regime.
#
# H7: at ~39 epochs the C-A gap goes <= 0 (loop beats params when data, not
# capacity, binds). Secondary: E3-A shrinks below its round-5 value, and A
# shows an actual val-loss turn-up (overfit proper, not just tax).
set -e
mkdir -p data12
[ -f data12/fineweb_train.bin ] || head -c 25000000 data500/fineweb_train.bin > data12/fineweb_train.bin
[ -f data12/fineweb_val.bin ]   || cp data500/fineweb_val.bin data12/fineweb_val.bin
for arm in A C E3; do
  ~/venv/bin/python train.py --arm $arm --steps 20000 --seed 7 \
    --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
    --out results/epoch39_${arm}_s7.json 2>&1 | tail -1
done
echo ROUND7_DONE
