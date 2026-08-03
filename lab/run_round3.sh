#!/bin/bash
# Round 3 — crossover test (H3): does eps-scaled looping overtake vanilla in
# the overtrained regime?
#
# Smoke runs (98M tokens) sit at ~2-3 tokens/param — deeply undertrained, the
# regime that maximally favors stored parameters. Ouro-style looped wins are
# reported at >>Chinchilla tokens/param. 20K steps = 490M tokens puts:
#   A  (21.3M non-emb) at ~23 tok/param  (~1.2x Chinchilla)
#   C  (10.6M)         at ~46            (~2.3x)
#   E3 ( 5.3M)         at ~92            (~4.6x)
# Same FLOPs/token for all arms (12 block applications). If the A-C / A-E3
# gap shrinks with tokens or flips sign, the recipe scales; if it is flat or
# growing, H1 is refuted at this scale.
set -e
for arm in A C E3; do
  ~/venv/bin/python train.py --arm $arm --steps 20000 --seed 7 \
    --data data500/fineweb_train.bin --val-data data500/fineweb_val.bin \
    --out results/long_${arm}_s7.json 2>&1 | tail -3
done
echo ROUND3_DONE
