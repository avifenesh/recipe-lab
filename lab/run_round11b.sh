#!/bin/bash
# Round 11b — revised after HC768 s7 failure (+0.2766 vs HA768).
#
# Diagnosis: layer-loop on a hybrid concentrates the two attention
# applications back-to-back (block0 x2 at positions 0,1 of 12), while HA768
# spreads them (positions 0 and 6). Interleaving is known-critical for
# hybrids. HD768 (model-loop) restores exact placement parity: L0..L5 twice
# puts attention at applications 0 and 6, same as HA768, while still halving
# stored params.
#
# Revised predictions:
#   P1': HD768 best-val < HA768 best-val (flip holds once placement matches).
#   P2: MC768 < MA768 (pure SSM has no placement issue; original bet stands).
#   P3: degradation ordering by params still holds everywhere.
# Runs: finish MA768 s7 (already running from 11), then MC768 s7, HD768 s7;
# seed 13 for whichever pairs sign-agree on seed 7... run both anyway (bar
# unchanged: 2 seeds minimum, 3rd iff 2/2).
set -e
while pgrep -f "train.py --arm MA768" > /dev/null; do sleep 60; done
for seed in 7 13; do
  for arm in MC768 HD768 MA768; do
    [ "$seed" = "7" ] && [ "$arm" = "MA768" ] && continue  # done by round 11
    ~/venv/bin/python train.py --arm $arm --steps 20000 --seed $seed \
      --batch-size 12 \
      --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
      --out results/epoch39_${arm}_s${seed}.json 2>&1 | tail -1
  done
done
echo ROUND11B_DONE
