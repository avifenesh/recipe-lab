#!/bin/bash
# Round 9 — does the round-7 robustness result survive 4x scale?
#
# The scale-up candidate is the epoch-overfit claim. Before calling it worth
# real money, validate the param-ordering at d=768 (A768 ~155M params, C768
# ~113M; 4x the FLOPs of the d=384 arms). Batch 12 to fit 46GB, so 20K steps
# = 245M tokens = 19.6 epochs over the 12.5M-token bin — well past the
# ~13-epoch crossover round 7 observed, and bigger models memorize this bin
# faster, so the cliff has room to show.
#
# Pre-registered predictions (from round-7 d=384 numbers):
#   P1: both arms' best val beats their d=384 counterparts (more capacity).
#   P2: A768 degradation past best > C768 degradation, ratio >= 1.5x
#       (round 7 saw 2.1x at d=384; some shrink allowed, ordering must hold).
#   P3: A768 bottoms at an earlier epoch than d=384 A (4500 steps = 8.8
#       epochs there) — bigger models memorize 12.5M tokens faster.
# If P2 fails, the robustness recipe does NOT scale and FINDINGS.md gets a
# second retraction. Either way the campaign answer improves.
set -e
for arm in A768 C768; do
  ~/venv/bin/python train.py --arm $arm --steps 20000 --seed 7 \
    --batch-size 12 \
    --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
    --out results/epoch39_${arm}_s7.json 2>&1 | tail -1
done
echo ROUND9_DONE
