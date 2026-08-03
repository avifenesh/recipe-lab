#!/bin/bash
# Round 3b — H2 LR-transfer, done properly.
#
# Round 2's probe compared 2x-LR runs against 4000-step curves — wrong
# baseline (different cosine schedule position). Here every run is 2000 steps
# so schedules are identical and only LR varies. H2 (eps-scaling restores LR
# transfer) predicts: the eps=1/N arm's optimal LR matches vanilla's, while
# the unscaled arm's optimal LR shifts down as N grows. So at 2x LR the
# unscaled arms (B, E1) should degrade more (or gain less) than the scaled
# arms (C, E3), and the effect should be bigger at N=4 than N=2.
#
# Also: bench_steptime.py — measure whether loop-freed memory buys any width
# at this scale (Loopie recipe step iii).
set -e
~/venv/bin/python bench_steptime.py 2>&1
# 3-point LR grid. The discriminating question: is eps=1/N a genuinely better
# recipe, or just an LR re-parameterization? If the latter, B at its own
# optimal LR (likely lower than 6e-4, since unscaled looping amplifies
# updates) matches C at its optimum. If C's optimum beats B's optimum,
# eps does forward-pass work LR cannot.
for lr in 3e-4 6e-4 1.2e-3; do
  for arm in A B C E1 E3; do
    # hilr_B_s7 / hilr_C_s7 already exist from round 2 (same config)
    [ "$lr" = "1.2e-3" ] && { [ "$arm" = "B" ] || [ "$arm" = "C" ]; } && continue
    ~/venv/bin/python train.py --arm $arm --steps 2000 --seed 7 --lr $lr \
      --out results/lr2k_${arm}_${lr}.json 2>&1 | tail -1
  done
done
echo ROUND3B_DONE
