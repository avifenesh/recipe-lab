#!/bin/bash
# Round 20 — why did the anti-curriculum win, and is it real?
#
# r19 result: wl (web->math->phil) .5116 > pm .5080 > cur .5051 >> std .4870.
# Staging beats shuffling by 1.8-2.5pt (robust, 4 cells). The ordering among
# staged cells spans 0.65pt — inside plausible seed noise, so it is NOT yet
# recipe-grade. Two things are worth exactly one box:
#
#   1. MECHANISM. wl is the only staged cell whose epoch ENDS on an abstract
#      stage. Recency, not priority, is the live hypothesis. `fm`
#      (web->phil->MATH) also ends abstract, but on math instead of phil.
#   2. REPLICATION. wl on seed 13 — the no-single-seed rule that killed the
#      round-6 SSM headline applies to good news too.
#
# Cells (dense V 1.30B, AdamW, data1b, 5 epochs — same regime as r19):
#   V_fm  seed 7   web -> phil -> math
#   V_wl  seed 13  web -> math -> phil   (replication of the r19 winner)
#
# Pre-registered (evals decide, 8-task mean, 1000/task):
#   P1 RECENCY: fm >= .508 (i.e. lands with wl .5116 / above cur .5051)
#      -> "end each epoch on abstract data" is the mechanism, and it is
#      stage-identity-agnostic. Recipe gets a real rule.
#   P2 PHIL-LAST: fm lands with cur/pm (.505-.508) and clearly below wl
#      -> phil-specifically-last is the knob (consistent with T2's
#      phil-buys-reasoning), not abstract-last in general.
#   P3 REPLICATION: wl_s13 within ~0.5pt of wl_s7 (.5116) AND above cur
#      -> the ordering claim survives a second seed and enters the recipe.
#      If wl_s13 drops to cur's level, the r19 ordering was seed noise and
#      the recipe keeps only "staged >> shuffled" — which is already settled.
#   Null outcome named in advance: if fm ~ wl ~ cur within 0.5pt on both
#      seeds, stage order is a free parameter; say so and stop probing it.
#
# Usage (on the p5): bash run_round20.sh
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(nvidia-smi -L | wc -l)
mkdir -p results ckpts

run_cell () {   # $1=order  $2=seed  $3=tag
  local out=results/r20_V_$3.json
  if [ ! -f "$out" ]; then
    torchrun --standalone --nproc_per_node=$NPROC train1b.py \
      --arm V --order "$1" --seed "$2" --lr 6e-4 \
      --ckpt ckpts/r20_V_$3.pt --out "$out"
  fi
  python eval1b.py --ckpt ckpts/r20_V_$3_best.pt \
    --out evals_r20_V_$3.json --limit 1000 2>&1 | tail -3
}

run_cell fm 7  fm_s7
run_cell wl 13 wl_s13
python -c "
import json
for t in ('fm_s7', 'wl_s13'):
    json.load(open(f'evals_r20_V_{t}.json'))
print('both eval files valid')"
touch ~/READY_FOR_TEARDOWN
echo ROUND20_DONE
