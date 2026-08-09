#!/bin/bash
# Round 15 — does the loop flip survive the move to a hybrid MoE backbone?
#
# Everything the campaign verified (11/11 paired runs, three mixer families,
# zero sign flips) was on dense stacks. The 4B pilot the owner wants is a
# hybrid MoE, which changes two things at once: the feed-forward becomes
# sparse (so a looped block's shared weights now also share a router), and
# most of the stack becomes a fixed-state recurrent mixer. Neither was in the
# regime where the flip was measured. This rung buys the sign before any p5
# spend, on the free L40S.
#
# Arms (model_moe.py): identical application sequence [s,s,s,s,a] x3 = 15
# applications, attention placement matched exactly, HL stores half the
# linear blocks.
#   HV: 0.307B total / 0.134B active
#   HL: 0.197B total / 0.093B active   (-36% total, -31% active)
# Regime: the exact one where the dense flip verified 3/3 seeds — d=768,
# 12.5M unique tokens, batch 12 x 1024, 20K steps ~= 19.6 epochs, lr 6e-4
# (rounds 9/10, results/epoch39_*768_s*.json). Batch is served as
# micro-bs 6 x accum 2 because capacity_factor 2.0 OOMs at 12 on the 46GB
# L40S; same tokens per step, and it exercises the accumulating balancer.
#
# Pre-registered predictions:
#   P1: HL best-val < HV best-val, both seeds. This is the flip carried onto
#       the hybrid-MoE backbone. If it fails, the 4B pilot must drop either
#       the loop or the MoE — do NOT spend on both.
#   P2: gap magnitude >= 0.02 (the SPEC_1B bar the dense rungs cleared 2.4x).
#       A smaller-but-negative gap is a weak pass: report as such, don't
#       round it up.
#   P3: drop_frac stays < 0.01 after step 200 in both arms. A collapsed
#       router would make the comparison meaningless (measured: the balancer
#       reaches 0.000 by step 100).
#   P4: HL's degradation past best <= HV's. Every dense rung showed the
#       epoch-regularizer effect param-ordered; if it inverts here, the
#       sparse FFN is interacting with the loop and that needs its own probe.
# Any of P1/P3 failing cancels the MoE+loop combination for the pilot.
#
# Budget: ~570 ms/step at micro-bs 6 x accum 2, 20K steps ~= 3.2h/arm,
# 4 runs (2 arms x 2 seeds) ~= 13h on a box that is already paid for.
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results ckpts evals
PY=~/venv/bin/python
DATA=../lab/data12/fineweb_train.bin
VAL=../lab/data12/fineweb_val.bin

for seed in 7 13; do
  for arm in HV HL; do
    out=results/r15_${arm}_s${seed}.json
    [ -f "$out" ] && continue
    $PY train_moe.py --arm $arm --size s --seed $seed \
      --flat-data $DATA --flat-val $VAL \
      --steps 20000 --micro-bs 6 --grad-accum 2 \
      --warmup 500 --lr 6e-4 --eval-every 250 --ckpt-every 1000 \
      --ckpt ckpts/r15_${arm}_s${seed}.pt --out "$out" 2>&1 | tail -2
  done
done
echo ROUND15_DONE
