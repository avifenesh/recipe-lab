#!/bin/bash
# Round 16 — the 1B hybrid-MoE rung: two variants, clean picture.
#
# Gated on round 15 (P1 must pass on the L40S sign-check) and on round 14's
# T1 answer. Do not launch this before both are in.
#
# Arms: HV vs HL at size "b" on the owner's winning curriculum (cur:
# math -> philosophy -> web), the order that won the dense 1B factorial
# (V_cur 2.6969, best of the four cells).
#   HV/b: 1.316B total / 0.412B active   (total matches dense V's 1.30B)
#   HL/b: 0.803B total / 0.260B active   (-39% total, -37% active)
# Both apply 25 blocks per token with attention placement matched, so the
# comparison is at matched active compute and the only variable is whether
# the linear blocks are weight-shared.
#
# EPOCH COUNT IS THE LOAD-BEARING CHOICE. The dense 1B factorial ran 5
# epochs and the loop LOST there (L_cur 2.7394 vs V_cur 2.6969); the flip
# boundary needs a high params-per-unique ratio, which at this scale means
# >= 8 epochs. Running this rung at 5 epochs would reproduce a known loss and
# teach nothing. Set STEPS from round 14's T1 result:
#   - L_std_ext8 < V_std_ext8 (2.6743)  -> flip onset is <= 8 epochs;
#     use STEPS=45840 (8 epochs).
#   - L_std_ext8 still behind          -> 8 epochs is not enough at 1B dense;
#     either raise epochs here or shrink the unique pool. Decide from the
#     measured gap trend, do not just double the spend.
#
# Usage: STEPS=45840 bash run_round16.sh <boxA|boxB>
#   boxA: HV   boxB: HL   (one p5 each, run in parallel)
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(nvidia-smi -L | wc -l)
mkdir -p results ckpts
: "${STEPS:?set STEPS from round 14 T1 (45840 = 8 epochs)}"
LR=6e-4

case "$1" in
boxA) ARM=HV ;;
boxB) ARM=HL ;;
*) echo "usage: STEPS=45840 $0 boxA|boxB"; exit 1;;
esac

out=results/r16_${ARM}_cur.json
if [ ! -f "$out" ]; then
  torchrun --standalone --nproc_per_node=$NPROC train_moe.py \
    --arm $ARM --size b --order cur --steps "$STEPS" --lr $LR \
    --warmup 2000 --micro-bs 2 --grad-accum 8 \
    --eval-every 1000 --ckpt-every 2000 \
    --ckpt ckpts/r16_${ARM}_cur.pt --out "$out" 2>&1 | tail -2
fi
# evals are the decision metric, not loss (owner's rule)
python eval_moe.py --ckpt ckpts/r16_${ARM}_cur_best.pt \
  --out evals_r16_${ARM}_cur.json --limit 1000 2>&1 | tail -3
python -c "import json; json.load(open('evals_r16_${ARM}_cur.json'))" \
  && touch ~/READY_FOR_TEARDOWN
echo ROUND16_$1_DONE
