#!/bin/bash
# Round 16p — ONE run, owner-approved 2026-08-09: HV (hybrid-MoE, un-looped)
# on the winning curriculum with the UPGRADED philosophy pool (data1b_p:
# math 600M + phil 600M modern SEP/PhilPapers + web 1800M = 3.0B unique).
#
# Owner redirect: the campaign winner is the CURRICULUM, not the loop. This
# rung tests whether the curriculum carries to the field-standard backbone
# (Mamba-2 + GQA + fine-grained MoE) with philosophy finally unstarved.
#
# 5 epochs / 28650 steps on purpose — the exact regime of the dense 1B
# factorial, so the eval readout compares directly to V_cur (.5051 mean,
# starved 159M pre-1928 phil). More epochs was T1's proven wrong knob.
#
# Decision metric: evals (owner's rule). At 1B the suite has signal — the
# dense cells separated cleanly (V_cur .5051 vs V_std .4870).
#
# Usage (on the p5): bash run_round16p.sh
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(nvidia-smi -L | wc -l)
mkdir -p results ckpts

out=results/r16p_HV_cur.json
if [ ! -f "$out" ]; then
  torchrun --standalone --nproc_per_node=$NPROC train_moe.py \
    --arm HV --size b --order cur --data-dir data1b_p \
    --steps 28650 --lr 6e-4 --warmup 2000 --micro-bs 2 --grad-accum 8 \
    --eval-every 1000 --ckpt-every 2000 \
    --ckpt ckpts/r16p_HV_cur.pt --out "$out" 2>&1 | tail -2
fi
python eval_moe.py --ckpt ckpts/r16p_HV_cur_best.pt \
  --out evals_r16p_HV_cur.json --limit 1000 2>&1 | tail -3
python -c "import json; json.load(open('evals_r16p_HV_cur.json'))" \
  && touch ~/READY_FOR_TEARDOWN
echo ROUND16P_DONE
