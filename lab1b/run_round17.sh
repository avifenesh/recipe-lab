#!/bin/bash
# Round 17 — the attribution run, owner-approved 2026-08-11: dense V on the
# cur curriculum with the UPGRADED philosophy pool (data1b_p). Identical to
# the factorial's V_cur cell in every respect except the data dir, so
# (V_cur_p − V_cur) isolates the phil-pool effect exactly:
#   V_cur   2.6969 loss / .5051 eval mean   (159M pre-1928 Gutenberg phil)
#   V_cur_p     ?                           (600M modern SEP+PhilPapers phil)
# Factorial invariants kept: LR 6e-4 (probe winner), warmup 2000, micro-bs 2,
# grad-accum 8, 28650 steps, seed default, eval-every 500 like train1b.
#
# Usage (on the p5): bash run_round17.sh
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(nvidia-smi -L | wc -l)
mkdir -p results ckpts

out=results/r17_V_cur_p.json
if [ ! -f "$out" ]; then
  torchrun --standalone --nproc_per_node=$NPROC train1b.py \
    --arm V --order cur --data-dir data1b_p --lr 6e-4 \
    --ckpt ckpts/r17_V_cur_p.pt --out "$out"
fi
python eval1b.py --ckpt ckpts/r17_V_cur_p_best.pt \
  --out evals_r17_V_cur_p.json --limit 1000 2>&1 | tail -3
python -c "import json; json.load(open('evals_r17_V_cur_p.json'))" \
  && touch ~/READY_FOR_TEARDOWN
echo ROUND17_DONE
