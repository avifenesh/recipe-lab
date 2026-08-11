#!/bin/bash
# Round 19 — curriculum ORDER permutations (owner goal restated 2026-08-12:
# SOTA recipe for a hybrid-MoE small model; loop is dead; the campaign
# question is which stage ORDER has impact).
#
# Two new cells on the dense 1B factorial regime (data1b, AdamW, 5 epochs,
# seed 7) so the four existing cells are direct controls:
#   V_cur  .5051  math->phil->web      (best known)
#   V_std  .4870  proportional shuffle (no staging)
#   V_pm    ?     phil->math->web      (is math-FIRST load-bearing, or just
#                                       abstract-before-web?)
#   V_wl    ?     web->math->phil      (anti-curriculum but still staged:
#                                       separates "staging helps" from
#                                       "THIS order helps")
# Dense not MoE on purpose: r17 showed the backbone moves evals ~0.2pt at
# matched data while data/order moved 2-10x that, and dense runs cost 2.1x
# less. AdamW not Muon on purpose: comparability with the paid baselines.
#
# Pre-registered (evals decide, 8-task mean, 1000/task):
#   P1 abstract-first mechanism: pm within noise of cur (>= .50), both
#      clearly above std -> order within {math,phil} doesn't matter.
#   P2 math-first load-bearing: pm lands clearly below cur (>= 1pt) ->
#      math specifically must open the run.
#   P3 staging-vs-order: wl vs std is the sharp read. wl > std = staging
#      itself helps regardless of order; wl <= std = wrong-order staging
#      actively hurts (late-stage forgetting dominates).
#
# Usage (on the p5): bash run_round19.sh
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(nvidia-smi -L | wc -l)
mkdir -p results ckpts

for order in pm wl; do
  out=results/r19_V_${order}.json
  if [ ! -f "$out" ]; then
    torchrun --standalone --nproc_per_node=$NPROC train1b.py \
      --arm V --order $order --lr 6e-4 \
      --ckpt ckpts/r19_V_${order}.pt --out "$out"
  fi
  python eval1b.py --ckpt ckpts/r19_V_${order}_best.pt \
    --out evals_r19_V_${order}.json --limit 1000 2>&1 | tail -3
done
python - <<'EOF'
import json
ok = all(json.load(open(f"evals_r19_V_{o}.json")) for o in ("pm", "wl"))
EOF
touch ~/READY_FOR_TEARDOWN
echo ROUND19_DONE
