#!/bin/bash
# Round 12 — third scale point + downstream evals (the last word).
#
# Trend so far (data-constrained best-val, loop minus vanilla):
#   d=384: +0.014 (vanilla)   d=768: -0.039 (loop, 3/3 seeds)
# Prediction P1: d=1024 gap more negative than -0.039 (monotone in scale).
# Prediction P2: downstream accuracy ordering matches val-loss ordering —
#   C1024 >= A1024 on the eval suite mean at each arm's best checkpoint.
# If P1 holds but P2 fails, the "loss win" is not an intelligence win and
# the recipe claim must be weakened accordingly. That is the point of evals.
#
# Protocol: identical to rounds 9/10 (12.5M unique tokens, 20K steps,
# batch 12 = 19.6 epochs). ~110 min/arm at d=1024, 4 arms ~7.5h, evals ~40m.
set -e
mkdir -p ckpts evals
for seed in 7 13; do
  for arm in A1024 C1024; do
    [ -f results/epoch39_${arm}_s${seed}.json ] && continue
    ~/venv/bin/python train.py --arm $arm --steps 20000 --seed $seed \
      --batch-size 12 \
      --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
      --save-ckpt ckpts/epoch39_${arm}_s${seed}.pt \
      --out results/epoch39_${arm}_s${seed}.json 2>&1 | tail -1
  done
done
for seed in 7 13; do
  for arm in A1024 C1024; do
    ~/venv/bin/python eval_downstream.py \
      --ckpt ckpts/epoch39_${arm}_s${seed}.pt \
      --tasks hellaswag,piqa,arc_easy,lambada --limit 1000 \
      --out evals/${arm}_s${seed}.json 2>&1 | tail -5
  done
done
echo ROUND12_DONE
