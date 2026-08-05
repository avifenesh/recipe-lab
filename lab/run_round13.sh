#!/bin/bash
# Round 13 — the 0.4B de-risk rung (gate for SPEC_1B).
#
# Two things at once:
#  (1) 4th scale point: d=1536 — A1536 ~417M vs C1536 ~247M params.
#  (2) Realism axis: 100M unique tokens x 8 epochs seen. Prior rungs used
#      12.5M x ~20 epochs (params/unique ~16); here the ratio is ~4.2 and
#      epochs are production-like. This is the conservative direction: less
#      repetition = less memorization pressure = smaller flip expected.
#
# Pre-registered:
#   P1: C1536 best-val < A1536 best-val, both seeds (flip survives realism).
#   P2: gap magnitude may SHRINK vs d=1024's -0.060/-0.039 (weaker
#       repetition), but must stay <= -0.02 to justify the 1B run.
#   P3: downstream suite mean ordering matches loss ordering (both seeds).
#   Any of P1-P3 failing -> SPEC_1B cancelled; publish with the sub-0.4B /
#   high-repetition bound. That outcome saves ~$4K and is a fine result.
#
# Budget: batch 12 x 1024, 65K steps = 800M tokens = 8 epochs over 100M.
# ~660ms/step at d=1536 -> ~12h/arm, 4 arms ~2 days. Checkpoints saved for
# eval (evals run in this same script; piqa parquet fix already in).
set -e
mkdir -p data100 ckpts evals
[ -f data100/fineweb_train.bin ] || \
  head -c 200000000 data500/fineweb_train.bin > data100/fineweb_train.bin
[ -f data100/fineweb_val.bin ] || cp data500/fineweb_val.bin data100/fineweb_val.bin
for seed in 7 13; do
  for arm in A1536 C1536; do
    [ -f results/derisk_${arm}_s${seed}.json ] && continue
    ~/venv/bin/python train.py --arm $arm --steps 65000 --seed $seed \
      --batch-size 12 --eval-every 500 \
      --data data100/fineweb_train.bin --val-data data100/fineweb_val.bin \
      --save-ckpt ckpts/derisk_${arm}_s${seed}.pt \
      --out results/derisk_${arm}_s${seed}.json 2>&1 | tail -1
  done
done
for seed in 7 13; do
  for arm in A1536 C1536; do
    ~/venv/bin/python eval_downstream.py \
      --ckpt ckpts/derisk_${arm}_s${seed}.pt \
      --tasks hellaswag,piqa,arc_easy,lambada --limit 1000 \
      --out evals/derisk_${arm}_s${seed}.json 2>&1 | tail -5
  done
done
echo ROUND13_DONE
