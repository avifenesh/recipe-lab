#!/bin/bash
# Round 14 — owner's four targets (2026-08-08):
#  T1 flip onset: V-std + L-std extended 5->8 epochs (28650 -> 45840 steps,
#     rewarmed LR over the extension, identical treatment both arms).
#  T2 philosophy effect: V-nophil (math->webx->web) vs V-cur.
#  T3 thinking stage: V-cur4 (math->phil->think->web) vs V-cur.
#  T4 loop x thinking: L-cur4 vs V-cur4 and vs L-cur.
# Box assignment (3 parallel p5 spots):
#  boxE: V-std-ext then L-std-ext (extensions, ~17h)
#  boxN: V-nophil then (nothing) (~13h)
#  boxT: V-cur4 then L-cur4 (~26h)  <- longest; consider a 4th box if wanted
# Usage on each box: bash run_round14.sh <boxE|boxN|boxT>
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(nvidia-smi -L | wc -l)
mkdir -p results ckpts
LR=6e-4

case "$1" in
boxE)
  for arm in V L; do
    [ -f results/${arm}_std_ext8.json ] && continue
    torchrun --standalone --nproc_per_node=$NPROC train1b.py \
      --arm $arm --order std --steps 45840 --lr $LR --warmup 1000 \
      --extend-from ckpts/${arm}_std.pt \
      --eval-every 1000 --ckpt-every 2000 \
      --ckpt ckpts/${arm}_std_ext8.pt \
      --out results/${arm}_std_ext8.json 2>&1 | tail -2
  done
  ;;
boxN)
  [ -f results/V_nophil.json ] || torchrun --standalone --nproc_per_node=$NPROC train1b.py \
    --arm V --order nophil --lr $LR \
    --eval-every 1000 --ckpt-every 2000 \
    --ckpt ckpts/V_nophil.pt --out results/V_nophil.json 2>&1 | tail -2
  ;;
boxT)
  for arm in V L; do
    [ -f results/${arm}_cur4.json ] && continue
    torchrun --standalone --nproc_per_node=$NPROC train1b.py \
      --arm $arm --order cur4 --lr $LR \
      --eval-every 1000 --ckpt-every 2000 \
      --ckpt ckpts/${arm}_cur4.pt --out results/${arm}_cur4.json 2>&1 | tail -2
  done
  ;;
*) echo "usage: $0 boxE|boxN|boxT"; exit 1;;
esac
echo ROUND14_$1_DONE
