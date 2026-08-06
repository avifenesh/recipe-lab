#!/bin/bash
# The 2x2 factorial on the p5 (run ON the p5, inside tmux/nohup).
# Prereqs: repo at ~/recipe-lab, data1b/ bins present, venv with torch.
#
# 0. LR probe: V-std at 3e-4 vs 6e-4, 1500 steps (~35 min each). Winner
#    becomes THE LR for all four runs (LR-transfer is part of the claim).
# 1-4. Cells in order: V-std, L-std, V-cur, L-cur (std pair first — it
#    contains the funded P1 comparison; if spot dies early we still have it).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
NPROC=$(nvidia-smi -L | wc -l)
mkdir -p results ckpts

lr_probe() {
  for lr in 3e-4 6e-4; do
    [ -f results/probe_${lr}.json ] && continue
    torchrun --standalone --nproc_per_node=$NPROC train1b.py \
      --arm V --order std --steps 1500 --lr $lr --warmup 300 \
      --ckpt ckpts/probe_${lr}.pt --out results/probe_${lr}.json 2>&1 | tail -1
  done
  $PY - <<'EOF'
import json
v = {lr: json.load(open(f"results/probe_{lr}.json"))["curve"][-1]["val_web"]
     for lr in ("3e-4", "6e-4")}
best = min(v, key=v.get)
print(f"probe: {v} -> LR={best}")
open("results/chosen_lr.txt", "w").write(best)
EOF
}

[ -f results/chosen_lr.txt ] || lr_probe
LR=$(cat results/chosen_lr.txt)
echo "factorial LR=$LR"

for cell in "V std" "L std" "V cur" "L cur"; do
  set -- $cell
  arm=$1; order=$2
  [ -f results/${arm}_${order}.json ] && continue
  torchrun --standalone --nproc_per_node=$NPROC train1b.py \
    --arm $arm --order $order --lr $LR \
    --ckpt ckpts/${arm}_${order}.pt \
    --out results/${arm}_${order}.json 2>&1 | tail -2
done
echo FACTORIAL_DONE
