#!/bin/bash
# Round 18 — does the loop+ε flip survive Muon? (free L40S, ~$0)
#
# Muon (torch.optim.Muon, Newton-Schulz orthogonalized momentum) is the
# field's AdamW replacement (Moonshot ~2× compute efficiency, Kimi K2/K3,
# NVIDIA Megatron report 2607.20548). The pilot would use it for the cost
# cut — but ε=1/N rescales residual-branch updates and Muon renormalizes
# update spectra, so they touch the SAME training dynamics. The flip must
# be re-verified under Muon, not assumed.
#
# Design: exact round-9/10 flip rung (A768 vs C768, d=768, 12.5M unique
# ×19.6 epochs, 20000 steps, batch 12, seeds 7/13) with --opt muon.
# AdamW baselines already on disk (epoch39_*_s{7,13}.json):
#   s7:  A768 4.6831  C768 4.6438  (gap -0.0393)
#   s13: A768 ~4.69   C768 ~4.66   (gap -0.0356)
# Muon lr: 2e-2 with match_rms_adamw on matrices, AdamW 6e-4 elsewhere.
# G0 (lr sanity, run FIRST): 2000-step probe at muon-lr {1e-2, 2e-2, 4e-2}
# on A768 only; pick best, use for all four main runs. A mis-tuned lr for
# the baseline arm would fake a flip win.
#
# Pre-registered:
#   P1 (flip survives): C768_muon best-val < A768_muon best-val, BOTH seeds,
#       gap <= -0.02. Any sign flip = Muon breaks the recipe, pilot keeps
#       AdamW or drops ε (decide then, on the per-seed pattern).
#   P2 (Muon helps at all): A768_muon < A768_adamw (4.6831/s7) — the 2×
#       claim should show as a clearly lower best-val at matched steps.
#       If Muon does NOT beat AdamW here, this rung says nothing about the
#       pilot cost cut and P1 is moot for spend decisions.
#   P3 (interaction direction, exploratory): |muon gap| vs |adamw gap|
#       (-0.039/-0.036). Bigger = ε and orthogonalization compose; smaller
#       but negative = partial overlap; ~zero = Muon subsumes ε — each of
#       these redirects the pilot differently, none kills the flip per P1.
#
# Usage (on fpv): bash run_round18.sh
set -e
cd "$(dirname "$0")"
mkdir -p results

# G0: lr probe, A768 only
for mlr in 1e-2 2e-2 4e-2; do
  out=results/r18_probe_A768_mlr${mlr}.json
  [ -f "$out" ] && continue
  ~/venv/bin/python train.py --arm A768 --opt muon --muon-lr $mlr \
    --steps 2000 --seed 7 --batch-size 12 \
    --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
    --out "$out" 2>&1 | tail -1
done
# the first grid (1e-2/2e-2/4e-2) was left-censored — monotone toward
# smaller lr, and 5e-3 then beat its best by 1.16 nats. Selection reads
# every probe on disk so grid extensions are picked up automatically.
MLR=$(~/venv/bin/python - <<'EOF'
import glob, json, re, sys
v = {}
for f in glob.glob("results/r18_probe_A768_mlr*.json"):
    m = re.search(r"mlr(.+)\.json", f).group(1)
    v[m] = json.load(open(f))["curve"][-1]["val_loss"]
best = min(v, key=v.get)
print(best)
print(f"probe: {v} -> muon-lr={best}", file=sys.stderr)
EOF
)
echo "round18 muon-lr=$MLR"

for seed in 7 13; do
  for arm in A768 C768; do
    out=results/r18_${arm}_muon_s${seed}.json
    [ -f "$out" ] && continue
    ~/venv/bin/python train.py --arm $arm --opt muon --muon-lr $MLR \
      --steps 20000 --seed $seed --batch-size 12 \
      --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
      --out "$out" 2>&1 | tail -1
  done
done
echo ROUND18_DONE
