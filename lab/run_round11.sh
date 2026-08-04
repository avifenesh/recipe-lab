#!/bin/bash
# Round 11 — the Mamba push: does the d=768 data-constrained flip hold for
# hybrid (2 attn + 10 ssm apps/token) and pure-SSM backbones?
#
# Why it matters beyond replication: looped attention shares weights but NOT
# inference memory (12 applications = 12 KV caches). A looped SSM/hybrid keeps
# O(d_state*d) state regardless of loop count — the only variant of this
# recipe with an inference-memory win. eps=1/N is what makes looped SSMs
# trainable (proof 5 / LT2 instabilities), so the pieces need each other.
#
# Mixer upgraded since round 6/8: depthwise causal conv (k=4) before
# selection, as in real Mamba — round-6 conv-less mixer was ~0.7 behind
# attention and seed-noisy.
#
# Protocol: identical to rounds 9/10 (12.5M unique tokens, 20K steps,
# batch 12 = 19.6 epochs, d=768). Seeds 7 and 13 now; seed 29 only if 2/2
# sign-agree (no-single-seed rule).
#
# Pre-registered predictions:
#   P1: HC768 best-val < HA768 best-val (flip holds for hybrid), both seeds.
#   P2: MC768 best-val < MA768 best-val (flip holds for pure SSM), both seeds.
#   P3: degradation past best: HA > HC and MA > MC (param-ordered cliff).
#   P4: hybrid arms beat pure-SSM arms at matched schedule (attention's
#       2/12 share is worth more than its FLOPs share).
set -e
for seed in 7 13; do
  for arm in HA768 HC768 MA768 MC768; do
    ~/venv/bin/python train.py --arm $arm --steps 20000 --seed $seed \
      --batch-size 12 \
      --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
      --out results/epoch39_${arm}_s${seed}.json 2>&1 | tail -1
  done
done
echo ROUND11_DONE
