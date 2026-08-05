# 1B-scale kill-or-confirm spec

Goal: decide whether the data-constrained loop flip is a production-relevant
training recipe or a small-scale curiosity. One experiment, sized to be
believed.

## The claim under test

At ~1B params with a realistic repetition ratio, layer-loop + ε=1/N beats a
FLOPs-matched vanilla model with ~35% more params on (a) best val loss and
(b) a standard zero-shot suite. Pre-registered: gap ≤ −0.02 nats and suite
mean ≥ +1 point, both seeds; either failing = recipe capped at sub-B scale,
publish as-is with that bound.

## Arms

| arm | shape | params (est) |
|-----|-------|--------------|
| V | vanilla, 24 blocks, d=2048, 16 heads | ~1.4B |
| L | layer-loop 12×2, d=2048, ε=1/(2√12) | ~0.9B |

Same 24 block-applications/token. 2 seeds each = 4 runs.
Optional 5th run if budget allows: L at N=3 (8 stored ×3, ε=1/(3√8)) — one
point on the N axis at scale.

## Data — the realism fix

- **3B unique tokens** (FineWeb-Edu or DCLM-baseline slice), trained for
  **~15B tokens seen ≈ 5 epochs**. This matches production repetition
  (4-16×), not the 20× of the small-scale runs. Params/unique-token ratio
  ~0.3-0.47 — inside the regime where the flip appeared (d=1024 was ~16;
  the flip trend says higher ratio = bigger flip, so 5 epochs is the
  CONSERVATIVE test. If it survives here it is real).
- Val: 20M held-out fresh tokens. Eval checkpoints every 500 steps near
  expected minima, every 2K elsewhere.

## Training config

- seq 4096 (production-like; also tests the recipe beyond seq-1024),
  GPT-NeoX or Llama-style: RoPE (drop learned pos-emb), RMSNorm, SwiGLU.
  Architecture modernization is deliberate — recipe must survive it.
- Global batch 2M tokens, AdamW (0.9/0.95, wd 0.1), warmup 2K steps,
  cosine to 10%. LR: tune V briefly at 1e-3/6e-4/3e-4 on a 5%-length run
  (~3 checkpoints), then USE THE SAME LR FOR L — that IS the LR-transfer
  claim, part of the test.
- bf16 + FSDP/ZeRO-2, activation checkpointing on looped blocks
  (recompute is cheap: weights shared, activations aren't).

## Evals

lm-eval-harness, zero-shot: hellaswag, piqa, arc_easy, arc_challenge,
lambada_openai, winogrande, boolq, sciq (full sets, not 1K subsets).
Report per-task + mean, both seeds, best-val checkpoint AND final
checkpoint (the final-vs-best gap measures the overfit-cliff claim at
scale).

## Compute + cost

Per run: ~15B tokens × ~1.4B params × 6 ≈ 1.3e23 FLOPs... realistic
throughput on 8×H100 ≈ 0.35-0.45 (looped arm slightly higher — fewer
params to all-gather): ~4.5-6 days per pair of runs if run serially.

| option | config | wall time (4 runs) | est cost |
|--------|--------|--------------------|----------|
| lean | 8×H100 node (Lambda/SF Compute, ~$2/GPU/h) | ~10-12 days serial | ~$4-5K |
| standard | 2× 8×H100 nodes | ~5-6 days | ~$4-5K (same GPU-hours) |
| shoestring | 8×A100-80G (~$1.3/GPU/h) | ~18 days | ~$4.5K |

Cheaper de-risk first (recommended): **0.4B rung** — d=1536, 20 blocks,
1.5B unique × 5 epochs, 2×A100 or keep the L40S ~4 days, ~$300-500. Same
pre-registration. If the flip misses at 0.4B/5-epochs, the 1B run is
cancelled and $4K saved; if it hits, run 1B with confidence.

## Schedule (standard option)

- Day 0: node up, data tokenized (script exists, scales linearly), V LR
  probe (~6h).
- Days 1-6: 4 runs (2 parallel per node).
- Day 6: lm-eval both arms both seeds (~4h).
- Day 7: verdict vs pre-registered bars, ledger update, PAPER.md v2.

## Failure modes + mitigations

- **Loop arm diverges at seq 4096 / RoPE / SwiGLU**: ε theory is
  architecture-agnostic (proof 3 covers pre-LN; RMSNorm equivalent), but if
  instability appears, first check ε includes the √L factor and gradient
  clip is on. Fall back to N=2→ε=0.5 exactly as validated.
- **No flip at 5 epochs**: try the planned-anyway 8-epoch continuation of
  the same checkpoints (data loader wraps; costs only extra steps on the
  same rental) before declaring failure — the flip is ratio-dependent and 5
  epochs is the conservative end.
- **Throughput imbalance** (L trains faster per step than V due to fewer
  params): report both tokens-matched (primary, pre-registered) and
  wall-clock-matched (secondary) comparisons.

## Deliverables

1. `results/1b/*.json` curves + `evals/1b/*.json` — same schema as repo.
2. PAPER.md v2 with the 1B section and the pre-registered verdict, either
   direction.
3. If confirmed: the scaled claim is "loop+ε beats params at 1B under
   5-epoch repetition" — recipe-paper grade.
