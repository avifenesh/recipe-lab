# Recipe hypothesis: scaled layer-loop pretraining

## Candidate sweep (2026-08-03)

Sources reviewed:
- **Loopie** (arXiv 2607.16051, IQuestLab): layer-loop recurrence (each layer applied
  R times before passing on) beats model-loop and beats compute-matched vanilla MoE
  across a 0.15B→1B scaling ladder. R=2 is the sweet spot. No special residual scaling
  reported — plain pre-LN + SmallInit + AdamW.
- **Residual scaling of looped transformers** (arXiv 2606.18524, Tsinghua/ByteDance):
  weight sharing makes per-step residuals *correlated*: ||Σ r_n||² = Θ(N²), not the
  Θ(N) random walk of independent layers. Bounded forward pass requires ε = 1/N per
  loop, and for L unique layers each looped N times, ε = λ/(N√L). Bonus: optimal LR
  becomes independent of N (hyperparameter transfer).
- **DeepLoop** (arXiv 2607.13491, Princeton): same conclusion from post-LN/DeepNorm side.
- Muon/SOAP (2607.20548): real but already proven at scale — not novel territory.
- SDPO (2601.20802): RL self-distillation, post-training on 8B — wrong scale for L40S from-zero work.
- TTT-E2E, curriculum learning: weaker/contested effects, or wrong fit for from-zero proof.

## The gap nobody has tested

Loopie proved **layer-loop** ordering wins but used **unscaled** residuals (ε=1).
The scaling paper proved **ε=λ/(N√L)** is required but analyzed **model-loop**-style reuse.

Our numerics (proof2) show layer-loop is *more* pathological than model-loop:
same-layer residual cosine ≈ 0.99 (layer-loop) vs 0.95 (model-loop), and unscaled
norm blowup is orders of magnitude worse (N=8: 2e9 vs 1e5). So the scaling fix
should matter *more* for layer-loop — yet the two have never been combined.

**Hypothesis H1**: layer-loop + ε=λ/(N√L) residual scaling beats
(a) unscaled layer-loop (Loopie as-published, at small scale),
(b) scaled model-loop, and
(c) a compute-matched vanilla transformer,
at equal pretraining FLOPs and tokens, from-zero.

**Hypothesis H2** (secondary): the correlated-gradient effect means the looped
layers see an effective ~N× gradient boost; ε-scaling also restores LR transfer,
so a single LR works across N (testable by LR sweep at N=1 vs N=2).

## Paper proofs (numerical, this repo)

- `proof1_accumulation.py` — Θ(N²) vs Θ(N) accumulation: **CONFIRMED**
  (shared/N grows linearly, shared/N² flat ~0.5-0.7; indep/N flat ~0.5).
- `proof2_layerloop_scaling.py` — layer-loop correlation > model-loop: **CONFIRMED**
  (cos 0.99 vs 0.95; ε=1/(N√L) bounds both schedules to R≈1.2).
- `proof3_preln.py` — pre-LN does NOT kill the effect: shared grows Θ(N)
  (3.5→6.8→13.6→26.7 as N doubles) vs indep Θ(√N) (2.5→3.6→5.2→7.2). Update
  amplification grows with N unscaled (0.013→0.018), flat when scaled (0.006).
  **CONFIRMED** — pathology survives the normalization used by every modern LM.

## Experiment design (rented L40S, 46GB)

From-zero GPT pretraining, compute-matched per token, same data/steps/seed.

| arm | architecture | blocks/token | residual ε |
|-----|-------------|--------------|-----------|
| A   | vanilla, L=12 stored | 12 | 1 |
| B   | layer-loop, L=6 stored, N=2 | 12 | 1 (Loopie style) |
| C   | layer-loop, L=6 stored, N=2 | 12 | λ/(N√L), learnable λ per layer |
| D   | model-loop, L=6 stored, N=2 | 12 | λ/(N√L) |

B/C/D have ~half the block parameters of A — the Loopie claim is that recurrence
converts equal compute into equal-or-better quality with fewer params. C vs B
isolates the scaling fix. C vs D isolates the ordering.

Config: d=384, 8 heads, seq 1024, GPT-2 BPE, FineWeb-Edu stream, bf16, AdamW,
warmup+cosine. Round 1: ~200M tokens/arm smoke → separation check → scale
winner to 1B tokens, 3 seeds.

Success bar: C < B and C < A on val loss at matched compute, stable across seeds,
gap not shrinking with training — then worth scaling.
