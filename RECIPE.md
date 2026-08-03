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

## Results

### Round 1 — smoke, 4000 steps ≈ 98M tokens, seed 7

| arm | desc | params | final val |
|-----|------|--------|-----------|
| A | vanilla L12 | 40.6M | **4.4439** |
| B | layer-loop 6×2 ε=1 | 30.0M | 4.4949 |
| C | layer-loop 6×2 ε=1/N | 30.0M | 4.4894 |
| D | model-loop 6×2 ε=1/N | 30.0M | 4.5231 |

- C−B = −0.0055 (scaling fix wins), C−D = −0.0337 (layer-loop ordering wins,
  reproduces Loopie), C−A = +0.0455 (**vanilla still wins** — Loopie's headline
  does not reproduce at 30-40M params / 98M tokens).

### Round 2 — ε ladder at N=4, multi-seed, LR probe

N=4 ladder (3 stored blocks × 4 loops, same 12 applications/token), seed 7:

| ε | final val |
|---|-----------|
| 1 (E1) | 4.5801 |
| 1/√N = 0.5 (E2) | 4.5579 |
| 1/N = 0.25 (E3) | **4.5465** |

Monotone in exactly the order Thm 1 predicts. E1−E3 gap = 0.034 ≈ 6× the N=2
C−B gap — the ε effect grows with N as theory says. Token-equivalent lead:
ε=1/N reaches ε=1's final loss 545 steps (13.6% of run) earlier.

Multi-seed C−B at N=2: −0.0055 (s7), −0.0079 (s13), −0.0099 (s29).
**Sign stable 3/3 seeds, mean −0.0078.** The scaling fix is real.

### Verdict so far

H1 parts (a),(b) **CONFIRMED**: ε=1/N beats unscaled layer-loop (3 seeds) and
scaled model-loop. Effect grows with N. H1 part (c) **NOT confirmed**: vanilla
beats all looped arms at ~2-3 tokens/param — this regime is maximally
param-hungry, and looped arms have 25-50% fewer params. Ouro-class wins are
reported at ≫Chinchilla tokens/param.

**H3 (round 3)**: the A−C(−E3) gap closes and flips in the overtrained regime.
20K steps = 490M tokens → A at ~23 tok/param (1.2× Chinchilla), C at ~46 (2.3×),
E3 at ~92 (4.6×). If the gap is flat or growing at 490M tokens, H1(c) is
refuted at this scale and the recipe only pays as a param-compression trick.

### Loopie deep-read correction (2026-08-03)

Re-read 2607.16051 in full. Two facts change the H1(c) design:

1. **Loopie's crossover vs compute-matched vanilla happens at ~600B tokens**
   (20B-A2B model); vanilla leads before that. Layer-loop overtakes model-loop
   at ~1.2T tokens. Our 98M-token smoke could not have shown the win — round 1
   "refutation" of H1(c) was premature, regime was wrong, not the recipe.
2. **The Loopie Recipe is 3 steps, not 2**: (i) halve stored layers, (ii)
   layer-loop ×2, (iii) reinvest the freed memory headroom (2× microbatch, then
   extra capacity) until *measured step time* matches the reference. Compute
   matching is wall-clock, not analytical FLOPs. Arms B/C/D only do (i)+(ii) —
   they are the "parameter-saving device" the paper says loses under fixed
   compute. The honest H1(c) arm is:

| arm | architecture | matching |
|-----|-------------|----------|
| F | layer-loop 6×2, ε=1/N, d_model widened (448/512/576) | measured ms/step == arm A |

`bench_steptime.py` picks the width. H1(c) restated: **F < A at equal
wall-clock and tokens**, with ε=1/N doing work B-style unscaled F cannot.
