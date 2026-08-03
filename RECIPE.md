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
- `proof4_lr_transfer.py` — H2 LR transfer: **CONFIRMED**. Shared-weight net,
  LR grid {3e-3…3}, N∈{1,2,4,8}. Unscaled optimal LR drops 10× at N=8
  (diverges at LR≥1); ε=1/N keeps optimal LR pinned at the N=1 value for all N.
  GPU grid (round 3b) tests the same claim in the real transformer.

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

Bench actuals (L40S, batch 24×1024): A=137.0 ms/step 18.2GB; F384=135.3 ms
18.1GB; F448=154.1 (+12%); F512=172.4 (+26%). **Looping frees no wall-clock
at this scale** — the Loopie reinvestment step needs the MoE memory pressure
of 20B-scale training to produce headroom; a 40M dense model has none. The
wall-clock-matched F arm degenerates to C. F448/F512 runs would be
compute-UNmatched → dropped. H1(c) at toy dense scale is C vs A, already
answered: refuted on fresh data, pending on repeated data (round 5).

### Pre-registered round-3 predictions (fit before data, 2026-08-03)

Loss-law fit `L(D) = E + B·D^-β` per arm on the 98M-token smoke curves
(steps ≥750), extrapolated to 490M tokens:

| arm | E (irreducible) | β | pred val @490M | pred gap vs A |
|-----|------|------|------|------|
| A | 3.507 | 0.550 | 3.8765 | — |
| C | 3.464 | 0.506 | 3.9001 | +0.024 (was +0.046 @98M) |
| E3 | 3.872 | 0.641 | 4.1012 | +0.225 (grows) |

Fitted C's asymptote sits **below** A's → predicted C−A crossover ≈ **2.1B
tokens**. E3 never crosses (fit). If round 3 lands near these numbers, the
recipe's scale-up bet is C-style N=2 at multi-B tokens; if C's gap does not
shrink to ≈0.024, the fit (and the crossover story) is wrong. Caveat: smoke
curves carry a 4000-step cosine tail that inflates late slope; rough numbers.

Actuals: A @490M = **3.8486** (pred 3.8765, fit pessimistic 0.028 — cosine-tail
caveat confirmed; gap predictions matter more than absolute values).

### Round 3 verdict — crossover REFUTED on fresh data

C−A gap by tokens: +0.020 @49M → +0.044 @98M → +0.059 @246M → **+0.0755 @490M**.
E3−A: +0.050 @49M → **+0.1555 @490M** (grows faster, as the higher-N/lower-param
arm should). Both monotonically GROWING, not shrinking. The loss-law fit's below-A asymptote for
C was an artifact of the cosine tail. On fresh (single-epoch) data, stored
params beat recurrence at matched FLOPs, and the deficit widens with tokens —
consistent with Loopie's own MoE framing (their looped arms hold total params
high via experts; dense layer-looping halves capacity, full stop).

**H1(c) refuted at this scale for the fresh-data regime.** Per the decision
rule: pivot. Two live threads —
1. ε=1/N + LR-transfer package (rounds 2, 3b, 4): proven, publishable,
   applies to any looped arch (Ouro/Loopie-class).
2. **H5 (round 5): the data-constrained regime.** The frontier constraint is
   the data wall, not params. Repeating data ~10 epochs, memorization capacity
   (∝ params) matters less and compute-per-token more (Muennighoff 2023:
   >4 epochs, params saturate). Same 20K steps on a 50M-token subset:
   if (C−A)|repeated ≪ +0.0755 or negative, the recipe's home is
   data-constrained pretraining — which is where the field is heading.

   Independent support: ELT (2604.09168) finds looped transformers "exhibit
   robustness against overfitting in data-constrained regimes" — but for
   visual generation, never tested for LM pretraining, and never with ε
   scaling. The H5 combination stays novel.

## Queue on box (serial, 2026-08-03 evening)

1. **Round 3** — long A/C/E3, 20K steps / 490M tokens: crossover trajectory.
2. **Round 3b** — bench_steptime + LR grid {3e-4, 6e-4, 1.2e-3} × {A,B,C,E1,E3}
   at 2000 steps: (a) real-transformer H2 test, (b) kills the "ε is just a
   smaller effective LR" objection by comparing each arm at its own optimum.
3. **Round 4** — N-sweep I6/I12 (+unscaled controls) + learnable-λ H/H4 smokes:
   val-loss-vs-N curve at N ∈ {1,2,4,6,12}, fixed 12 applications/token.

### Round 3b results (LR grid, 2000 steps)

| arm | 3e-4 | 6e-4 | 1.2e-3 |
|-----|------|------|--------|
| A (N=1) | 5.3998 | 5.0222 | 4.7608 |
| B (N=2, ε=1) | 5.4289 | 5.0949 | **4.7588** |
| C (N=2, ε=1/N) | 5.4109 | 5.0836 | 4.7681 |
| E1 (N=4, ε=1) | 5.4786 | 5.1515 | 4.9175 |
| E3 (N=4, ε=1/N) | 5.4397 | 5.0623 | **4.7972** |

Findings:
1. **N=4: ε does forward-pass work LR cannot buy.** E3@opt beats E1@opt by
   0.120 — not an LR re-parameterization. At N=2 the effect inverts smally
   (+0.009, B edges C at 1.2e-3), consistent with N=2 correlation being weak
   enough for LR to absorb.
2. **All arms right-censored at 1.2e-3** — true optima higher; grid extends
   to 2.4e-3/4.8e-3 in round 3c. Proof 4 predicts B/E1 hit the stability edge
   first; if instead everything keeps improving equally, the smoke-scale LR
   (6e-4) was just too low and rankings must be re-checked at optimum (hi4k
   reruns queued).
3. **B ≈ A at 1.2e-3 (4.7588 vs 4.7608) at 2000 steps** — the fresh-data
   param-deficit story is LR-sensitive; short-horizon near-parity at higher
   LR means round-3's 6e-4 verdict is not the final word on H1(c).

### Round 4 results (N-sweep + learnable λ, 98M tokens, seed 7)

| N | stored | unscaled | ε=1/N | ε gain |
|---|--------|----------|-------|--------|
| 1 | 12 | 4.4439 | 4.4439 | — |
| 2 | 6 | 4.4949 | 4.4894 | +0.0055 |
| 4 | 3 | 4.5801 | 4.5465 | +0.0336 |
| 6 | 2 | 4.6779 | 4.6140 | +0.0639 |
| 12 | 1 | 4.8687 | 4.7341 | +0.1346 |

- **ε gain = 0.0122·(N−1), R²=0.993** — linear in N with the correct anchor
  (N=1: no loop, no correlation, zero gain). Beats gain∝N (R²=0.96),
  ∝log N (0.76), power law N^1.77 (0.95). Matches the theory: the correlated
  term the fix removes is the (N−1)-term cross-covariance sum. At N=12 the
  fix recovers 32% of the loop's deficit vs vanilla.
- Val loss still monotone in N on fresh data — param starvation dominates;
  no free lunch from recurrence alone at this scale (consistent with rd 3).
- Learnable λ: ±0.005, noise-level. Fixed λ=1 is fine; drop the H arms.

## Round-5 decision rule (pre-committed)

Let g(t) = C−A val-loss gap at token count t from round 3.

- **If g(490M) ≤ 0.024** (fit prediction) **and still shrinking**: fit updated
  crossover; if ≤ ~3B tokens, run the money shot — C vs A (+ F512 if
  bench_steptime shows headroom) at ~2.5B tokens (102K steps ≈ 4.2 h/arm),
  2 seeds. Crossover observed in-run = recipe proven, write it up.
- **If g flat/growing at 490M**: H1(c) refuted at this scale — recurrence
  does not beat params here even ε-corrected. Pivot: the proven deliverable
  becomes the ε=1/N + LR-transfer package for *existing* looped archs (Ouro,
  Loopie fine-tunes), validated by rounds 2/3b/4; no vanilla-beating claim.
- **Either way**: novelty of the combination is confirmed — 2606.18524 never
  tests layer-loop (zero mentions) nor a vanilla baseline; Loopie never scales
  ε. Rounds 2+3b+4 are publishable ablations regardless of crossover.
