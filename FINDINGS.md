# Findings: ε=1/N layer-loop scaling — what is proven, what is refuted

Status: 2026-08-04. Rounds 1-4 complete; rounds 5 (data-constrained), 6 (SSM
mixer), 3c (LR-grid extension) in flight. All runs: GPT-2-BPE FineWeb-Edu,
d=384, 12 block-applications/token (FLOPs-matched), bf16, AdamW, L40S.

## Proven (paper math + from-zero training)

1. **Correlated-accumulation pathology** (proofs 1-3, CONFIRMED numerically):
   weight-shared residual blocks accumulate ||Σr||² = Θ(N²) vs Θ(N) for
   independent layers; survives pre-LN; *stronger* under layer-loop ordering
   (cos ≈0.99) than model-loop (≈0.95). ε=1/N (per unique-layer-count:
   λ/(N√L)) restores bounded forward/update dynamics.

2. **ε=1/N ordering, on GPU** (rounds 1-2): val loss monotone in ε exactly as
   Thm-1 order predicts (N=4: 4.5801 ε=1 > 4.5579 ε=1/√N > 4.5465 ε=1/N).
   C<B stable across 3 seeds at N=2 (mean −0.0078).

3. **Gain law** (round 4): ε-scaling gain vs unscaled = **0.0122·(N−1)**,
   R²=0.993 across N∈{2,4,6,12} — the removed term is the (N−1)-count
   cross-covariance sum, and the empirical gain counts it. At N=12 the fix
   recovers 32% of the loop deficit.

4. **Not an LR re-parameterization** (round 3b): at N=4, scaled-at-optimum
   beats unscaled-at-optimum by 0.120 on a 3-point LR grid. At N=2 LR can
   absorb the difference (+0.009) — the fix matters where correlation is
   strong. Caveat: grid right-censored at 1.2e-3; 3c extends.

5. **LR transfer** (proof 4, CONFIRMED on CPU numerics): unscaled optimal LR
   falls ~10× by N=8 (diverges at moderate LR); ε=1/N pins the optimum at its
   N=1 value. GPU grid so far consistent (no divergence anywhere yet — need
   3c's higher rungs to see the unscaled edge).

6. **Mixer-agnostic** (proof 5, CONFIRMED numerically): identical pathology
   with a selective-SSM (Mamba-style) residual branch — shared-unscaled norm
   grows linearly in N while ε=1/N stays flat. Explains reported looped-Mamba
   training instabilities (LT2). GPU test = round 6.

## Refuted

1. **H1(c) — looped-beats-vanilla at matched FLOPs, fresh data, 30-40M
   params**: REFUTED. C−A gap grows monotonically +0.020→+0.0755 over
   49→490M tokens (E3−A: +0.050→+0.156). Stored params dominate when every
   token is new. Loopie's crossover (~600B tokens, 20B MoE) is out of reach
   of this testbed by ~3 orders of magnitude.

2. **Loopie width-reinvestment at small dense scale**: REFUTED as a lever
   here. Looping frees no step time at 40M params (F384 135.3 ms vs A
   137.0 ms) — nothing to reinvest; the recipe's third step requires
   MoE-scale memory pressure.

3. **Learnable λ**: noise (±0.005 vs fixed λ=1). Simplicity wins.

## The recipe, as of now

For any looped/weight-tied architecture (Ouro-, Loopie-, looped-Mamba-class):

- **Scale every shared residual branch by ε = λ/(N√L)** (λ=1 fine); gain over
  unscaled grows as 0.012·(N−1) in val loss at 98M tokens — worth ~0.13 at
  N=12, larger the deeper you loop.
- **Keep the LR you tuned at N=1**; ε-scaling makes it transfer across N
  (proof 4 + grid evidence pending 3c).
- **Applies to SSM mixers too** (proof 5; GPU pending round 6).
- Do NOT expect looping to beat a param-matched-FLOPs vanilla model on fresh
  data below multi-B-token scale; the case for looping is param efficiency,
  data-constrained training (round 5 pending), or reasoning-depth effects not
  measured by val loss.

## New since round 4

7. **Data-constrained direction** (round 5): repeat tax at 9.8 epochs is
   monotone in param count (A +0.1326 > C +0.1017 > E3 +0.0974) and the C−A
   gap falls with tokens on repeated data (+0.053@10K→+0.045@20K) while
   rising on fresh. Loop's disadvantage is regime-dependent, as H5 claimed.
   No flip at 9.8 epochs; round 7 pushes to 39.

7b. **Epoch-overfit robustness** (round 7, 39 epochs): the overfit cliff is
   param-ordered — degradation past best val: A +1.069, C +0.520, E3 +0.283.
   C−A final −0.535, crossover ≈13 epochs. Early-stop best still A by +0.014,
   so the claim is *robustness*, not peak quality: loop+ε is a strong
   epoch-regularizer wherever token budget exceeds the early-stop point.

7c. **LR-grid capstone** (round 3c): all arms' optima at 2.4e-3; at optimum
   the scaled arm beats unscaled at BOTH N (N=2 −0.030, N=4 −0.128); past-
   optimum damage is ε-ordered (C +0.018 ≪ E3 +0.092 ≪ A +0.190 < B +0.243 <
   E1 +0.281) — scaled looping has a wider LR basin than vanilla, unscaled a
   narrower one. GPU-scale H2 confirmation. A@opt < C@opt: fresh-data verdict
   not a tuning artifact.

8. **SSM inversion: RETRACTED** (round 8 seed check). The round-6 seed-7
   ordering (MC < MB < MA) does not survive seeds 13/29 — MC−MA sign flips
   (−0.034/+0.048/−0.041), every arm wins one seed. SSM arms are 5-10×
   noisier than attention arms here. Weak residual signal: MC best 3-seed
   mean, never last. 20K-token-horizon MA-vs-MC single-seed trajectory
   pending as a longer lever on the same question.

## Open (in flight)

- Round 3c: unscaled stability edge; 4000-step rankings at 1.2e-3.
- Round 7: 39-epoch H7 — does C−A flip negative?
- Round 8: SSM result seed-stability + fresh-data trajectory MA vs MC.
