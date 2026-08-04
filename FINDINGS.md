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

8. **SSM inversion** (round 6, seed 7): looped+scaled SSM beats vanilla SSM
   on fresh data at matched FLOPs — MC 5.1035 < MB 5.1156 < MA 5.1374, 24%
   fewer params. First loop-beats-params result in the lab. ε gain 2.2× the
   attention-mixer gain at N=2. Needs seeds + trajectory (round 8) before
   the headline is trusted; N=4 inverted (MC4 > MB4), unexplained.

## Open (in flight)

- Round 3c: unscaled stability edge; 4000-step rankings at 1.2e-3.
- Round 7: 39-epoch H7 — does C−A flip negative?
- Round 8: SSM result seed-stability + fresh-data trajectory MA vs MC.
