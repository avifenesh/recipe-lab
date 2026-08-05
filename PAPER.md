# Looping Beats Parameters Under the Data Wall

*Draft v1 — 2026-08-05. Numbers, receipts, and scripts: this repo
(`FINDINGS.md`, `RECIPE.md`, `lab/`, `results/`, `evals/`).*

## Abstract

Scaling doctrine says that under a fixed compute budget, buying parameters
beats reusing them: an N-times-looped (weight-tied) transformer loses to a
vanilla model of N× the depth-parameters trained on the same FLOPs. We
confirm this decisively on fresh, single-epoch data — and show it **inverts
in the data-constrained regime** the field is now entering. When a fixed
set of unique tokens is repeated for multiple epochs, a layer-looped model
with ε=1/N residual scaling reaches *better* validation loss than a
FLOPs-matched vanilla model carrying 26–37% more parameters, and the margin
**grows with model scale** (gap at d=384: +0.014 against looping; d=768:
−0.039; d=1024: −0.060). The result replicates across three architecture
families — attention, selective-SSM (Mamba-style), and hybrid — in 9/9
paired training runs with zero sign flips, and carries to downstream task
accuracy (4-task zero-shot suite, 2/2 seeds at d=1024; LAMBADA accuracy
nearly doubles). Mechanism: parameters buy memorization speed, which is the
asset on fresh data and the liability on repeated data; looping spends the
same FLOPs on per-token computation instead. Alongside the main result we
contribute a practical training package for looped models: ε=λ/(N√L)
residual scaling (gain over unscaled fits 0.0122·(N−1), R²=0.99),
learning-rate transfer from the unlooped model, a hybrid design rule
(*loop the state-mixer, never the retriever*), and a test-time rule for
loop extrapolation (keep the training ε).

## 1. Setup

All experiments: from-zero GPT pretraining, GPT-2 BPE, FineWeb-Edu,
sequence 1024, AdamW (0.9/0.95, wd 0.1, clip 1.0), warmup+cosine, bf16, one
L40S. **Every compared pair executes the same number of block applications
per token** (12), so FLOPs per token are matched; the looped arm stores
fewer unique blocks and applies each N times consecutively (layer-loop),
with every residual branch scaled by ε.

Arms referenced below (d=model width; params exclude tied positional/embed
differences):

| arm | shape | ε |
|-----|-------|---|
| A | vanilla, 12 stored blocks | 1 |
| B | layer-loop 6×2 | 1 |
| C | layer-loop 6×2 | 1/2 |
| E1/E3 | layer-loop 3×4 | 1 / 1/4 |
| A768/C768, A1024/C1024 | as A/C at d=768, d=1024 | |
| MA/MC | as A/C with selective-SSM mixer | |
| HA/HE | hybrid: 2 attention + 10 SSM applications | |

## 2. The training package for looped models (theory-first results)

The residual-scaling theory (arXiv 2606.18524) says weight sharing makes
per-step residuals correlated — ‖Σr‖² = Θ(N²) instead of the Θ(N) random
walk of independent layers — so bounded dynamics require ε=1/N per loop,
ε=λ/(N√L) for L unique layers looped N times. We verified the accumulation
law, its survival under pre-LN, and its *stronger* form under layer-loop
ordering numerically before training (proofs 1–3 in repo), then confirmed
four GPU-scale predictions:

1. **ε ordering.** Val loss is monotone in ε exactly as predicted
   (N=4, 98M tokens: 4.580 at ε=1 > 4.558 at 1/√N > 4.547 at 1/N).
2. **Gain law.** The ε-scaling gain over unscaled looping fits
   **0.0122·(N−1)** with R²=0.993 across N∈{2,4,6,12} — the removed term is
   the (N−1)-count cross-covariance sum, and the measured gain counts it.
3. **Not an LR re-parameterization.** On a 5-point LR grid all arms peak at
   the same LR (2.4e-3), and at each arm's own optimum the scaled arm still
   wins at both N (N=2: −0.030; N=4: −0.128). Past-optimum damage is
   ε-ordered: scaled-loop +0.018 ≪ vanilla +0.190 < unscaled-loop +0.281 —
   scaled looping has a *wider* LR basin than vanilla, unscaled a narrower
   one. Hence **LR transfer**: tune once at N=1, reuse at any N.
4. **Mixer-agnostic.** The same correlated-accumulation blowup appears with
   a selective-SSM residual branch (proof 5), which retro-explains the
   looped-Mamba training instabilities reported in the literature; ε=1/N
   removes it there too.

## 3. Fresh data: the doctrine holds (negative result, high confidence)

On never-repeated data the looped arms lose to vanilla at matched FLOPs at
every N, both mixers, all LRs tested, and the gap **grows** with tokens
(C−A: +0.020 at 49M tokens → +0.076 at 490M; E3−A: +0.050 → +0.156). The
published looped-model crossovers (Loopie: ~600B tokens, 20B MoE) sit 2–3
orders of magnitude above this testbed, and Loopie's width-reinvestment
step is inert at dense small scale (looping freed no measurable step time:
135.3 vs 137.0 ms/step). Below multi-B-token scale on fresh data, buy
parameters.

## 4. The flip: repeated data, and it grows with scale

Protocol: 12.5M unique FineWeb-Edu tokens, 20K steps (batch 12×1024 ≈ 245M
tokens seen ≈ 19.6 epochs), best validation loss over the run (early-stop
metric), val set held out and fresh.

**Scale trend (attention family, loop − vanilla):**

| d_model | params (loop vs vanilla) | gap | seeds |
|---------|--------------------------|-----|-------|
| 384 | 30M vs 41M | **+0.014** (vanilla wins) | 1 |
| 768 | 81M vs 124M | **−0.039** | 3/3, σ≈0.003 |
| 1024 | 127M vs 203M | **−0.060 / −0.039** | 2/2 |

**Architecture generality at d=768 (2 seeds each):** pure SSM −0.026/−0.013
(74M vs 110M params); hybrid −0.018/−0.048 (83M vs 112M). Combined with
attention: **9/9 paired runs, zero sign flips, three families.**

**Overfit robustness, same protocol:** past its val minimum the vanilla arm
degrades 1.8–3.8× faster than the looped arm (e.g. d=768: +1.47 vs +0.82;
d=384 at 39 epochs: +1.07 vs +0.52 vs +0.28 for N=2/N=4). Degradation
ordering follows parameter count everywhere tested.

**Mechanism.** More parameters memorize the fixed corpus faster (larger
models hit their val minimum at *earlier* epochs: 5.9 vs 8.8 at d=768 vs
384), then collapse harder. The looped model spends identical FLOPs
refining each token twice through shared weights instead of storing the
data — weight sharing is an implicit regularizer whose value converts from
"robustness" to "outright best-val win" as the params-per-unique-token
ratio grows. This is exactly the ratio the field is pushing as unique
high-quality tokens run out.

## 5. Downstream: the loss win is not an artifact

Best checkpoints at d=1024 evaluated zero-shot (length-normalized MC
logprob; 1000 samples/task):

| | HellaSwag | PIQA | ARC-Easy | LAMBADA | mean |
|--|-----------|------|----------|---------|------|
| vanilla (2-seed avg) | .295 | .523 | .303 | .018 | .285 |
| looped (2-seed avg) | .300 | .536 | .318 | **.034** | **.297** |

The looped model wins the suite mean on both seeds (+0.003, +0.022).
LAMBADA — exact final-word prediction from long context, the task most
sensitive to per-token computation — nearly doubles. Absolute numbers are
small-model-small-data; the certified claim is *ordering*, and ordering
agrees with loss on both seeds.

## 6. Hybrids: loop the state-mixer, never the retriever

Looping the attention blocks of a hybrid fails badly regardless of
schedule: layer-loop +0.277 and placement-matched model-loop +0.255 against
the vanilla hybrid. Looping **only the SSM blocks** while applying each
attention block once (same 2-attention + 10-SSM applications per token)
wins 2/2 seeds (−0.018/−0.048) with 26% fewer parameters. Shared-weight
reuse helps state-mixing blocks and hurts retrieval blocks.

A mechanistic probe (MQAR, multi-query associative recall) explains the
division of labor: looping cannot recover information the SSM state never
held (state-starved recall fails at any loop count), but *multiplies* what
the state does hold — one looped block (×4, ε=1/4) reaches 59.8% recall
where four independent blocks reach 23.8%, at a quarter of the parameters.
Loops buy computation over state, not memory. Corollary for serving: a
looped SSM adds zero inference-memory per loop (one fixed-size state; no
per-application KV cache), so this is the only variant of the recipe that
also pays at inference. Test-time loop extrapolation works if ε is kept at
its **training** value (trained N=4: 61.7%; evaluated N=8: 49.6% with
frozen ε vs 27.8% readjusted).

## 7. What we retracted (protocol notes)

Every round pre-registered predictions before data; two single-seed
headlines were killed by our own multi-seed bar and are documented in the
ledger: (i) a round-6 "looped SSM beats vanilla on fresh data" ordering
that sign-flipped across seeds, (ii) an initial hybrid-flip design whose
failure was a placement artifact before it was a looping result. The
surviving claims all cleared ≥2 seeds, and the headline cleared 3 seeds, 3
architectures, and 3 scale points.

## 8. Limitations

Scale ceiling 203M params / 12.5M unique tokens; repetition ratio (~20
epochs) is more extreme than production (4–16); single dataset (FineWeb-Edu),
single tokenizer, custom minimal SSM (conv + selective scan, not full
Mamba); val loss + 4 small zero-shot tasks, no instruction-tuned evals; LR
schedule not re-tuned per arm at the largest scale. The 1B specification
(`SPEC_1B.md`) is written to remove exactly these.

## 9. Recipe card

For data-constrained pretraining (tokens repeat ≳4 epochs):

1. Halve (or better) stored depth; loop each layer N× consecutively.
2. Scale every shared residual branch by **ε = λ/(N√L)**, λ=1.
3. Keep the LR tuned at N=1. It transfers; the scaled model also tolerates
   2× overshoot that damages vanilla.
4. Hybrids: loop only SSM/state blocks; apply attention blocks once.
5. Serving: looped-SSM loops are inference-memory-free; extrapolate loop
   count at test time with the training ε.
6. Do **not** use this to beat vanilla on fresh single-epoch data below
   multi-B-token scale — it loses there, reliably.
