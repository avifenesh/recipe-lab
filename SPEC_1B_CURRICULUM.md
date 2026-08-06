# 1B run, revised: curriculum × loop factorial (owner design, 2026-08-06)

Owner's insight: the 4-run budget buys TWO experiments if the runs form a
2×2 factorial — data ORDER × ARCHITECTURE — instead of 2 arms × 2 seeds.
Hypothesis (owner's): teaching logic/reasoning first (math → philosophy →
general knowledge) builds stronger circuits than shuffled data; and the
looped model, which spends more compute per token, should convert
logic-dense early tokens into structure better than vanilla — so curriculum
should help the loop MORE (interaction term).

## Arms (one seed each, seed 7)

| run | arch | data order |
|-----|------|-----------|
| V-std | vanilla 24 blocks d=2048 (~1.4B) | shuffled mix |
| V-cur | same | math → phil → web |
| L-std | loop 12×2 ε=1/(2√12) d=2048 (~0.9B) | shuffled mix |
| L-cur | same | math → phil → web |

Readouts:
- **Loop main effect** (avg L − avg V): replicates the flip at 1B — the
  original SPEC_1B claim, still tested.
- **Curriculum main effect** (avg cur − avg std): owner's ordering claim.
- **Interaction** (does curriculum help L more than V): the novel cell —
  nobody has published curriculum × weight-shared-recurrence.

Tradeoff accepted: 1 seed per cell instead of 2 (four cells > two cells
with seed replication at equal cost). Sub-1B rungs showed seed noise
±0.003-0.006, ~10× smaller than expected effects; any ambiguous cell gets
a seed-13 rerun before claims (budget reserve).

## Data (single tokenizer GPT-2 BPE, identical tokens across arms)

| stage | source | tokens |
|-------|--------|--------|
| 1 math | HuggingFaceTB/finemath (finemath-3plus, 34B avail) | 0.6B |
| 2 philosophy | gutenberg_english filtered philosophy/logic/ethics (fallback pg19 title filter) | 0.2B |
| 3 web | HuggingFaceFW/fineweb-edu sample-10BT | 2.2B |

Total 3.0B unique × 5 epochs = 15B tokens seen per run (SPEC_1B budget).

- **Curriculum arms**: epoch-major — each epoch replays stage1 → stage2 →
  stage3 (so late training still touches math; pure stage-sequential would
  catastrophically forget stage 1 — known failure mode).
- **Standard arms**: same three bins, sampled proportionally at random —
  identical token multiset, order the only difference.
- Val tracked separately on val_web (10M) AND val_math (5M) every 500
  steps: curriculum should show math-val advantage that persists, general-
  val parity or better.

## Architecture (modernized per SPEC_1B)

seq 4096, RoPE, RMSNorm, SwiGLU (ratio 8/3 rounded to 256), no biases,
untied embeddings dropped — tied head, GPT-2 BPE. bf16, DDP over 8 GPUs,
grad-accum to 2M tokens/step-group, AdamW (0.9/0.95, wd 0.1, clip 1.0),
warmup 2K, cosine to 10%. LR: quick probe on V-std at {6e-4, 3e-4} for 3K
steps → winner used for ALL arms (LR-transfer is part of the recipe claim).

Spot-safe: checkpoint every 1000 steps (model+opt+loader state), auto-resume
from latest on relaunch.

## Pre-registered predictions

- P1 (flip at 1B): mean(L) < mean(V) on best general-val. Bar: ≤ −0.02.
- P2 (curriculum main): mean(cur) ≤ mean(std) on general val AND
  math-val gap ≥ 0.05 in curriculum's favor. (Weak-form: curriculum not
  worse on general, better on math.)
- P3 (interaction, the novel claim): (L-cur − L-std) < (V-cur − V-std),
  i.e. curriculum helps the looped model more. Any margin counts if signs
  of P1 hold; report effect size regardless.
- P4 (downstream): lm-eval suite; L-cur is the predicted best cell overall;
  math tasks (gsm8k-style few-shot too hard at 1B — use arithmetic +
  math-qa subsets) favor cur arms.

Honest risk note: curriculum literature (2601.21698: 300B-token study)
finds ordering mainly changes phase timing, with effects shrinking at
scale, mostly at ≤160M params. Owner's bet is the loop interaction changes
that. If P2/P3 fail, the factorial still delivers P1 (the funded claim) at
full power — no wasted budget.

## Infra

p5.48xlarge (8×H100) spot us-east-2 (~$20.5/h). Per run ~11-14h wall →
4 runs + LR probe + evals ≈ 60-70h ≈ **$1.3-1.5K** total. p4d.24xlarge
($6.2/h, 8×A100-40G) fallback if p5 capacity dry (~2.5× wall).

Data prepped on the fpv box (same region), synced to the p5 via S3 or scp.
