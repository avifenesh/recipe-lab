# 1B factorial — results ledger (recording as they land)

Protocol: 1.30B vanilla (24 blocks d=2048) vs 0.70B looped (12×2, ε=0.5),
Llama-style (RoPE/RMSNorm/SwiGLU), seq 4096, GPT-2 BPE, LR 6e-4 (probe-chosen),
3.0B unique tokens × 5 epochs = 15B seen per run. Data: 600M FineMath +
150M Gutenberg-philosophy + 2.25B FineWeb-Edu — identical multiset all arms;
curriculum = epoch-major math→phil→web, std = proportional shuffle.
Val: 10M held-out web + 5M held-out math, every 500 steps (L_std: 1000).
Infra: p5.48xlarge spot fleet (8×H100 each), $20.5/h; one interruption
mid-L_std recovered via persistent volume + snapshot; queue reordered onto
3 parallel boxes.

Checkpoints: HF `Avifenesh/recipe-lab-1b` (public).

## The 2×2 — FINAL (best val_web)

| | std | cur | curriculum margin |
|--|-----|-----|-------------------|
| **V 1.30B** | 2.7062 | **2.6969** | −0.0093 |
| **L 0.70B** | 2.7487 | 2.7394 | −0.0093 |

Loop margin: +0.0425 (std), +0.0425 (cur). The factorial is exactly
additive — identical margins on both axes to the fourth decimal.

## Downstream — FINAL (zero-shot, 1000/task)

| task | V_std | V_cur | L_std | L_cur |
|------|-------|-------|-------|-------|
| hellaswag | .398 | .402 | .398 | .394 |
| piqa | .690 | **.706** | .690 | .681 |
| arc_easy | .517 | .523 | **.535** | .532 |
| arc_challenge | .304 | **.334** | .308 | .298 |
| winogrande | **.541** | .530 | .499 | .524 |
| boolq | .491 | .589 | **.615** | .605 |
| sciq | **.620** | .607 | .610 | .593 |
| lambada | .334 | **.350** | .324 | .325 |
| **MEAN** | .4870 | **.5051** | .4974 | .4939 |

- **V-cur is the best model of the four** — best loss AND best eval mean.
- Curriculum on vanilla: +1.8 eval points (.487→.505), driven by
  arc_challenge +3.0, boolq +9.8, lambada +1.6, piqa +1.6 — the
  reasoning/comprehension-flavored tasks. Fact-lookup (sciq) unchanged/−.
- Curriculum on loop: loss win did NOT carry to the suite mean
  (.4974→.4939, −0.35pt; mixed per-task). P4 holds for vanilla, not loop.
- Oddity worth noting: L_std beats V_std on the suite mean (.4974 vs .4870)
  despite worse loss — driven by boolq (.615 vs .491). Small-N noise on
  2-choice tasks likely; would need full sets to certify.

## Pre-registered verdicts

**P1 — loop flip at 1B (bar ≤ −0.02): REFUTED.** L-std − V-std = **+0.0425**.
Gap was +0.06 through step 20K, narrowed only to +0.043 in the decay; no
late crossover. Boundary explanation: this run is 5 epochs at 0.23–0.43
params/unique-token; vanilla never overfit (val still falling at end). The
sub-1B flip lived at ≥8 epochs / params-per-unique ≥ ~4 (5 rungs, 11/11
paired runs). Combined finding: **the loop advantage is a data-wall effect
with a mapped onset — absent at 5 epochs, decisive by 8+.** Both edges now
measured; that boundary is the publishable object.

**P2 — curriculum (owner's logic-first thesis): CONFIRMED on the vanilla
pair.** V-cur beat V-std by **−0.0093** best val_web — on *general web*
val, not just math. Notable against literature (2601.21698: curriculum
effects shrink with scale, mostly ≤160M): a clean ordering win at 1.3B.
Dynamics: curriculum dips behind mid-math-stage each epoch, exits ahead
after each cycle; lead present through the whole back half.

**P3 — interaction (curriculum helps loops more): REFUTED.** Curriculum
margin identical on both architectures (−0.0093 = −0.0093). The mid-run
L-cur lead over L-std's curve converged to the same final margin. Verdict:
curriculum and architecture are **independent, additive effects** — no
synergy, but perfect composability. (A recipe ingredient that transfers
unchanged across architectures is operationally valuable even without
interaction.)

**P4 — downstream ordering: SPLIT.** Vanilla pair: confirmed — V-cur beats
V-std on loss AND suite mean (+1.8pt, concentrated in reasoning tasks).
Loop pair: loss ordering did not carry to suite mean (−0.35pt, mixed).
Honest bound: 1000-sample subsets; per-task swings of ±2pt are within
noise on 2-choice tasks.

## Owner's thesis (recorded 2026-08-08)

"Thinking is the math; philosophy is the best language-form of thinking and
intelligence; on top of that add knowledge. I would also add thinking
training." → next experiment queued: **cur4 = math → philosophy →
reasoning-traces → web** head-to-head vs cur3 (today's winner), corpora
verified available (OpenThoughts-114k, NaturalReasoning), sign-check at
small scale before any 1B spend.

## Incidents log (ops lessons, all recovered)

1. Spot interruption killed b0 mid-L_std @step 10000 — persistent volume +
   snapshot preserved everything; resumed on a fresh box from step 10001.
2. Cancelling a persistent spot request **terminates** its stopped instance
   — volume non-deletion + snapshot made it a non-event. Rule: pull data
   before cancel; cancel before terminate on live boxes.
3. Missing `tiktoken`/`datasets` in the DLAMI venv crashed the on-box eval
   watchers — the evals-json teardown gate correctly held both boxes
   instead of tearing down eval-less. Gate design validated in production.
4. ISP IP rotation twice broke SSH mid-run (SG allowlist) — not a training
   event; boxes unaffected.

## Cost

~13.4h/run × 4 runs + probe + evals across 4 spot instances ≈ **$1.15-1.3K**
(vs $4-5K original spec estimate; parallel fleet added ~$50 overlap).
