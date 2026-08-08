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

## The 2×2 (best val_web; final = last step)

| cell | params | best val_web | best val_math | status |
|------|--------|--------------|---------------|--------|
| V-std | 1.30B | 2.7062 | 1.6214 | DONE |
| V-cur | 1.30B | **2.6969** | 1.6317* | DONE |
| L-std | 0.70B | 2.7487 | 1.6601 | DONE |
| L-cur | 0.70B | pending (~2.73 proj) | pending | ~72%, ETA 2h |

*best-ckpt math val; curve minimum may differ slightly.

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

**P3 — interaction (curriculum helps loops more): pending L-cur.** At every
matched step so far L-cur leads L-std by ~0.02–0.035, vs the V pair's final
−0.0093 — trending toward CONFIRMED at ~2× the vanilla margin. Final
number decides.

**P4 — downstream ordering: pending** (8-task suite runs on each box after
training; V-cur/L-std evals in progress now).

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
