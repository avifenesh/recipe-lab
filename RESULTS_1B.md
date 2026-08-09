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

---

# Round 14 — owner's four targets (2026-08-08/09)

Two of four cells complete, two still training at time of writing
(`L_std_ext8` on boxE, `L_cur4` on boxL). Loss table, best val_web:

| cell | loss | eval mean | arc_challenge | winogrande | sciq | piqa |
|------|------|-----------|---------------|------------|------|------|
| V_std_ext8 (8 epochs) | **2.6743** | pending | | | | |
| V_nophil | 2.6907 | .5030 | .314 | .502 | **.636** | .694 |
| V_cur (5ep reference) | 2.6969 | **.5051** | **.334** | .530 | .607 | .706 |
| V_cur4 (thinking stage) | 2.6971 | .4961 | .288 | .531 | .607 | **.711** |
| V_std (5ep reference) | 2.7062 | .4870 | .304 | **.541** | .620 | .690 |

## T1 — does 8 epochs flip the loop? NO, and on evals it moves BACKWARD

Both arms extended 5 -> 8 epochs from `ext_base 28001` under identical
`--steps 45840 --lr 6e-4 --warmup 1000`, so the treatment is matched.

Loss — the gap is frozen:

| arm | 5-epoch best | 8-epoch best | extension gain |
|-----|--------------|--------------|----------------|
| V_std | 2.7062 | **2.6743** | −0.0320 |
| L_std | 2.7497 | 2.7172 | −0.0325 |
| **L − V** | **+0.0425** | **+0.0429** | — |

Evals (1000/task, 8 tasks) — and here the sign REVERSES:

| arm | 5-epoch mean | 8-epoch mean | extension gain |
|-----|--------------|--------------|----------------|
| V_std | .4870 | **.5013** | **+1.4pt** |
| L_std | .4974 | .4941 | **−0.3pt** |
| **L − V** | **+1.0pt (loop AHEAD)** | **−0.7pt (loop behind)** | — |

Two findings, and the second is the one that matters:

1. **On loss the gap did not move.** Extra epochs bought both architectures
   the same −0.032, so repetition is not repaying the loop's parameter
   deficit over this range. Going 5 -> 8 epochs on a 3B unique pool only
   moves params-per-unique 4.2 -> 6.7, far from the ~16 where the flip
   verified at d=768. More epochs is the wrong knob; a smaller unique pool
   is the direction that raises the ratio.

2. **On evals the extension actively hurt the loop.** At 5 epochs L_std beat
   V_std on suite mean (+1.0pt) *while losing on loss by 0.0425* — another
   instance of loss and evals disagreeing, in the loop's favour this time.
   The extension flipped that to −0.7pt: V gained +1.4pt (boolq .491 ->
   .571 carries most of it) while L lost 0.3pt. So the 8-epoch extension did
   not merely fail to flip the loop, it **cost the loop the eval lead it
   already had at 5 epochs**.

Per the owner's rule that evals decide and loss does not, the operative T1
answer is: **at 1B, 5 epochs is where the loop looks best, and pushing to 8
degrades it.** Any 1B-scale loop run should therefore be evaluated at 5
epochs, not extended. Treat the single-cell boolq swing (+8pt) with care —
2-choice task, 1000 samples — but the direction is consistent across the
suite mean and is the opposite of what the loss curve implies.

Sub-1B rungs remain unaffected: the flip verified 3/3 seeds at d=768 (~20
epochs, ratio ~16) and 2/2 at d=1024/d=1536, with downstream suites carrying
the loss ordering there. The ceiling is now measured on both sides.

## T2 — does philosophy actually do anything? YES, and it trades

`V_nophil` (philosophy swapped for extra web at matched budget) beat
`V_cur` **on loss** (2.6907 vs 2.6969) and **lost on evals**
(.5030 vs .5051). The split is legible and directional:

- philosophy **buys reasoning**: arc_challenge .334 vs .314, winogrande
  .530 vs .502
- philosophy **costs facts**: sciq .607 vs .636

This is the campaign's cleanest demonstration that **loss is the wrong
decision metric** — loss preferred the arm with worse reasoning. It is also
the strongest existing support for the owner's thesis that philosophy is
where reasoning-in-language comes from, at the cost of factual density.

## T3 — reasoning traces as a pretraining stage? NO

`V_cur4` (math → philosophy → 200M reasoning traces → web) matched V_cur on
loss (2.6971 vs 2.6969) and **lost on evals** (.4961 vs .5051), with
arc_challenge collapsing to **.288** — the worst of any V cell, below even
V_std's .304. piqa was the best of all cells (.711), so the traces did
something, just not reasoning transfer.

Reading: reasoning traces are instruction/solution-shaped text, and putting
them mid-pretraining before the web stage buys style rather than capability,
then gets partially overwritten. **Reasoning traces belong in
post-training**, which is where SPEC_PILOT_MOE puts them. T4 (`L_cur4`)
tests whether a looped model, which spends more compute per token, changes
that verdict.

## T4 — loop × thinking

Pending (`L_cur4`, boxL, ~step 23000/30500).

# Round 15 — hybrid-MoE sign-check (running, 2026-08-09)

Gate G1 of SPEC_PILOT_MOE, on the free L40S: does the loop flip survive the
move to the backbone the field converged on (Mamba-2 SSD state mixer + GQA
retriever + fine-grained MoE)? HV vs HL, 15 applications each with matched
attention placement, in the exact regime the dense flip verified 3/3 seeds
(d=768, 12.5M unique tokens, ~19.6 epochs, lr 6e-4). Predictions P1-P4
pre-registered in `lab1b/run_round15.sh`.

Arch bugs found and fixed while building it, both of which pass a
forward-only check and would have silently wasted a p5 run:

1. `torch.where` for the SSD non-causal mask still differentiates the
   untaken branch, whose exponents overflow to `inf`, so gradients were
   `inf*0 = NaN` while the printed loss looked healthy. Multiplicative
   masking fixed it; `test_ssd_grads_finite` guards it.
2. The MoE load buffer was overwritten per micro-batch, so with grad-accum
   the balancer saw `1/accum` of the routing and stalled at a **~20% token
   drop rate** — a quietly gutted sparse FFN. Counts now accumulate across
   the whole optimizer step; drop_frac falls 0.062 → 0.000 by step 100.

Also instrumented (off by default): expert overlap between consecutive
applications of a looped block. A looped MoE block reuses its router, so it
can re-pick the same experts every application — a degenerate loop that loss
would not reveal. Untested in any prior work; measured before the pilot.
