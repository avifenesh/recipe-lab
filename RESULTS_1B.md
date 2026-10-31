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
5. **boxL host went `impaired` at ~03:06 on 2026-08-09, mid-L_cur4 @step
   28000/30500 (92% done).** Not a spot reclaim — the request stayed
   `active/fulfilled` — and not the ISP-rotation failure mode either: the
   operator IP was still allowlisted and the SG unchanged. Both instance and
   system status checks read `impaired` with port 22 dead, while console
   output showed a clean 21:16 boot and nothing since, i.e. the host severed
   the instance rather than the guest crashing. Diagnosis order that mattered:
   check spot-request state and own public IP *before* touching the box, since
   the two cheap explanations look identical from a timed-out SSH.
   Recovery: root volume had `DeleteOnTermination=False`, so a stop/start
   migrates to fresh hardware with the checkpoint intact. Took a snapshot
   (`snap-049f3e04c344609ed`) first as insurance independent of the
   instance's fate. The forced stop took ~16 min (expected for a wedged
   host). `start-instances` first failed with `IncorrectSpotRequestState`
   because a persistent request parks as `disabled /
   instance-stopped-by-user` after an operator stop; it succeeded on retry
   once the request settled. Box came back on a new IP (3.16.54.79),
   **resumed from step 28001 with zero steps lost**. The on-box eval watcher
   died with the host and had to be relaunched by hand — worth noting, since
   the teardown gate depends on it and a silently dead watcher means a box
   that trains, finishes, and then idles at ~$20/h.
   Rule added: pause the teardown watcher during any recovery on a persistent
   spot request, or it can reap a box that is mid-migration.

## Artifact status — two gaps, one of them a real loss (2026-08-09)

Audited after the fleet came down, and the first item is a mistake worth
naming rather than burying:

**`V_std_best.pt` (5-epoch baseline weights) is gone from both local disk and
HF.** It lived on the rescued volume that boxE mounted; when that box was torn
down I pulled `evals_V_std.json` off it but never the weights. An earlier
audit in this session reported it present because the glob `V_std*_best.pt`
matched `V_std_ext8_best.pt` — a check that confirmed the wrong file.

What survives for that cell: the full 8-task eval suite (mean .4870), the
complete 57-point 5-epoch curve (best 2.7063, preserved inside
`V_std_ext8.json` because the extension inherited the log), and the 8-epoch
continuation weights. The 8-epoch weights cannot regenerate the 5-epoch ones —
the extension resumed from step 28001 and overwrote forward.

Recovery is staged and waiting on one approval. Snapshot
`snap-07ebb7d1de3fc90dd` (`completed`, "recipe-lab-1b: V_std done, L_std
@10000") has been restored to **`vol-02807be986503cd20`** (`available`, 400GB,
us-east-2a, tagged `recipe-lab-V_std-recovery-20260809`) — the same AZ as the
fpv box, so no new instance is needed.

Remaining step, which needs the owner to name the target instance: attach that
volume to fpv (`i-0dfdd059edcc3dec0`) and copy `ckpts/V_std_best.pt` off it.
Attaching was deliberately not done unattended — fpv is mid-round-15 and is
the machine currently producing the G1 evidence, so a device-mapping surprise
there would cost more than the file is worth. Attaching to a fresh t3 instead
is equally fine and avoids touching fpv at all.

    aws ec2 attach-volume --region us-east-2 \
      --volume-id vol-02807be986503cd20 --instance-id <target> --device /dev/sdp
    # then on the box: mount the partition read-only, copy
    # recipe-lab/lab1b/ckpts/V_std_best.pt, unmount, detach, delete the volume

Note the restored volume costs ~$32/mo while it sits, so it should be deleted
once the file is copied. **This is also why the orphaned-volume
cleanup must not be run blind** — `vol-0535ab38485db1901` was in the batch I
attempted to delete, and deleting it would have made this permanent.

**Only 3 of 9 best-checkpoints are on HF** (`V_cur`, `L_cur`, `L_std`).
Missing: `V_nophil`, `V_cur4`, `L_cur4`, `V_std_ext8`, `L_std_ext8` (all
present locally and verified as intact archives) plus `V_std` (see above).
Those five are currently single-copy on one machine. Upload is blocked on
credentials, not effort: the token used for the original uploads lived in the
environment of a p5 box that has since been torn down, and neither the local
machine nor the fpv box has one cached. Needs the owner to supply a token.

Lesson recorded: pull *weights* off a rescue volume at the same time as the
JSONs, and verify artifact presence with exact filenames rather than globs
that a sibling file can satisfy.

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

## T4 — loop × thinking: the FIRST real interaction in the campaign

`L_cur4` finished at loss 2.7405, eval mean **.4998**. Both cur4 arms ran the
same 28650 steps, so the pair is matched.

On loss, nothing happens — thinking is free and the loop's deficit is
untouched:

| | loss | vs its cur3 arm |
|---|---|---|
| V_cur / V_cur4 | 2.6969 / 2.6971 | +0.0002 |
| L_cur / L_cur4 | 2.7394 / 2.7405 | +0.0011 |
| **L − V** | **+0.0425 → +0.0434** | gap unchanged |

On evals, the sign splits by architecture:

| | eval mean | vs its cur3 arm |
|---|---|---|
| V_cur → V_cur4 | .5051 → .4960 | **−0.91pt (hurts)** |
| L_cur → L_cur4 | .4939 → .4998 | **+0.59pt (helps)** |
| L − V | −1.12pt → **+0.38pt** | **loop overtakes** |

**This is the interaction P3 looked for and did not find in the original
factorial.** There, curriculum was perfectly additive: −0.0093 on both
architectures. Here the reasoning-trace stage moves the two architectures in
*opposite directions*, and the loop pair ends up ahead of the vanilla pair on
the metric that decides — the only cell in the campaign where a looped 0.70B
model beats a 1.30B vanilla model on eval mean.

Mechanically this is what the owner predicted when he asked for T4: a looped
model spends more compute per token, and reasoning traces are the data type
where extra per-token computation should pay. Where the traces land:
arc_challenge .298 → .308 and lambada .325 → .350 for the loop, against
arc_challenge .334 → .288 and lambada .350 → .319 for vanilla. Vanilla's
collapse on exactly the tasks the loop gains on is the cleanest form this
result could take.

Caveats kept honest: single seed per cell, 1000 samples/task, and the total
margin (+0.38pt) is inside the ±2pt per-task noise this suite showed earlier
— it is the *direction and the mirrored per-task pattern* across four cells
that carries the claim, not the magnitude. The right confirmation is a seed-13
replication of `L_cur4` and `V_cur4`, which is ~$180 of p5 time and is the
cheapest high-value run left on the board.

Combined with T3: reasoning traces failed as a pretraining stage **for
vanilla** (which is what T3 measured and reported) but not universally — for
a weight-shared looped model they help. The SPEC_PILOT_MOE recommendation to
move traces into post-training therefore holds for a dense/vanilla base and
needs revisiting if the pilot ships the looped arm.

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

## Round 15 VERDICT — G1 PASSES. The flip survives the hybrid-MoE backbone.

All four runs complete (HV/HL × seeds 7/13, 20000 steps each, d=768, 12.5M
unique tokens ≈19.6 epochs, lr 6e-4 — the exact regime the dense flip verified
3/3 seeds in, so the backbone is the only changed variable).

| arm | best val | @step | final | degradation | max drop_frac |
|-----|----------|-------|-------|-------------|---------------|
| HV_s7 | 4.7231 | 3000 | 7.4653 | +2.742 | 0.00048 |
| HL_s7 | **4.6891** | 3250 | 7.6436 | +2.955 | 0.00117 |
| HV_s13 | 4.7334 | 2750 | 7.4696 | +2.736 | 0.00169 |
| HL_s13 | **4.6962** | 3250 | 7.6198 | +2.924 | 0.00244 |

HV: 0.307B total / 0.134B active. HL: 0.197B / 0.093B — **35.8% fewer total,
30.4% fewer active**, at 15 matched applications with matched attention
placement.

**P1 (HL < HV, both seeds): PASS.** Gaps **−0.0340** (s7) and **−0.0372** (s13).
Zero sign flips, extending the campaign to 13/13 paired runs since d=768.

**P2 (gap ≤ −0.02, both seeds): PASS.** Clears the SPEC_1B bar by 1.7-1.9×.
The margin is not seed noise: the two HV bests differ by only 0.0103 and the two
HL bests by 0.0071, so the effect is 3.3-3.6× the observed seed spread.

**P3 (drop_frac < 0.01): PASS.** Max 0.00244 across all four runs, 4-20× under
threshold. No routing collapse, so the comparison reflects architecture rather
than a silently gutted sparse FFN.

**P4 (HL degradation ≤ HV): FAILS as written, on both seeds — and the metric is
wrong, identically on both.** HL "degrades" more (+2.955/+2.924 vs
+2.742/+2.736), but the endpoint statistic contradicts the entire curve:

| step | s7 HL−HV | s13 HL−HV |
|------|----------|-----------|
| 1000 | +0.022 | +0.029 |
| 3000 | −0.033 | −0.043 |
| 4000 | −0.118 | −0.110 |
| 6000 | −0.181 | −0.221 |
| 10000 | **−0.235** | **−0.257** |
| 15000 | −0.036 | −0.066 |
| 19750 | +0.170 | +0.140 |

HL is ahead at **all 49 measured points from 3000 to 15000 on both seeds**
(mean −0.173 and −0.183 — five times the best-val gap), and does not cross back
until step **16000/16250 of 20000**, i.e. past ~31 of 39 epochs. Its larger
degradation number is an artifact of measuring distance from a lower floor to a
worse endpoint that no production run would reach.

Restated on horizons a real run would actually see, P4 is a wash rather than a
loss: degradation at 2× best-step is HL +0.449 vs HV +0.481 on seed 7 (HL
better) and HL +0.490 vs HV +0.419 on seed 13 (HV better) — the two seeds
disagree, so there is no effect at that horizon in either direction.

Honest conclusion: the dense rungs' epoch-regularizer effect (rounds 7/9/11,
where degradation past best was cleanly param-ordered) **does not reproduce on
this backbone as an endpoint statistic**, and reproduces as neither-arm-wins at
a sane horizon. What does reproduce, larger than anywhere in the dense
campaign, is the loop staying better for longer under repetition. Future rungs
should pre-register P4 as *mean gap over the post-best region* or *degradation
at 2× best-step*, never distance to a fixed terminal step.

**G1: PASS** on P1/P2/P3 — the three predictions that gate the pilot — with P4
failed as written and explained above. The loop flip therefore holds on the
backbone the field converged on (Mamba-2 SSD state mixer + GQA retriever +
fine-grained MoE with shared expert and aux-loss-free balancing), not just on
dense stacks.

Cost: **$0** — ran entirely on the already-paid-for L40S, ~11.7h wall.

### What G1 does and does not clear

Cleared: the loop × hybrid-MoE combination is not broken, so the pilot does not
have to drop one of the two.

**Not cleared — round 16 is still gated, and G1 does not change the arithmetic
that matters.** Two independent reasons to hold:

1. **Scale.** G1 is at d=768 with params-per-unique ≈16. Round 14's T1 measured
   the loop *losing* at 1B with that ratio at 4.2-6.7, and the gap frozen across
   5→8 epochs. G1 says the backbone is fine; it says nothing about the 1B
   regime, which is exactly where the dense flip died. `run_round16.sh` still
   refuses to run without an explicit STEPS decision for this reason.
2. **Evals, not loss.** Every number above is validation loss. T1 and T2 both
   showed loss and evals disagreeing — T1's 8-epoch extension *reversed* sign on
   evals, and T2's V_nophil won on loss while losing on evals. Per the owner's
   rule, G1's loss-only PASS is suggestive, not decisive. Running the eval suite
   on the four r15 best-checkpoints is free (L40S, ~1h) and should happen before
   any 1B spend.

Recommendation: run the r15 evals first (free), and treat the T1 ratio problem
as the real gate on round 16. Owner approval still required for any spend.

### Eval-readout decision, pre-registered before seeing the comparison

The r15 eval suite is running on all four best-checkpoints (free, L40S, ~30 min
each). The first result (HV_s7, mean **.3484**) exposes a measurement problem
that has to be settled *before* the arms are compared, so it is recorded here
first:

| task | HV_s7 | chance | above chance |
|------|-------|--------|--------------|
| sciq | .351 | .25 | +.101 |
| arc_easy | .328 | .25 | +.078 |
| hellaswag | .292 | .25 | +.042 |
| lambada | .031 | .00 | +.031 |
| piqa | .530 | .50 | +.030 |
| boolq | .517 | .50 | +.017 |
| arc_challenge | .247 | .25 | **−.003 (at chance)** |
| winogrande | .491 | .50 | **−.009 (at chance)** |

This is the expected consequence of the regime, not a bug: these are 0.31B and
0.20B models trained on 12.5M unique tokens, deliberately data-starved to create
the high params-per-unique ratio the flip needs. They have almost no world
knowledge, so a suite built for 1B-scale models sits near its floor.

**arc_challenge and winogrande are at chance and therefore contribute only
noise to the suite mean.** Including them would let a ±2pt coin-flip on two dead
tasks swamp a real signal on the five live ones. So the r15 eval readout will be
reported on the **5 tasks with signal above chance** (sciq, arc_easy, hellaswag,
lambada, piqa), with the full 8-task mean also shown for continuity but not used
as the decision statistic.

Stating this now, from the first checkpoint only, so the choice cannot be fitted
to the HL-vs-HV outcome once all four land. boolq (+.017) is a marginal call and
is kept in the 8-task figure but excluded from the 5-task readout.

Note also that a near-floor suite is *weak evidence either way* at this scale.
If the 5-task readout is ambiguous, the correct conclusion is "evals cannot
resolve G1 at d=768/12.5M", not "evals refute the loss result" — the honest
follow-up then is the 1B rung's own evals, not a louder claim from this one.

### Eval result — that ambiguous case is exactly what happened

All four suites complete (1000 samples/task, free, L40S).

| arm | sciq | arc_easy | hellaswag | lambada | piqa | 5-task | 8-task |
|-----|------|----------|-----------|---------|------|--------|--------|
| HV_s7 | .351 | .328 | .292 | .031 | .530 | .3064 | .3484 |
| HL_s7 | .358 | .305 | .301 | .033 | .541 | **.3077** | .3278 |
| HV_s13 | .340 | .307 | .288 | .033 | .536 | .3008 | .3200 |
| HL_s13 | .341 | .305 | .283 | .023 | .523 | .2951 | .3364 |

**5-task decision readout: the seeds disagree and the effect is inside the
noise.** HL is +0.12pt on seed 7 and −0.57pt on seed 13, mean −0.23pt. The
same-arm seed spread is 0.0056 (HV) and 0.0126 (HL) — *larger than the
between-arm difference*. There is no eval signal here in either direction.

**The 8-task mean flips sign between seeds** (−2.06pt then +1.64pt), and the
cause is entirely boolq, which is bimodal at this scale: .517 / .375 / .376 /
.495 across the four runs, i.e. the models are essentially answering all-yes or
all-no on a 2-choice task. That is precisely the artifact the pre-registered
5-task readout was written to exclude, and its behaviour here vindicates
excluding it rather than being a reason to revisit the choice.

**Verdict: evals cannot resolve G1 at d=768 / 12.5M unique tokens.** This is the
outcome named in advance, so it is reported as such and not spun. Concretely:

- The loss result stands on its own terms — −0.0340 and −0.0372, 2/2 seeds,
  3.3-3.6× the loss seed spread, routers healthy.
- The evals neither confirm nor contradict it. They are near their floor (3 of 8
  tasks at chance, lambada at .03) because a 0.20-0.31B model trained on 12.5M
  unique tokens has almost no world knowledge to test. The suite lacks the
  resolution for a 0.03-nat loss difference.
- **G1's status is therefore: PASS on loss, UNRESOLVED on evals.** Under the
  owner's rule that evals decide, that is *not* a full pass, and it should not be
  presented as one.

What this changes for the pilot: G1 still does its job of ruling out the
catastrophic case — loop × hybrid-MoE is not broken, so the pilot need not drop
either half. It does not deliver positive eval evidence, and the only place that
evidence can come from is a rung big enough for the suite to have signal. That
is the 1B rung, which is also where T1 says the ratio problem lives.

Recommendation unchanged and now better grounded: **round 16 stays gated.** If it
runs, it must (a) fix the params-per-unique ratio rather than repeat T1's 5→8
epoch mistake, and (b) be judged on its own evals, which at 1B do have signal —
the dense 1B cells separated cleanly (V_cur .5051 vs V_std .4870).

Also instrumented (off by default): expert overlap between consecutive
applications of a looped block. A looped MoE block reuses its router, so it
can re-pick the same experts every application — a degenerate loop that loss
would not reveal. Untested in any prior work; measured before the pilot.

# Data upgrade — philosophy pool rebuilt (2026-08-09)

Owner redirect: the campaign's real winner is the curriculum (math →
philosophy → world knowledge → agentic/thinking post-training), not the
loop; the next rung compares curriculum-on-MoE, not architectures. The 1B
factorial's philosophy stage was the starved one — it wanted 0.2B tokens
against a 159M pool of pre-1928 Gutenberg. Rebuilt per SPEC_PILOT_MOE
(`lab1b/prepare_phil_pool.py`, built on fpv in ~25 min, $0):

| bin | source | tokens |
|-----|--------|--------|
| `phil_sep.bin` | SEP Dec-2023 scrape (dmarx), stubs <2000 chars dropped | 44.0M |
| `phil_papers.bin` | PhilPapers open-access 2023-10-28 (malteos), English rows >5000 chars (~75% of rows are en) | 691.9M |
| `phil_arxiv.bin` | proof-pile-2 arXiv joined on math.LO/cs.LO ids from the arXiv metadata snapshot (both old/new id spellings) | 600.0M (capped) |

Total **1.336B modern philosophy/logic tokens** — 8.4× the old pool, all
post-Gutenberg, kept as separate bins so stage mixes stay tunable. Same
tokenizer/dtype contract as the 1B bins (GPT-2 BPE, uint16, EOT-joined).
On fpv at `lab1b/data_phil/`, manifest checked against file sizes.

Also located (stale note corrected): the darklanes SFT corpus IS generated —
`~/projects/sft-traces`, 2,000 verified agentic traces (500 each of bug-fix /
refactor / test-writing / explain, 1,482 tests_pass with zero-exit evidence,
98.4% keep rate, $15 spent). The pilot's A1/A2 post-training data exists;
only a tokenizer-specific re-convert is needed.

# Round 16p — HV_cur on the upgraded phil pool (2026-08-11, CLOSED)

Owner-approved single run: HV hybrid-MoE (1.316B total / 0.412B active) on
the cur curriculum with the rebuilt philosophy stage (600M modern
SEP+PhilPapers replacing 159M pre-1928 Gutenberg; math 600M, web 1800M,
3.0B unique, 5 epochs, 28650 steps — the dense factorial's exact regime).
One spot interruption (3.5h capacity wait, resumed step 18001, replayed
val within 0.0006 — determinism receipt). Total ~30 box-hours ≈ $610.
Best val_web **2.8142** @27000. No router collapse: drop_frac ≤0.005
mid-stage; the only >0.01 readings (0.0133/0.0115/0.0129) were web→math
stage-boundary transients that resolved to <0.001 by the next record.

Eval readout (1000/task) vs dense V_cur, the campaign's best cell:

| task | HV_cur (0.41B act) | V_cur (1.30B dense) | Δ |
|------|-------|-------|---|
| hellaswag | .378 | .402 | −2.4 |
| piqa | .664 | .706 | −4.2 |
| arc_easy | .5053 | .5228 | −1.8 |
| arc_challenge | .2977 | .3344 | −3.7 |
| winogrande | .503 | .530 | −2.7 |
| boolq | **.607** | .589 | +1.8 |
| sciq | .564 | .607 | −4.3 |
| lambada | .320 | .350 | −3.0 |
| **mean** | **.4799** | **.5051** | **−2.5pt** |

Honest read — this comparison moves TWO variables (backbone AND phil pool),
so it attributes nothing by itself:

- The −2.5pt eval / +0.117 loss deficit is consistent with the **3.2× active-
  param gap** (0.41B vs 1.30B): sparse active compute costs quality at
  matched total params. That is the known MoE trade, not a curriculum verdict.
- What the run DOES establish: the hybrid-MoE backbone trains stably for a
  full 1B-scale curriculum run (15B tokens, zero NaN, balancer healthy at
  56 experts, spot-resume exact), and the upgraded phil pool is drop-in.
- What it CANNOT establish: whether the new phil pool beats the old one
  (needs same-arch control: dense V_cur on data1b_p, or HV_cur on old
  data1b), or whether MoE-at-matched-active-compute is competitive (needs a
  dense ~0.4B control).
- boolq +1.8 is the lone gain; with boolq's known bimodality (R15) it
  carries no weight alone.

Cheapest attribution run if wanted next: dense V_cur retrained on data1b_p
(~$270 at dense speed) isolates the phil-pool effect exactly.

# Round 17 — phil-pool attribution (2026-08-11, CLOSED)

The single-variable control round 16p needed: dense V 1.30B, cur order,
identical to the factorial's V_cur cell except `--data-dir data1b_p`
(600M modern SEP+PhilPapers phil replacing 159M Gutenberg; web trimmed
2.25B→1.8B to hold 3.0B unique). One spot interruption (~2.5h wait,
resumed step 21001 from the periodic ckpt, zero steps lost). ~$300.
Best val_web 2.7252 @27000 vs V_cur 2.6969 (+0.028 — but T2 already
proved loss picks the wrong arm on philosophy; evals decide).

## Verdict: the upgraded phil pool LOSES on evals — new pool −2.3pt

| task | V_cur_p | V_cur | Δ phil-pool |
|------|---------|-------|---|
| hellaswag | .405 | .402 | +0.3 |
| piqa | .670 | .706 | −3.6 |
| arc_easy | .5053 | .5228 | −1.8 |
| arc_challenge | .2943 | .3344 | **−4.0** |
| winogrande | .511 | .530 | −1.9 |
| boolq | .555 | .589 | −3.4 |
| sciq | .596 | .607 | −1.1 |
| mean | **.4821** | **.5051** | **−2.3pt** |

The T2 prediction FAILED: modern academic philosophy did not buy
reasoning — arc_challenge (−4.0) and winogrande (−1.9) are among the
LARGEST losses, exactly the tasks the phil stage was supposed to feed.
The T2 result (phil > no-phil) came from GUTENBERG philosophy — classic
primary texts (argumentative prose, dialogues). The replacement pool is
PhilPapers open-access papers + SEP: modern academic register, citation
apparatus, LaTeX-flavored arXiv logic. Same stage budget, 3.8× the
unique tokens, worse result on every reasoning task. **The pool's
REGISTER matters more than its size or recency.** "More/newer
philosophy" is not the knob; "primary argumentative prose" apparently is.

## Bonus: r16p reread — the MoE backbone is nearly free at matched data

With r17 as the matched-data dense control, r16p's gap decomposes:

| | eval mean | active params |
|---|---|---|
| V_cur_p (dense, new phil) | .4821 | 1.30B |
| HV_cur_p (MoE, new phil) | .4799 | 0.41B |

**−0.2pt for 3.2× fewer active params.** The −2.5pt I attributed partly
to sparse compute was almost entirely the phil-pool regression. The
hybrid-MoE backbone at 0.41B active matches dense 1.30B on this suite —
strong support for the pilot's MoE architecture choice.

## What this changes for the pilot

1. **Data plan REVISED**: keep Gutenberg-register philosophy as the
   stage core; SEP/PhilPapers at most a minority mix — never a swap.
   An A/B mix rung (e.g. 150M Gutenberg + 150M SEP) is the cheap probe
   if we want to salvage the modern pool.
2. **Architecture CONFIRMED**: hybrid-MoE carries dense-level evals at
   ~1/3 active compute in the exact pilot regime.
3. SPEC_PILOT_MOE's data table needs the phil row rewritten (it
   currently recommends exactly the swap that just lost).

# Round 19 — curriculum ORDER permutations (2026-08-13, CLOSED)

Two new dense cells (V, 1.30B, AdamW, data1b, 5 epochs, seed 7) against
the factorial's controls. One brief spot blip (resumed 26001, replay val
identical to 4 decimals). ~$580.

| order | sequence | best loss | eval mean |
|---|---|---|---|
| **wl** | **web→math→phil** | **2.7110** | **.5116** |
| pm | phil→math→web | 2.6988 | .5080 |
| cur | math→phil→web | 2.6969 | .5051 |
| std | proportional shuffle | 2.7062 | .4870 |

Per-task (cur / pm / wl): arc_easy .5228/.5333/**.5439**, lambada
.350/.336/**.362**, sciq .607/.646/.633, arc_challenge **.3344**/.3144/.3278,
winogrande **.530**/.520/.525, piqa .706/.697/.702, boolq .589/.608/.598,
hellaswag .402/.409/.401.

## Pre-registered verdicts

- **P1 (abstract-first mechanism): PASS as written, then overtaken by P3.**
  pm (.5080) within noise of cur (.5051), both >> std. Order within
  {math, phil} doesn't matter — math-first is NOT load-bearing.
- **P2 (math-first load-bearing): FAIL** — pm did not land below cur.
- **P3 (staging vs order): STAGING IS THE MECHANISM, and the "anti"
  order is the best cell of the campaign.** wl > std by +2.5pt, and
  wl > cur by +0.65pt. All three staged orders beat the shuffle by
  1.8-2.5pt; the spread BETWEEN staged orders (0.65pt) is ~4x smaller
  than staged-vs-shuffle (1.8-2.5pt).

## Honest read

1. **Epoch-major staging is the recipe ingredient, not the mythology of
   which stage opens.** The campaign's "math first" story does not
   survive: web→math→phil — the order designed as the anti-curriculum —
   is the best cell at 1B/5ep/seed 7.
2. Why wl edges the others (mechanism guess, single seed): whatever
   stage comes LAST before the eval checkpoint is freshest; wl ends each
   epoch on phil+math (abstract), starts on web — and lambada/arc_easy
   (language/knowledge tasks) are exactly where wl gains. Recency, not
   priority, may be the real knob. A deliberate rung (same stages, vary
   only the final stage) would isolate it.
3. Caveats: 1 seed, 1000/task; wl's +0.65pt over cur is inside the
   plausible seed noise for this suite (dense same-arm spreads ~0.5pt).
   The robust claim is staged >> shuffled (2-2.5pt, replicated across
   4 staged cells); the wl>cur ordering claim needs a seed-13 pair
   before it goes in the recipe.
4. Loss again mis-ranks: cur has the best loss (2.6969) and the WORST
   eval mean of the staged trio. Fourth time evals reversed loss.

## Recipe state after r19

- Backbone: hybrid-MoE (r16p/r17: ≈dense at 1/3 active) — IN
- Optimizer: Muon, lr bracketed, early-stop mandatory (r18) — IN
- Data: Gutenberg-register phil (r17), staged epoch-major (r19) — IN
- Stage ORDER: staging >> shuffle is settled; exact order is a ~0.5pt
  refinement pending seeds. Thinking/agentic stay post-training (T3/T4).

# Round 20 — recency mechanism; ABORTED before replication (2026-08-13)

Owner closed the box after the fm cell finished ("taking too much time")
— V_wl seed-13 replication was ~5% in and is discarded. One completed
cell, ~$640 total for the round including the interrupted wl start.

**V_fm (web→phil→math, seed 7): best loss 2.7096, eval mean .5022.**
Per-task: arc_easy .5456 (best of ALL cells), lambada .378 (best of all),
sciq .597, boolq .591, piqa .698, hellaswag .391, winogrande .503,
arc_challenge .3144.

## Pre-registered verdicts

- **P1 (recency of abstract data): FAIL.** fm ends each epoch on an
  abstract stage, exactly like wl — and landed at .5022, below cur
  (.5051) and 0.94pt below wl (.5116). "End on abstract" is NOT the knob.
- **P2 (phil-specifically-last): SUPPORTED, one seed.** What separates
  wl from fm is only WHICH abstract stage closes the epoch. The per-task
  mirror is clean: fm keeps wl's knowledge/completion gains (arc_easy,
  lambada — both campaign bests) but gives back the reasoning pair
  (winogrande .503 vs wl .525, arc_challenge .3144 vs .3278). Philosophy
  adjacent to the checkpoint protects reasoning; math there does not.
  Consistent with T2 (phil buys reasoning) from an independent direction.
- **P3 (wl seed-13 replication): NOT RUN** — aborted. wl > cur therefore
  remains a single-seed observation and does NOT enter the recipe.

## What stands after r19+r20 (order block closed unless owner reopens)

1. **Staged epoch-major >> shuffled: settled** (+1.5-2.5pt, 5 staged
   cells vs std). This is the recipe ingredient.
2. **Order within the epoch: unresolved refinement, bounded at ~0.9pt.**
   Single-seed ranking wl .5116 > pm .5080 > cur .5051 > fm .5022, with
   a coherent phil-last story but no replication. Recorded as a
   HYPOTHESIS, not a rule.
3. Practical default for the pilot: keep phil as the last stage of each
   epoch (wl order) — best single-seed cell and mechanistically
   consistent — while claiming only the staging effect.
