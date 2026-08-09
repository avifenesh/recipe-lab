# Pilot spec — hybrid-MoE base + post-training fork (owner design, 2026-08-09)

Owner's target model, in his words: *"hybrid moe that start training with a lot
of math, from basic to frontier max phd level - then more pilosophy then we did
and to much higher level - then workld knowledge, a lot and up to date - base
model, on top of that few direction -> A1 keep training with my sft for agentic
then reasoning / A2 keep training first reasoning then agentic / B1 RL sft then
reasoning / B2 RL reasoning then agentic"*.

This spec is **gated**, not scheduled. It commits money only after the two
cheap rungs below answer the two questions that can kill it. Both rungs run on
hardware that is either free (the L40S) or already sized from measured numbers.

## Gates before any pilot spend

| gate | question | rung | status |
|------|----------|------|--------|
| G1 | does the loop flip survive a hybrid-MoE backbone at all? | round 15, L40S, free | running |
| G2 | does it survive at 1B on the winning curriculum, in a flip-possible epoch regime? | round 16, 2×p5 | blocked on G1 + round 14 T1 |

**G1 failing cancels the loop half of the pilot** — build the pilot as a
plain hybrid MoE (still the field-standard backbone, still worth the run) and
drop the weight sharing. **G2 failing** means the flip is a sub-1B phenomenon
on this backbone; the pilot then uses HV sizing and the campaign result gets a
scale bound rather than a scaling claim.

Do not read a pass on G1 as a pass on G2. The dense campaign's own history is
the warning: the flip verified 3/3 seeds at d=768 and then **lost** at 1B/5
epochs, because the flip needs a high params-per-unique ratio. That is a
measured boundary, not a fluke, and it is the single most likely way this
pilot wastes money.

## Architecture

`lab1b/model_moe.py`, validated by `lab1b/test_moe.py` (SSD scan exact to the
fp32 roundoff floor vs a naive fp64 recurrence, chunk-invariant, causal,
finite gradients; balancer direction; arms provably application-matched).

The backbone is what three independent 2026 lines converged on — a minority of
softmax-attention layers over a majority of fixed-state recurrent layers, with
a sparse feed-forward: Qwen3-Next and Qwen3.5 (Gated DeltaNet 3:1), Kimi Linear
(KDA ~3:1), Nemotron-3 (Mamba-2 ~4:1). Using it means the scaling claim is
tested on the substrate people actually ship, and it is also the substrate the
campaign's design rule was written for: **loop the state-mixer, never the
retriever** (round 11c — looping attention lost by +0.25 to +0.28 even with
placement matched; looping only the SSM blocks won 2/2 seeds).

Pilot sizing, 4:1 linear:attention, `d_model` 2048:

| | total | active | notes |
|---|---|---|---|
| base | ~4.0B | ~1.1B | serves on the owner's 5090 at Q8 |

Loop arm (if G1+G2 pass) stores half the linear blocks for the same
applications per token, which is where the parameter saving comes from.

One integration risk stays open and is worth naming: MoE + weight-shared
linear blocks means a looped block reuses its **router** as well as its
weights across applications. Nothing in the campaign or the published
hybrid-MoE work tests that. Round 15's `drop_frac` gate catches the crude
failure (routing collapse); a subtler "same expert every application" mode
would not show up in loss. If G1 passes, log per-application expert overlap in
round 16 before the pilot.

## Data — upgrading each stage the owner named

Corpora chosen for 2026 availability, not for what the 1B run happened to use.
The 1B run's own numbers say which stage matters and why.

| stage | 1B run used | pilot uses | why |
|-------|-------------|-----------|-----|
| math (basic → PhD) | finemath-3plus, 0.6B | **Nemotron-CC-Math-4+** (52B avail) + finemath; keep a graded easy→hard ordering *within* the stage | Nemotron-CC-Math-4+ reports beating MegaMath/FineMath/OpenWebMath; it is the current best open math pretraining corpus and it is large enough that the math stage stops being pool-capped |
| philosophy (higher level) | gutenberg philosophy, pool capped at ~159M | **SEP full corpus + PhilPapers open-access + arXiv math.LO/cs.LO**; target ≥1B | the 1B run wanted 0.2B and the pool only had 0.159B — the owner's central thesis stage was the one starved. Gutenberg is also pre-1928 by construction, so "higher level modern philosophy" needs SEP, not more Gutenberg |
| world knowledge (up to date) | fineweb-edu sample-10BT | **fineweb-edu 2026 dumps** + wikipedia current | "up to date" is the owner's requirement and the 10BT sample is a fixed old slice |

Ordering stays epoch-major (each epoch replays math → phil → knowledge) —
pure stage-sequential catastrophically forgets stage 1, which is why the 1B
factorial used epoch-major and why its curriculum effect was measurable at all.

### What the 1B evidence says about this data plan

Honest reading, because two of these results cut against the plan's optimism:

- **Curriculum works, but small.** V_cur 2.6969 < V_std 2.7062 on loss, and
  −0.0093 identically on both architectures. On evals the V pair moved +1.8pt.
  Real, reproducible, additive — not a phase change.
- **Philosophy buys reasoning and costs facts.** V_nophil scored *better loss*
  (2.6907) than V_cur while scoring *worse evals*, with the split legible:
  arc_challenge .334 vs .314 and winogrande .530 vs .502 for philosophy,
  sciq .607 vs .636 against it. This is the strongest existing support for the
  owner's thesis, and it is also why the pilot must be judged on reasoning
  evals, not loss. Loss would have picked the wrong arm here.
- **Thinking-as-a-pretraining-stage failed.** V_cur4 (reasoning traces as a
  fourth stage) landed at eval mean .4961 vs V_cur .5051, with
  arc_challenge collapsing to .288. So reasoning traces belong in
  **post-training**, where the A/B fork below puts them — not in the base
  curriculum. Round 14's L_cur4 tests whether a looped model changes that.

## Post-training fork

The owner named four directions. Two of them are unbuildable today and saying
so now is cheaper than discovering it later:

- **A1/A2 (SFT: agentic then reasoning / reasoning then agentic)** — buildable.
  The tooling exists in `~/projects/darklanes/sft-pipeline` (validate → verify
  outcome by patch-apply + tests → K3→ChatML convert → stats), but the corpus
  is **not generated**: only fixtures are on disk. Pilot corpus is ~$85-150
  per CORPUS.md, and CORPUS.md also records that Kimi/Qwen subscriptions and
  the NVIDIA free API are contractually barred from corpus generation, so that
  cost is real and cannot be worked around.
- **B1/B2 (RL)** — no reward model, no RL loop, no infrastructure in this repo.
  `verify_outcome.py` is a genuine reusable reward gate for agentic tasks, and
  that is the only RL piece that exists. Building B is a separate project, not
  a stage of this one.

**Recommendation: run A2 first, and only A2 vs A1.** The curriculum result is
the reason — this campaign's one transferable finding about ordering is that
teaching the general capability before the applied one helps
(math/philosophy before web, −0.0093 and +1.8pt evals). A2 is that same
ordering at the post-training level: reasoning first, then agentic. A1 is its
control. Running A1 vs A2 costs one extra fine-tune on a base model that is
already paid for, and it tests the owner's ordering thesis a second time in a
regime where it has never been tested. Defer B until A has a verdict.

## Cost, from measured throughput

Anchored on the dense 1.3B run's real numbers (`V_cur`: 28650 steps in 13.3h
on 8×H100 = 1.672 s/step at 0.524M tokens/step ≈ 313k tok/s), p5.48xlarge spot
at ~$20.4/h in us-east-2. Scaling by active params and carrying a **1.5×
MoE overhead factor** for routing and dispatch, which the naive
active-parameter scaling ignores:

| item | tokens | est. wall | est. cost |
|------|--------|-----------|-----------|
| G1 round 15 (4 runs, L40S) | — | ~13 h | **$0** (already paid) |
| G2 round 16 (HV+HL, 2 boxes parallel) | 24B each | ~7-10 h | ~$300-450 |
| pilot base, 4B/1.1B-active | 120B | ~135 h (1 box) | ~$2.8K |
| A1 + A2 fine-tunes | — | ~10 h | ~$200 |
| SFT corpus generation | — | — | ~$85-150 |
| **total to a forked, evaluated pilot** | | | **~$3.4-3.7K** |

Cheaper than the $8-15K I floated before the numbers were measured. The base
run is the whole cost; everything else is rounding. Multi-box parallelism cuts
wall time at the same GPU-hours, as it did in round 14.

Token budget note: 120B over ~1.1B active is ~109 tokens/active-param, inside
the 100-200 range that is now normal practice for production models. If the
loop arm wins G2, the flip's own precondition (high params-per-unique) argues
for **fewer unique tokens at more epochs** rather than a bigger pool — decide
that from round 16's measured gap, not from this table.

## Kill conditions

- G1 fails → drop the loop, build a plain hybrid MoE.
- G2 fails → the flip is sub-1B on this backbone; publish the bound.
- Per-application expert overlap high in round 16 → the loop is degenerate
  under MoE; drop the loop even if loss looks fine.
- Pilot base eval mean below the dense 1B V_cur (.5051) on reasoning tasks →
  the stack is worse than what already exists; stop before post-training.

Evals decide, not loss. V_nophil is the receipt: it won on loss and lost on
evals, and loss would have picked the wrong model.
