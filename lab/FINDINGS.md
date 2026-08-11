
## Round 18 — Muon on the flip rung (2026-08-12, free L40S, $0)

torch.optim.Muon (torch 2.13, match_rms_adamw) on 2D hidden matrices,
AdamW on embeddings/norms/scalars. LR probed properly after the first
grid left-censored (1e-2/2e-2/4e-2 all monotone down; extended to an
interior minimum at **muon-lr 1e-3** — 20x below the published 2e-2,
match_rms_adamw rescales). Exact round-9/10 regime, seeds 7/13.

| run | best val | @step |
|---|---|---|
| A768 muon s7 | 4.6302 | 3500 |
| C768 muon s7 | 4.6327 | 3500 |
| A768 muon s13 | 4.6373 | 3250 |
| C768 muon s13 | 4.6289 | 4500 |
| (AdamW A768 s7/s13) | 4.6831 / 4.6837 | 6000 / 5750 |
| (AdamW C768 s7/s13) | 4.6438 / 4.6481 | 7000 / 6500 |

- **P2 PASS 4/4 — Muon beats AdamW on every arm/seed** (A: −0.053/−0.046,
  C: −0.011/−0.019) and reaches best-val in ~half the steps (3250-4500 vs
  5750-7000): the ~2x-efficiency claim reproduces at d=768.
- **P1 FAIL — the loop flip does NOT survive Muon**: gaps +0.0025 (s7) /
  −0.0084 (s13). Sign flips across seeds, both inside seed noise, neither
  near the −0.02 bar. 13/13 streak was an AdamW-conditional result.
- **P3 = "Muon subsumes ε" branch**: Muon helps the VANILLA arm 3-5x more
  than the looped arm, erasing the gap from the A side. Consistent with
  mechanism: ε=1/N tames correlated residual accumulation; Muon's
  orthogonalized, spectrally-flat updates suppress the same pathology for
  every block, looped or not.
- Late-run overfit is much steeper under Muon (final val ~7.0 vs best
  ~4.63 at 19.6 epochs) — best-val early-stop essential; a fixed-horizon
  read would call Muon a disaster and be wrong.

**Recipe consequence (goal = SOTA hybrid-MoE small model): Muon IN,
loop+ε OUT.** Muon delivers more than the loop's entire margin, on both
arms, and the loop's advantage doesn't exist under it. The 1B lesson
(T1: loop loses at realistic ratios) now has a mechanism-level echo at
d=768. ε=1/N stays shelved unless some future rung shows Muon-scale
instability it uniquely fixes.
