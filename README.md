# recipe-lab

**Question.** Does layer-looping (weight-tied block reuse) plus the
ε = λ/(N√L) residual-scaling fix beat a FLOPs-matched vanilla transformer —
and if so, where? Loopie (arXiv 2607.16051) proved layer-loop ordering but
never scaled residuals; the residual-scaling paper (arXiv 2606.18524) proved
ε = 1/N but never tested layer-loop. This repo combined them, pre-registered
every round, and ran the campaign to a verdict.

**Verdict** (eleven GPU rounds + five numerical proofs + one MQAR probe;
full ledger with receipts in [FINDINGS.md](FINDINGS.md)):

- **ε = λ/(N√L) is real and mandatory for looped training.** Gain over
  unscaled = 0.0122·(N−1) in val loss (R² = 0.993), holds at every arm's
  optimal LR, and pins the optimal LR at its N=1 value (hyperparameter
  transfer). Applies to attention and SSM mixers alike.
- **On fresh single-epoch data, stored params beat looping** at every scale
  tested — refuted at every N, both mixers, all LRs.
- **In the data-constrained (multi-epoch) regime the verdict flips with
  scale**: at d=768, loop+ε beat the param-matched vanilla on
  best-achievable val for attention (3/3 seeds, −0.039) and pure SSM
  (2/2 seeds, −0.026/−0.013), with a param-ordered overfit cliff (vanilla
  degrades 2-4× more past its minimum).
- **In hybrids, loop the state-mixer, never the retriever**: looping
  attention through shared weights cost +0.25-0.28; looping only the SSM
  blocks won by −0.018 with 26% fewer params (one compared seed).
- **At inference, extrapolate loops with the training ε** (MQAR probe);
  looped SSMs keep O(d_state·d) state regardless of loop count — the only
  variant with an inference-memory win.

Two single-seed headlines were caught and retracted by the pre-registered
no-single-seed rule; the retractions are kept inline in FINDINGS.md.

## Layout

- [FINDINGS.md](FINDINGS.md) — the proven/refuted ledger, round by round.
- [RECIPE.md](RECIPE.md) — hypothesis log: sources, pre-registered
  predictions, per-round designs and results as they happened.
- `proof1..5_*.py` — CPU-runnable numerical proofs of the accumulation
  pathology, layer-loop vs model-loop correlation, pre-LN survival, LR
  transfer, and mixer-agnosticism.
- `lab/` — training code (`model.py`, `train.py`), per-round drivers
  (`run_round*.sh`, pre-registered predictions in the headers), analyzers,
  and the MQAR probe (`mqar_loop.py`, receipts in `mqar.log`/`mqar2.log`).
- `results/` — every run's loss curve and config as JSON, named
  `<phase>_<arm>_s<seed>.json`.

## Reproduce

Any CUDA GPU with ~46 GB works for the d=768 rounds; d=384 rounds fit in
much less. Rounds were run on a single rented L40S; the MQAR probe runs on
a desktop GPU in ~10 minutes.

```bash
pip install torch numpy tiktoken datasets

cd lab
# tokenize FineWeb-Edu (12.5M-token bin for the epoch rounds used --tokens 12500000)
python prepare_data.py --tokens 250000000 --out-dir data

# any single arm
python train.py --arm C --steps 4000 --out results/smoke_C_s7.json

# a full pre-registered round
bash run_round10.sh

# numerical proofs (CPU, seconds each)
python ../proof1_accumulation.py

# MQAR probe
python mqar_loop.py
```

## License

MIT.
