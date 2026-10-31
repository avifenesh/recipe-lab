# recipe-lab

**Goal.** Find a training recipe that raises downstream quality for a small
**hybrid-MoE** model — and prove each ingredient on its own rung before it
goes in. Every round is pre-registered (predictions written into the run
script *before* launch), judged on **evals, not loss**, and replicated across
seeds before it is believed. Retractions are kept inline.

Checkpoints, loss curves and per-task evals for every 1B cell:
[huggingface.co/Avifenesh/recipe-lab-1b](https://huggingface.co/Avifenesh/recipe-lab-1b).

## The recipe so far

| ingredient | status | evidence |
|---|---|---|
| **Hybrid-MoE backbone** (Mamba-2 SSD + GQA 4:1 + fine-grained MoE, shared expert, aux-loss-free balancing) | **IN** | at matched data, 0.41B-active MoE scores .4799 vs 1.30B-dense .4821 — **−0.2 pt for 3.2× less active compute** (r16p/r17) |
| **Muon optimizer** (orthogonalized momentum on 2D hidden matrices, AdamW on embeddings/norms/scalars) | **IN** | beats AdamW on 4/4 arm×seed cells (−0.046 … −0.053 on vanilla) and reaches best-val in ~half the steps (r18) |
| **Epoch-major staged data** (replay math → phil → web each epoch) | **IN** | every staged order beats a proportional shuffle by **1.8–2.5 pt** eval mean; 4 cells (r19) |
| **Gutenberg-register philosophy** | **IN** | swapping in 3.8× more *modern academic* philosophy (SEP + PhilPapers) **lost 2.3 pt**, worst on the reasoning tasks it was meant to feed — register beats size and recency (r17) |
| **Stage order within the epoch** | **open** | spread between staged orders is 0.94 pt, single seed each; r20 refuted "end each epoch on abstract data" (fm .5022); phil-last is a one-seed hypothesis, pilot default only (r20) |
| Reasoning traces as a pretraining stage | **OUT** | −0.9 pt on the dense base, arc_challenge .334 → .288; they belong in post-training (T3) |
| Layer-loop + ε=λ/(N√L) | **OUT** | won 13/13 paired runs under AdamW, then **vanished under Muon** (gaps +0.0025 / −0.0084, sign-flipping) — the flip was AdamW-conditional (r18) |

### Two results worth reading in full

**Loss is the wrong instrument.** Four separate times the loss ranking and
the eval ranking disagreed, and the evals were right: philosophy ablation
(better loss, worse evals), 8-epoch extension (frozen loss gap, reversed
eval lead), the phil-pool swap, and the order permutations (`cur` has the
best loss of the staged trio and the *worst* eval mean). Any rung decided on
loss alone here would have shipped the wrong arm.

**The curriculum story survived; its mythology did not.** Staging is worth
~2 pt over shuffling, robustly. But "math first" — the thesis the campaign
was built around — is not load-bearing: `phil→math→web` ties it, and
`web→math→phil`, designed as the anti-curriculum, is the best cell measured
(.5116 vs .5051). Recency of abstract data failed its test in r20
(`web→phil→math` .5022); ending the epoch on philosophy is the remaining
single-seed hypothesis.

Full ledger with receipts: [RESULTS_1B.md](RESULTS_1B.md) (1B rounds) and
[FINDINGS.md](FINDINGS.md) (small-scale rounds + the closed loop campaign).

## Layout

- [RESULTS_1B.md](RESULTS_1B.md) — 1B ledger: the factorial, the four
  owner-set targets, the MoE/attribution/order rounds, incidents log.
- [FINDINGS.md](FINDINGS.md) — small-scale ledger, the loop campaign's
  proven/refuted list, the Muon verdict.
- [RECIPE.md](RECIPE.md) — hypothesis log: sources, pre-registered
  predictions, designs as they happened.
- [PAPER.md](PAPER.md): draft paper on the loop campaign ("Looping Beats
  Parameters Under the Data Wall").
- [SPEC_1B.md](SPEC_1B.md), [SPEC_1B_CURRICULUM.md](SPEC_1B_CURRICULUM.md):
  the 1B kill-or-confirm spec and the curriculum × loop factorial.
- [SPEC_PILOT_MOE.md](SPEC_PILOT_MOE.md) — the 4B-total/1.1B-active pilot
  spec, cost model from measured throughput, post-training fork.
- `lab/` — small-scale harness (`model.py`, `train.py` with `--opt muon`),
  per-round drivers, numerical proofs, MQAR probe.
- `lab1b/` — 1B harness: `model1b.py` (dense), `model_moe.py` (hybrid-MoE),
  `train1b.py` / `train_moe.py`, `prepare_staged_data.py`,
  `prepare_phil_pool.py`, `eval1b.py` / `eval_moe.py`, `upload_hf.py`,
  per-round `run_round*.sh` + `watch_teardown_*.sh`.
- `results/`, `evals/`, `lab1b/results_1b/` — every curve and eval as JSON.
- `proof1..5_*.py` — CPU numerical proofs from the loop campaign.

## Reproduce

Small-scale rounds fit on one 46 GB GPU; the 1B rounds used 8×H100 spot.

```bash
pip install torch numpy tiktoken datasets huggingface_hub

# --- small scale (d=384..1536) ---
cd lab
python prepare_data.py --tokens 250000000 --out-dir data
mkdir -p data12                                       # 12.5M-token subset the rounds use
head -c 25000000 data/fineweb_train.bin > data12/fineweb_train.bin
cp data/fineweb_val.bin data12/
python train.py --arm A768 --opt muon --muon-lr 1e-3 --steps 20000 \
  --data data12/fineweb_train.bin --val-data data12/fineweb_val.bin \
  --out results/smoke.json
bash run_round18.sh          # a full pre-registered round

# --- 1B scale ---
cd ../lab1b
python prepare_staged_data.py --out-dir data1b        # 3B-token staged corpus
python prepare_phil_pool.py  --out-dir data_phil      # modern phil pool
torchrun --standalone --nproc_per_node=8 train1b.py \
  --arm V --order cur --data-dir data1b --lr 6e-4 --out results/V_cur.json
python eval1b.py --ckpt ckpts/V_cur_best.pt --out evals_V_cur.json --limit 1000
python upload_hf.py --tag V_cur                       # publish weights+evals
```

### Operational notes (earned the hard way)

- **Muon's LR lives on a different scale.** With `adjust_lr_fn=match_rms_adamw`
  the optimum here was `1e-3`, 20× below the published default; the first
  probe grid was left-censored and picked its own edge. Always bracket an
  interior minimum.
- **Muon overfits multi-epoch data fast** (final val 7.0 vs best 4.63 at
  19.6 epochs). Best-val early-stop is mandatory, and any
  degradation-to-a-fixed-terminal-step metric will lie to you.
- **Verify artifacts by exact filename.** A glob (`V_std*_best.pt`) was
  satisfied by a sibling (`V_std_ext8_best.pt`) and a 5-epoch checkpoint was
  lost. Every teardown watcher now gates on exact names, and
  `upload_hf.py` treats a missing `_best.pt` as fatal.
- **Diagnose an unreachable box in this order**: spot request state → your
  own public IP → only then touch the instance. A timed-out SSH looks
  identical for spot reclaim, ISP rotation (SG no longer allows you) and a
  genuinely wedged host; all three happened, each needed a different fix.
- **Two silent MoE bugs that pass a forward-only check**: `torch.where`
  masking in the SSD scan differentiates the untaken branch (inf × 0 = NaN
  grads while the printed loss looks fine — mask multiplicatively), and the
  expert-load buffer must accumulate across the whole optimizer step or the
  balancer sees 1/grad_accum of the routing and parks at ~20% token drop.

## License

MIT.
