"""Compute-matched pretraining harness for the loop-scaling experiment.

All arms process the same tokens through the same number of block applications
per token (effective depth 12), so FLOPs per token are matched up to the
parameter-count difference in embeddings. Data order is seed-fixed and
identical across arms.

Usage:
  python train.py --arm A --steps 4000 --seed 7 --out results/A_s7.json
Arms: A=vanilla12  B=layerloop6x2  C=layerloop6x2-scaled  D=modelloop6x2-scaled
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig

ARMS = {
    "A": dict(n_stored=12, n_loop=1, schedule="vanilla", res_scale=1.0),
    "B": dict(n_stored=6, n_loop=2, schedule="layer", res_scale=1.0),
    "C": dict(n_stored=6, n_loop=2, schedule="layer", res_scale=0.5),
    "D": dict(n_stored=6, n_loop=2, schedule="model", res_scale=0.5),
    # N=4 ladder: same 12 block-applications/token, discriminating test of
    # eps ordering 1/N < 1/sqrt(N) < 1 predicted by Thm 1 (2606.18524)
    "E1": dict(n_stored=3, n_loop=4, schedule="layer", res_scale=1.0),
    "E2": dict(n_stored=3, n_loop=4, schedule="layer", res_scale=0.5),
    "E3": dict(n_stored=3, n_loop=4, schedule="layer", res_scale=0.25),
}


def get_batches(data: np.ndarray, batch_size: int, block_size: int,
                seed: int, device: str):
    rng = np.random.default_rng(seed)
    n = len(data) - block_size - 1
    while True:
        ix = rng.integers(0, n, size=batch_size)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, batch_size, block_size, device, iters=50):
    model.eval()
    gen = get_batches(data, batch_size, block_size, seed=1234, device=device)
    losses = []
    for _ in range(iters):
        x, y = next(gen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", default="data/fineweb_train.bin")
    ap.add_argument("--val-data", default="data/fineweb_val.bin")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda"
    cfg = GPTConfig(block_size=args.block_size, **ARMS[args.arm])
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"arm={args.arm} schedule={cfg.schedule} stored={cfg.n_stored} "
          f"loops={cfg.n_loop} res_scale={cfg.res_scale} params={n_params/1e6:.1f}M")

    train_data = np.memmap(args.data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    gen = get_batches(train_data, args.batch_size, args.block_size,
                      seed=args.seed, device=device)
    log = {"arm": args.arm, "seed": args.seed, "params": n_params,
           "config": vars(args), "curve": []}
    t0 = time.time()
    tokens_per_step = args.batch_size * args.block_size

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = next(gen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            val = evaluate(model, val_data, args.batch_size, args.block_size, device)
            elapsed = time.time() - t0
            rec = {"step": step, "tokens": step * tokens_per_step,
                   "train_loss": round(loss.item(), 4), "val_loss": round(val, 4),
                   "gnorm": round(float(gnorm), 3), "sec": round(elapsed, 1)}
            log["curve"].append(rec)
            print(rec, flush=True)

    log["final_val"] = log["curve"][-1]["val_loss"]
    log["wall_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(log, f, indent=1)
    print(f"done: final_val={log['final_val']} wall={log['wall_sec']}s -> {args.out}")


if __name__ == "__main__":
    main()
