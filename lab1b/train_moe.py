"""Trainer for the hybrid-MoE arms (HV vs HL), single-GPU or DDP.

Same spot-safe contract as train1b.py: atomic model+opt+log checkpoints,
exact resume, data order a pure function of (seed, step). Adds the MoE
balancer step and expert-load logging.

The staged curriculum orders (cur/cur4/nophil) are reused verbatim from
train1b.py so the MoE rung can be compared to the dense factorial. The
sign-check rung uses a single flat bin instead (--flat-data).

Launch (sign-check, 1 GPU):
  python train_moe.py --arm HL --size s --flat-data data12/fineweb_train.bin \
      --flat-val data12/fineweb_val.bin --steps 20000 --out results/x.json
Launch (1B, 8 GPU):
  torchrun --standalone --nproc_per_node=8 train_moe.py --arm HL --size b \
      --order cur --out results/HL_cur.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model_moe import ARMS_MOE, SIZES_MOE, build
from train1b import STAGE_SETS, StagedData, evaluate


class FlatData:
    """Single-bin sampler for the sign-check rung (matches lab/train.py)."""

    def __init__(self, path, batch_size, block_size, seed):
        self.bin = np.memmap(path, dtype=np.uint16, mode="r")
        self.bs, self.bl, self.seed = batch_size, block_size, seed

    def batch(self, step, rank, world):
        rng = np.random.default_rng((self.seed, step, rank))
        ix = rng.integers(0, len(self.bin) - self.bl - 1, size=self.bs)
        arr = np.stack([self.bin[i:i + self.bl + 1].astype(np.int64)
                        for i in ix])
        t = torch.from_numpy(arr)
        return (t[:, :-1].cuda(non_blocking=True),
                t[:, 1:].cuda(non_blocking=True))

    def stage_for(self, step):
        return "flat"


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    return rank, world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS_MOE, required=True)
    ap.add_argument("--size", choices=SIZES_MOE, default="b")
    ap.add_argument("--order", choices=list(STAGE_SETS), default="cur")
    ap.add_argument("--data-dir", default="data1b")
    ap.add_argument("--flat-data", default=None,
                    help="single-bin training data (sign-check rung); "
                         "when set, --order/--data-dir are ignored")
    ap.add_argument("--flat-val", default=None)
    ap.add_argument("--steps", type=int, default=28650)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=None,
                    help="defaults to the size preset's block_size")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--track-overlap", action="store_true",
                    help="log router reuse across loop applications; forces "
                         "--no-compile since the tracker's side effect would "
                         "graph-break. For short probes, not full runs.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank, world = setup_dist()
    master = rank == 0
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    over = {}
    if args.block_size:
        over["block_size"] = args.block_size
    if args.track_overlap:
        over["track_router_overlap"] = True
        args.no_compile = True
    raw = build(args.arm, args.size, **over).cuda()
    cfg = raw.cfg
    block = cfg.block_size
    model = raw if args.no_compile else torch.compile(raw)
    if world > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()])

    n_total, n_active = raw.num_params(), raw.num_active_params()
    if master:
        types = "".join("a" if t else "s" for t in raw.types)
        print(f"arm={args.arm} size={args.size} stored=[{types}] "
              f"applications={raw.n_applications()} "
              f"total={n_total/1e9:.3f}B active={n_active/1e9:.3f}B "
              f"world={world}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    tokens_per_step = args.micro_bs * args.grad_accum * world * block
    if args.flat_data:
        data = FlatData(args.flat_data, args.micro_bs, block, args.seed)
        val_paths = {"val": args.flat_val}
    else:
        unique = sum(os.path.getsize(f"{args.data_dir}/{s}.bin") // 2
                     for s in STAGE_SETS[args.order])
        spe = max(1, unique // tokens_per_step)
        data = StagedData(args.data_dir, args.order, args.micro_bs, block,
                          args.seed, spe)
        val_paths = {"val_web": f"{args.data_dir}/val_web.bin",
                     "val_math": f"{args.data_dir}/val_math.bin"}
    main_val = list(val_paths)[0]

    start_step = 0
    log = {"arm": args.arm, "size": args.size,
           "order": "flat" if args.flat_data else args.order,
           "params": n_total, "active_params": n_active,
           "applications": raw.n_applications(),
           "chance_overlap": raw.chance_overlap(),
           "config": vars(args), "curve": []}
    ckpt_path = args.ckpt or f"ckpts/{args.arm}_{args.size}.pt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        log = ck["log"]
        if master:
            print(f"resumed from step {start_step}", flush=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(p, 1.0))))

    best = min((c[main_val] for c in log["curve"]), default=float("inf"))
    t0 = time.time()
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = data.batch(step * args.grad_accum + micro, rank, world)
            ctx = (model.no_sync() if world > 1 and micro < args.grad_accum - 1
                   else torch.enable_grad())
            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                (loss / args.grad_accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # read the routing stats for this step BEFORE rebalance() consumes them
        moe_stats = (raw.drop_frac(), raw.router_overlap())
        raw.rebalance()      # aux-loss-free load balancing, no extra gradient

        if step % args.eval_every == 0 or step == args.steps - 1:
            if master:
                vals = {k: round(evaluate(raw, p, args.micro_bs, block), 4)
                        for k, p in val_paths.items()}
                rec = {"step": step, "tokens": step * tokens_per_step,
                       "train_loss": round(loss.item(), 4), **vals,
                       "gnorm": round(float(gnorm), 3),
                       "drop_frac": round(moe_stats[0], 5),
                       # looped arms only: ~1.0 means the loop re-picks the
                       # same experts every application (degenerate), chance
                       # is k/E. NaN for un-looped arms.
                       "router_overlap": round(moe_stats[1], 4),
                       "stage": data.stage_for(step * args.grad_accum),
                       "sec": round(time.time() - t0, 1)}
                log["curve"].append(rec)
                print(rec, flush=True)
                if vals[main_val] < best:
                    best = vals[main_val]
                    os.makedirs(os.path.dirname(ckpt_path) or ".",
                                exist_ok=True)
                    torch.save({"model": raw.state_dict(), "arm": args.arm,
                                "size": args.size, "step": step,
                                "block_size": block, **vals},
                               ckpt_path.replace(".pt", "_best.pt"))
            if world > 1:
                dist.barrier()

        if step % args.ckpt_every == 0 and step > start_step and master:
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            tmp = ckpt_path + ".tmp"
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                        "step": step, "log": log}, tmp)
            os.replace(tmp, ckpt_path)

    if master:
        log[f"final_{main_val}"] = log["curve"][-1][main_val]
        log[f"best_{main_val}"] = best
        log["wall_sec"] = round(time.time() - t0, 1)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(log, open(args.out, "w"), indent=1)
        print(f"done: best={best:.4f} -> {args.out}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
