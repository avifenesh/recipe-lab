"""DDP trainer for the 1B curriculum x loop factorial (SPEC_1B_CURRICULUM).

Cells: --arm {V,L} x --order {std,cur}
  std: every batch samples from the stage bins proportionally (shuffled mix)
  cur: epoch-major curriculum — within each epoch, batches walk
       stage1(math) -> stage2(phil) -> stage3(web) in proportion

Spot-safe: checkpoints model+optimizer+step every --ckpt-every steps to
--ckpt (atomic rename); auto-resumes if the file exists. Data order is a
pure function of (seed, step) so resume is exact.

Launch: torchrun --standalone --nproc_per_node=8 train1b.py \
    --arm L --order cur --out results/L_cur.json
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

from model1b import ARMS_1B, Config1B, GPT1B

# --order selects both the stage set and the serving order:
#   std/cur use the original 3 stages; cur4 inserts the thinking corpus;
#   nophil swaps philosophy for extra web (matched total budget).
STAGE_SETS = {
    "std": ["stage1_math", "stage2_phil", "stage3_web"],
    "cur": ["stage1_math", "stage2_phil", "stage3_web"],
    "cur4": ["stage1_math", "stage2_phil", "stage25_think", "stage3_web"],
    "nophil": ["stage1_math", "stage3b_webx", "stage3_web"],
    # order permutations (round 19+): same bins, different epoch-major order.
    # pm: philosophy before math — is math-first load-bearing or is it just
    # "abstract stages before web"?
    "pm": ["stage2_phil", "stage1_math", "stage3_web"],
    # wl (web-last control inverted): web FIRST, then math, then phil —
    # maximally anti-curriculum while stage-sequential; separates "staging
    # helps" from "THIS order helps".
    "wl": ["stage3_web", "stage1_math", "stage2_phil"],
}


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    return rank, world


class StagedData:
    """Serves batches in std (proportional shuffle) or cur (epoch-major
    staged) order. Deterministic in (seed, global_step)."""

    def __init__(self, data_dir, order, batch_size, block_size, seed,
                 steps_per_epoch):
        self.stages = STAGE_SETS[order]
        self.bins = {s: np.memmap(f"{data_dir}/{s}.bin", dtype=np.uint16,
                                  mode="r") for s in self.stages}
        self.sizes = {s: len(self.bins[s]) for s in self.stages}
        total = sum(self.sizes.values())
        self.props = [self.sizes[s] / total for s in self.stages]
        self.order = order
        self.bs, self.bl, self.seed = batch_size, block_size, seed
        self.spe = steps_per_epoch
        # curriculum: cumulative stage boundaries within an epoch,
        # proportional to size (any number of stages)
        acc, self.bounds = 0, []
        for p in self.props[:-1]:
            acc += int(p * steps_per_epoch)
            self.bounds.append(acc)

    def stage_for(self, step):
        e_step = step % self.spe
        for i, b in enumerate(self.bounds):
            if e_step < b:
                return self.stages[i]
        return self.stages[-1]

    def batch(self, step, rank, world):
        rng = np.random.default_rng((self.seed, step, rank))
        if self.order != "std":   # every non-std order is stage-sequential
            stage = self.stage_for(step)
            src = self.bins[stage]
            ix = rng.integers(0, len(src) - self.bl - 1, size=self.bs)
            xs = [src[i:i + self.bl + 1] for i in ix]
        else:
            xs = []
            stages = rng.choice(len(self.stages), size=self.bs, p=self.props)
            for si in stages:
                src = self.bins[self.stages[si]]
                i = int(rng.integers(0, len(src) - self.bl - 1))
                xs.append(src[i:i + self.bl + 1])
        arr = np.stack([a.astype(np.int64) for a in xs])
        t = torch.from_numpy(arr)
        return t[:, :-1].cuda(non_blocking=True), t[:, 1:].cuda(non_blocking=True)


@torch.no_grad()
def evaluate(model, path, batch_size, block_size, iters=40):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(1234)
    model.eval()
    losses = []
    for _ in range(iters):
        ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        arr = np.stack([data[i:i + block_size + 1].astype(np.int64) for i in ix])
        t = torch.from_numpy(arr)
        x, y = t[:, :-1].cuda(), t[:, 1:].cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS_1B, required=True)
    ap.add_argument("--order", choices=list(STAGE_SETS), required=True)
    ap.add_argument("--extend-from", default=None,
                    help="ckpt to continue: loads model+opt, keeps its curve, "
                         "runs steps [ckpt.step+1, --steps) with THIS run's "
                         "LR schedule (rewarm+cosine over the extension)")
    ap.add_argument("--data-dir", default="data1b")
    ap.add_argument("--steps", type=int, default=28650)   # 15B tok @ 0.524M/step
    ap.add_argument("--micro-bs", type=int, default=2)    # per GPU (4 OOMs on 80GB)
    ap.add_argument("--grad-accum", type=int, default=8)  # 2*8*8*4096=0.524M tok
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank, world = setup_dist()
    master = rank == 0
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = Config1B(block_size=args.block_size, **ARMS_1B[args.arm])
    model = GPT1B(cfg).cuda()
    raw = model
    model = torch.compile(model)
    if world > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    n_params = raw.num_params()
    if master:
        print(f"arm={args.arm} order={args.order} params={n_params/1e9:.2f}B "
              f"world={world}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    tokens_per_step = args.micro_bs * args.grad_accum * world * args.block_size
    # steps per epoch over the unique tokens of THIS order's stage set
    unique = sum(os.path.getsize(f"{args.data_dir}/{s}.bin") // 2
                 for s in STAGE_SETS[args.order])
    spe = max(1, unique // tokens_per_step)
    data = StagedData(args.data_dir, args.order, args.micro_bs,
                      args.block_size, args.seed, spe)

    start_step = 0
    ext_base = 0
    log = {"arm": args.arm, "order": args.order, "params": n_params,
           "config": vars(args), "curve": []}
    ckpt_path = args.ckpt or f"ckpts/{args.arm}_{args.order}.pt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        log = ck["log"]
        ext_base = ck.get("ext_base", 0)
        if master:
            print(f"resumed from step {start_step}", flush=True)
    elif args.extend_from and os.path.exists(args.extend_from):
        ck = torch.load(args.extend_from, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        ext_base = start_step          # LR schedule restarts from here
        log = ck["log"]
        log["extended_at"] = start_step
        if master:
            print(f"EXTENDING from {args.extend_from} @step {start_step}; "
                  f"rewarm LR over [{start_step}, {args.steps})", flush=True)

    def lr_at(step):
        # extension runs re-anchor warmup+cosine at ext_base so the added
        # epochs get their own rewarm and decay (identical for V and L)
        s = step - ext_base
        span = args.steps - ext_base
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, span - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(p, 1.0))))

    best = min((c["val_web"] for c in log["curve"]), default=float("inf"))
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

        if step % args.eval_every == 0 or step == args.steps - 1:
            if master:
                vw = evaluate(raw, f"{args.data_dir}/val_web.bin",
                              args.micro_bs, args.block_size)
                vm = evaluate(raw, f"{args.data_dir}/val_math.bin",
                              args.micro_bs, args.block_size)
                rec = {"step": step, "tokens": step * tokens_per_step,
                       "train_loss": round(loss.item(), 4),
                       "val_web": round(vw, 4), "val_math": round(vm, 4),
                       "gnorm": round(float(gnorm), 3),
                       "stage": data.stage_for(step * args.grad_accum),
                       "sec": round(time.time() - t0, 1)}
                log["curve"].append(rec)
                print(rec, flush=True)
                if vw < best:
                    best = vw
                    os.makedirs("ckpts", exist_ok=True)
                    torch.save({"model": raw.state_dict(), "arm": args.arm,
                                "order": args.order, "step": step,
                                "val_web": vw, "val_math": vm},
                               ckpt_path.replace(".pt", "_best.pt"))
            if world > 1:
                dist.barrier()

        if step % args.ckpt_every == 0 and step > start_step and master:
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            tmp = ckpt_path + ".tmp"
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                        "step": step, "log": log, "ext_base": ext_base}, tmp)
            os.replace(tmp, ckpt_path)

    if master:
        log["final_val_web"] = log["curve"][-1]["val_web"]
        log["best_val_web"] = best
        log["wall_sec"] = round(time.time() - t0, 1)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(log, open(args.out, "w"), indent=1)
        print(f"done: best={best:.4f} -> {args.out}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
