"""Measure wall-clock per optimizer step for candidate arm-F widths.

Loopie's recipe matches *measured* step time, not analytical FLOPs: looping
halves stored params, the freed memory/efficiency headroom is reinvested into
width until step time again matches the vanilla reference. This script finds
the widest layer-loop 6x2 config whose step time matches vanilla-12 d=384.
"""

import argparse
import time

import torch

from model import GPT, GPTConfig


def bench(cfg, batch_size, iters=30, warmup=10):
    model = GPT(cfg).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                            betas=(0.9, 0.95), weight_decay=0.1)
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size),
                      device="cuda")
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size),
                      device="cuda")

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 2**30
    n = model.num_params()
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return ms, mem, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    candidates = [
        ("A ref", dict(n_stored=12, n_loop=1, schedule="vanilla",
                       res_scale=1.0, d_model=384, n_head=6)),
        ("F384", dict(n_stored=6, n_loop=2, schedule="layer",
                      res_scale=0.5, d_model=384, n_head=6)),
        ("F448", dict(n_stored=6, n_loop=2, schedule="layer",
                      res_scale=0.5, d_model=448, n_head=7)),
        ("F512", dict(n_stored=6, n_loop=2, schedule="layer",
                      res_scale=0.5, d_model=512, n_head=8)),
        ("F576", dict(n_stored=6, n_loop=2, schedule="layer",
                      res_scale=0.5, d_model=576, n_head=9)),
    ]
    print(f"{'name':>6} {'d':>4} {'params':>8} {'ms/step':>8} {'peakGB':>7}")
    for name, kw in candidates:
        cfg = GPTConfig(**kw)
        ms, mem, n = bench(cfg, args.batch_size)
        print(f"{name:>6} {kw['d_model']:>4} {n/1e6:>7.1f}M {ms:>8.1f} {mem:>7.2f}")


if __name__ == "__main__":
    main()
