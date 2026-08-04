"""MQAR probe: can looping substitute for attention at GENERATION time?

Owner's question: hybrid/pure Mamba misses attention's exact-recall ability
when generating. Does looping the SSM recover it?

Theory says NO for memory-bound recall (state too small = info gone; loops
reprocess the same compressed state) and YES for compute-bound depth (more
sequential processing of stored info). MQAR (multi-query associative recall,
Zoology/Based line) separates the two walls cleanly:

  memory wall:  kv_pairs * (key+val bits) vs d_state capacity
  compute wall: retrieval/composition depth vs number of block applications

Design: tiny looped-SSM models (d=128), 4 block applications per token for
every arm. Grid:
  d_state in {8, 64}       (starved vs sufficient state)
  kv_pairs in {4, 16}      (light vs heavy memory load)
  arms: L1 = 4 stored blocks, no loop     (params 4x, apps 4)
        L4 = 1 stored block, looped 4x, eps=1/4  (params 1x, apps 4)
        L4u= 1 stored block, looped 4x, eps=1    (unscaled control)

Pre-registered predictions:
  P1 (memory wall): at d_state=8, kv=16: ALL arms fail (< 60% acc);
     looping does not rescue recall the state cannot hold.
  P2 (compute substitution): at d_state=64: L4 within 5 acc points of L1 on
     every kv load — loop substitutes stored depth for recall that fits.
  P3 (eps): L4 >= L4u everywhere; gap grows where the task is harder.

Also test-time elasticity (train N=4, eval N in {2,4,8} with eps
readjusted to 1/N_eval vs frozen at 1/4):
  P4: with readjusted eps, accuracy degrades gracefully and N=8 does not
      blow up; with frozen eps, N=8 output norms explode (proof-1 dynamics).

Runs on the local 5090 under a CPU quota; ~10 min total.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"


def make_mqar(n, seq_len, kv_pairs, vocab_kv, seed):
    """Sequence: k1 v1 k2 v2 ... then queries: k_i ? -> predict v_i.
    Keys from [2, 2+vocab_kv), values from [2+vocab_kv, 2+2*vocab_kv).
    Query section re-presents keys in random order; target is the value."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, seq_len), dtype=np.int64)
    y = np.full((n, seq_len), -1, dtype=np.int64)
    for i in range(n):
        keys = rng.choice(vocab_kv, size=kv_pairs, replace=False) + 2
        vals = rng.integers(0, vocab_kv, size=kv_pairs) + 2 + vocab_kv
        pos = 0
        for k, v in zip(keys, vals):
            x[i, pos], x[i, pos + 1] = k, v
            pos += 2
        order = rng.permutation(kv_pairs)
        for j in order:
            x[i, pos] = keys[j]
            y[i, pos] = vals[j]     # predict value when key re-shown
            pos += 1
    return torch.from_numpy(x), torch.from_numpy(y)


class SSMBlock(nn.Module):
    def __init__(self, d, ds):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)
        self.W_dt = nn.Linear(d, 1)
        self.a_log = nn.Parameter(torch.zeros(ds))
        self.B = nn.Linear(d, ds, bias=False)
        self.C = nn.Linear(ds, d, bias=False)
        self.D = nn.Parameter(torch.zeros(d))
        self.gate = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, eps):
        h = self.ln(x)
        T = h.shape[1]
        hc = F.silu(self.conv(h.transpose(1, 2))[..., :T].transpose(1, 2))
        dt = F.softplus(self.W_dt(hc)).float()
        decay = -F.softplus(self.a_log.float())[None, None] * dt
        u = dt * self.B(hc).float()
        logP = decay.cumsum(1).clamp(min=-30.0)
        P = logP.exp()
        s = P * (u / P).cumsum(1)
        m = self.C(s.to(x.dtype)) + hc * self.D
        x = x + eps * self.out(m * F.silu(self.gate(h)))
        x = x + eps * self.mlp(self.ln2(x))
        return x


class LoopSSM(nn.Module):
    def __init__(self, vocab, d, ds, n_stored, n_loop, eps):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(SSMBlock(d, ds) for _ in range(n_stored))
        self.n_loop, self.eps = n_loop, eps
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, n_loop=None, eps=None):
        n_loop = n_loop or self.n_loop
        eps = eps or self.eps
        h = self.emb(x)
        for blk in self.blocks:
            for _ in range(n_loop):
                h = blk(h, eps)
        return self.head(self.ln_f(h))


def train_eval(arm, ds, kv, seed=0, steps=1500, n=4096, seq_len=None,
               vocab_kv=64, d=128, lr=1e-3):
    seq_len = seq_len or (kv * 3 + 4)
    vocab = 2 + 2 * vocab_kv
    torch.manual_seed(seed)
    cfgs = {
        "L1": dict(n_stored=4, n_loop=1, eps=1.0),
        "L4": dict(n_stored=1, n_loop=4, eps=0.25),
        "L4u": dict(n_stored=1, n_loop=4, eps=1.0),
    }
    model = LoopSSM(vocab, d, ds, **cfgs[arm]).to(DEV)
    xtr, ytr = make_mqar(n, seq_len, kv, vocab_kv, seed)
    xte, yte = make_mqar(1024, seq_len, kv, vocab_kv, seed + 999)
    xtr, ytr, xte, yte = (t.to(DEV) for t in (xtr, ytr, xte, yte))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bs = 256
    for step in range(steps):
        i = torch.randint(0, n, (bs,), device=DEV)
        logits = model(xtr[i])
        loss = F.cross_entropy(logits.view(-1, vocab), ytr[i].view(-1),
                               ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(xte)
        mask = yte >= 0
        acc = (logits.argmax(-1)[mask] == yte[mask]).float().mean().item()
    return model, acc, xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    print("=== MQAR grid (mean of 2 seeds) ===")
    print(f"{'d_state':>8} {'kv':>4} {'L1(4 stored)':>13} {'L4(1x4 eps=1/4)':>16} {'L4u(eps=1)':>11}")
    models = {}
    for ds in (8, 64):
        for kv in (4, 16):
            row = f"{ds:>8} {kv:>4}"
            for arm in ("L1", "L4", "L4u"):
                accs = []
                for seed in (0, 1):
                    m, a, xte, yte = train_eval(arm, ds, kv, seed=seed,
                                                steps=args.steps)
                    accs.append(a)
                    if arm == "L4" and seed == 0:
                        models[(ds, kv)] = (m, xte, yte)
                row += f" {100*np.mean(accs):>12.1f}%" if arm == "L1" else \
                       f" {100*np.mean(accs):>15.1f}%" if arm == "L4" else \
                       f" {100*np.mean(accs):>10.1f}%"
            print(row, flush=True)

    print("\n=== Test-time loop elasticity (L4 trained N=4, d_state=64, kv=4) ===")
    m, xte, yte = models[(64, 4)]   # the cell where L4 actually works (59%)
    vocab = 2 + 2 * 64
    mask = yte >= 0
    print(f"{'N_eval':>7} {'eps readjusted':>15} {'eps frozen=1/4':>15}")
    for n_eval in (2, 4, 8):
        accs = []
        for eps in (1.0 / n_eval, 0.25):
            with torch.no_grad():
                logits = m(xte, n_loop=n_eval, eps=eps)
                acc = (logits.argmax(-1)[mask] == yte[mask]).float().mean().item()
                norm = logits.norm(dim=-1).mean().item()
            accs.append((acc, norm))
        print(f"{n_eval:>7} {100*accs[0][0]:>13.1f}%  {100*accs[1][0]:>13.1f}%"
              f"   (logit norms {accs[0][1]:.1f} / {accs[1][1]:.1f})")


if __name__ == "__main__":
    main()
