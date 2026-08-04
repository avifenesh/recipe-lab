"""Minimal GPT with configurable block-reuse schedule and residual scaling.

Arms:
  vanilla     — n_stored blocks, each applied once (effective depth = n_stored)
  layer-loop  — n_stored blocks, each applied n_loop times consecutively
  model-loop  — n_stored-block stack traversed n_loop times

`res_scale` multiplies every residual branch (attn and mlp). The scaled arms
use 1/n_loop, targeting exactly the correlated-accumulation term that weight
sharing introduces (arXiv 2606.18524 Thm 1); the vanilla arm keeps 1.0 so the
comparison isolates the loop-correlation fix and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304  # GPT-2 BPE rounded up to /64
    block_size: int = 1024
    d_model: int = 384
    n_head: int = 6
    n_stored: int = 12   # unique transformer blocks
    n_loop: int = 1      # applications per block (schedule decides ordering)
    schedule: str = "vanilla"  # vanilla | layer | model
    res_scale: float = 1.0
    learn_lambda: bool = False  # per-block learnable multiplier on res_scale
    mixer: str = "attn"  # attn | ssm (selective scan, Mamba-style) | hybrid
    attn_every: int = 6  # hybrid: block i is attention iff i % attn_every == attn_offset
    attn_offset: int = 0
    d_state: int = 16    # SSM state size (mixer="ssm"/"hybrid")
    dropout: float = 0.0


class SelectiveSSM(nn.Module):
    """Minimal S6-style selective scan: input-dependent step size, diagonal
    state decay, gated output. Exact scan via log-space cumulative products."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d, ds = cfg.d_model, cfg.d_state
        # depthwise causal conv before selection, as in Mamba (d_conv=4)
        self.conv = nn.Conv1d(d, d, kernel_size=4, padding=3, groups=d)
        self.W_dt = nn.Linear(d, 1)
        self.a_log = nn.Parameter(torch.zeros(ds))
        self.B = nn.Linear(d, ds, bias=False)
        self.C = nn.Linear(ds, d, bias=False)
        self.D = nn.Parameter(torch.zeros(d))
        self.gate = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # x: (B, T, D). s_t = exp(decay_t) s_{t-1} + u_t, causal by cumsum.
        # Scan runs in fp32: u/P spans up to e^30, far past bf16 range.
        T = x.shape[1]
        xc = F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))
        dt = F.softplus(self.W_dt(xc)).float()             # (B, T, 1)
        decay = -F.softplus(self.a_log.float())[None, None] * dt
        u = dt * self.B(xc).float()                        # (B, T, ds)
        logP = decay.cumsum(dim=1).clamp(min=-30.0)        # running log-decay
        P = logP.exp()
        s = P * (u / P).cumsum(dim=1)
        y = self.C(s.to(x.dtype)) + xc * self.D
        return self.out(y * F.silu(self.gate(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, use_attn: bool = True):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        if not use_attn:
            self.attn = None
            self.ssm = SelectiveSSM(cfg)
        else:
            self.attn = nn.MultiheadAttention(d, cfg.n_head,
                                              dropout=cfg.dropout,
                                              bias=True, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d),
        )
        # lambda in eps = lambda/(N sqrt(L)): learnable per block, init 1
        self.lam = (nn.Parameter(torch.ones(())) if cfg.learn_lambda
                    else None)

    def forward(self, x, attn_mask, res_scale: float):
        s = res_scale if self.lam is None else res_scale * self.lam
        h = self.ln1(x)
        if self.attn is None:
            a = self.ssm(h)
        else:
            a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False,
                             is_causal=True)
        x = x + s * a
        x = x + s * self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)

        def block_is_attn(i: int) -> bool:
            if cfg.mixer == "attn":
                return True
            if cfg.mixer == "ssm":
                return False
            return i % cfg.attn_every == cfg.attn_offset  # hybrid
        self.blocks = nn.ModuleList(Block(cfg, use_attn=block_is_attn(i))
                                    for i in range(cfg.n_stored))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied

        mask = torch.full((cfg.block_size, cfg.block_size), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("attn_mask", mask, persistent=False)

        self.apply(self._init)
        # GPT-2 residual-projection scaling by *effective* depth so all arms
        # share identical init statistics per applied block
        eff_depth = self.effective_depth()
        for n, p in self.named_parameters():
            if n.endswith(("mlp.2.weight", "attn.out_proj.weight",
                           "ssm.out.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * eff_depth))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def effective_depth(self) -> int:
        return self.cfg.n_stored * self.cfg.n_loop

    def block_order(self):
        c = self.cfg
        if c.schedule == "vanilla":
            assert c.n_loop == 1
            return list(range(c.n_stored))
        if c.schedule == "layer":
            return [i for i in range(c.n_stored) for _ in range(c.n_loop)]
        if c.schedule == "model":
            return [i for _ in range(c.n_loop) for i in range(c.n_stored)]
        raise ValueError(c.schedule)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = self.attn_mask[:t, :t]
        for li in self.block_order():
            x = self.blocks[li](x, mask, self.cfg.res_scale)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos_emb.weight.numel()
