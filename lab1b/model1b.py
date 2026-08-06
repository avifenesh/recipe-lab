"""Modernized GPT for the 1B curriculum x loop factorial.

Llama-style: RoPE, RMSNorm, SwiGLU, no biases, tied head. Keeps the
recipe-lab loop machinery: layer-loop schedule with res_scale eps on every
residual branch. seq 4096.

Arms (SPEC_1B_CURRICULUM):
  V: n_stored=24, n_loop=1, eps=1.0            (~1.4B params)
  L: n_stored=12, n_loop=2, eps=1/(2*sqrt(12)) (~0.9B params)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config1B:
    vocab_size: int = 50304
    block_size: int = 4096
    d_model: int = 2048
    n_head: int = 16
    n_stored: int = 24
    n_loop: int = 1
    res_scale: float = 1.0
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def rope_cache(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                           / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh). stack+flatten keeps the result contiguous so SDPA
    # dispatches to flash attention (strided writes forced the math backend:
    # O(T^2) memory + ~10x slower at seq 4096).
    T = x.shape[2]
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: Config1B):
        super().__init__()
        d, h = cfg.d_model, cfg.n_head
        self.n_head, self.head_dim = h, d // h
        self.ln1 = RMSNorm(d)
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.ln2 = RMSNorm(d)
        hidden = 256 * round(int(8 * d / 3) / 256)
        self.w1 = nn.Linear(d, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, d, bias=False)   # down

    def forward(self, x, cos, sin, res_scale):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.wqkv(h).split(D, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + res_scale * self.wo(a)
        h = self.ln2(x)
        x = x + res_scale * self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class GPT1B(nn.Module):
    def __init__(self, cfg: Config1B):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_stored))
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_head,
                              cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        eff_depth = cfg.n_stored * cfg.n_loop
        for n, p in self.named_parameters():
            if n.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * eff_depth))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def block_order(self):
        c = self.cfg
        return [i for i in range(c.n_stored) for _ in range(c.n_loop)]

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos, self.rope_sin
        for li in self.block_order():
            x = self.blocks[li](x, cos, sin, self.cfg.res_scale)
        x = self.ln_f(x)
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            return logits, loss
        return self.head(x[:, -1:]), None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# eps=1/N exactly as validated at every rung (the sqrt(L) across-layer term
# is carried by the depth-scaled init of wo/w2, same as the small-scale arms)
ARMS_1B = {
    "V": dict(n_stored=24, n_loop=1, res_scale=1.0),
    "L": dict(n_stored=12, n_loop=2, res_scale=0.5),
}
