"""Hybrid (Mamba-2 SSD + GQA attention) MoE stack, with the campaign's loop
schedule carried over.

Why this shape: the 2026 field converged on the same backbone from three
independent directions (Qwen3-Next/Qwen3.5 Gated-DeltaNet 3:1, Kimi Linear
KDA ~3:1, Nemotron-3 Mamba-2 ~4:1) — a minority of softmax-attention layers
for content lookup, a majority of fixed-state recurrent layers, MoE for the
feed-forward. That is exactly the substrate the campaign's design rule speaks
about: *loop the state-mixer, never the retriever* (FINDINGS round 11c —
HE768 < HA768 on 2/2 seeds, while looping attention lost by +0.25 to +0.28).

Arms (matched applications = matched active FLOPs per token):
  HV: every block stored once. 5 groups of [s,s,s,s,a] -> 25 applications.
  HL: linear blocks stored once and applied twice, attention applied once.
      5 groups of [s,s,a] -> the same 25 applications from 15 stored blocks.
Both arms emit the identical *sequence of block types* (s,s,s,s,a per group),
so attention placement is exactly matched — the control that round 11b showed
is necessary (HC768 failed because looping stacked the attention
applications; HD768 restored placement and still lost, isolating the mixer
identity rather than placement as the cause).

res_scale is applied globally, as in the validated HE768 configuration.

The SSD mixer is Mamba-2's state-space-dual chunkwise form: exact for the
scalar-decay recurrence S_t = a_t S_{t-1} + dt_t B_t x_t^T, computed in
O(T) with chunk-local quadratic blocks. Validated against the naive
recurrence in test_ssd.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfigMoE:
    vocab_size: int = 50304
    block_size: int = 4096
    d_model: int = 1024
    # --- attention (retriever) ---
    n_head: int = 16
    n_kv_head: int = 4          # GQA
    rope_theta: float = 10000.0
    # --- SSD (state mixer) ---
    ssm_expand: int = 2         # d_inner = expand * d_model
    ssm_head_dim: int = 64      # P
    d_state: int = 64           # N, shared across heads (ngroups=1)
    d_conv: int = 4
    ssd_chunk: int = 128
    # --- MoE feed-forward ---
    n_expert: int = 56
    n_expert_active: int = 10
    n_shared_expert: int = 1
    d_expert: int = 256
    capacity_factor: float = 2.0     # measured: 1.25 dropped ~20% of routes
    bias_update_rate: float = 1e-2   # aux-loss-free balancing step
    # --- stack / loop schedule ---
    n_group: int = 5            # groups of [s,s,s,s,a] applications
    ssm_per_group: int = 4      # linear applications per attention application
    n_loop: int = 1             # applications per stored linear block
    res_scale: float = 1.0


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


def rope_cache(seq_len, head_dim, theta, device="cpu"):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                           / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # stack+flatten keeps the result contiguous so SDPA can pick flash
    T = x.shape[2]
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


class Attention(nn.Module):
    """GQA — the retriever. Never looped (round 11b/11c)."""

    def __init__(self, cfg: ConfigMoE):
        super().__init__()
        d = cfg.d_model
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.head_dim = d // cfg.n_head
        self.wq = nn.Linear(d, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=self.n_kv != self.n_head)
        return self.wo(a.transpose(1, 2).contiguous().view(B, T, -1))


def ssd(x, dt, A, Bm, Cm, chunk):
    """Mamba-2 state-space dual, chunkwise-exact.

    Recurrence per head h:  S_t = exp(dt_t A_h) S_{t-1} + dt_t B_t x_t^T
                            y_t = C_t^T S_t
    x  : (B, H, T, P)   values
    dt : (B, H, T)      positive step size
    A  : (H,)           negative decay rate
    Bm, Cm : (B, T, N)  shared across heads (ngroups=1)
    returns (B, H, T, P)

    Decay algebra runs in fp32: the running log-decay spans far more than
    bf16's exponent range (same reason the round-6 scan was forced to fp32).
    """
    B, H, T, P = x.shape
    N = Bm.shape[-1]
    pad = (-T) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        dt = F.pad(dt, (0, pad))
        Bm = F.pad(Bm, (0, 0, 0, pad))
        Cm = F.pad(Cm, (0, 0, 0, pad))
    Tp = T + pad
    NC, L = Tp // chunk, chunk

    xc = x.view(B, H, NC, L, P).float()
    dtc = dt.view(B, H, NC, L).float()
    Bc = Bm.view(B, NC, L, N).float()
    Cc = Cm.view(B, NC, L, N).float()

    dA = dtc * A.view(1, H, 1, 1).float()          # (B,H,NC,L), <= 0
    dA_cum = dA.cumsum(dim=-1)

    # intra-chunk: y_i = sum_{j<=i} (C_i.B_j) exp(dA_cum_i - dA_cum_j) dt_j x_j
    # Causal entries (i >= j) always have seg <= 0 because dA <= 0, so
    # clamp(max=0) is a no-op there and only tames the masked-out upper
    # triangle. Masking by multiplication rather than torch.where matters:
    # where() still differentiates the untaken branch, and exp() of the
    # positive upper-triangle exponents overflows to inf, making the
    # gradient inf*0 = NaN even though the forward value is correct.
    seg = (dA_cum.unsqueeze(-1) - dA_cum.unsqueeze(-2)).clamp(min=-30.0, max=0.0)
    causal = torch.ones(L, L, device=x.device, dtype=torch.float32).tril()
    decay = seg.exp() * causal
    CB = torch.einsum("bcln,bcmn->bclm", Cc, Bc)   # (B,NC,L,L)
    M = CB.unsqueeze(1) * decay * dtc.unsqueeze(-2)
    y = torch.einsum("bhclm,bhcmp->bhclp", M, xc)

    # inter-chunk: carry the state across chunks. Weight for absorbing token j
    # into the end-of-chunk state is exp(dA_cum_last - dA_cum_j) <= 0 in the
    # exponent, so it never grows.
    S = x.new_zeros(B, H, N, P, dtype=torch.float32)
    w_in = (dA_cum[..., -1:] - dA_cum).clamp(min=-30.0).exp() * dtc  # (B,H,NC,L)
    outs = []
    for c in range(NC):
        y_inter = torch.einsum("bln,bhnp->bhlp", Cc[:, c], S)
        outs.append(y[:, :, c] + y_inter * dA_cum[:, :, c].clamp(min=-30.0)
                    .exp().unsqueeze(-1))
        S = (S * dA_cum[:, :, c, -1].clamp(min=-30.0).exp()[..., None, None]
             + torch.einsum("bln,bhlp->bhnp", Bc[:, c],
                            xc[:, :, c] * w_in[:, :, c].unsqueeze(-1)))
    out = torch.stack(outs, dim=2).view(B, H, Tp, P)
    return out[:, :, :T].to(x.dtype)


class SSDMixer(nn.Module):
    """Mamba-2 style state mixer — the block that gets looped."""

    def __init__(self, cfg: ConfigMoE):
        super().__init__()
        d = cfg.d_model
        self.d_inner = cfg.ssm_expand * d
        self.P = cfg.ssm_head_dim
        self.H = self.d_inner // self.P
        self.N = cfg.d_state
        self.chunk = cfg.ssd_chunk
        self.in_x = nn.Linear(d, self.d_inner, bias=False)
        self.in_z = nn.Linear(d, self.d_inner, bias=False)
        self.in_BC = nn.Linear(d, 2 * self.N, bias=False)
        self.in_dt = nn.Linear(d, self.H, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, cfg.d_conv,
                              padding=cfg.d_conv - 1, groups=self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.H + 1).float()))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm = RMSNorm(self.d_inner)
        self.out = nn.Linear(self.d_inner, d, bias=False)

    def forward(self, u):
        B, T, _ = u.shape
        x = self.in_x(u)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))
        x = x.view(B, T, self.H, self.P).transpose(1, 2)          # (B,H,T,P)
        dt = F.softplus(self.in_dt(u) - 4.0).transpose(1, 2)      # (B,H,T)
        Bm, Cm = self.in_BC(u).split(self.N, dim=-1)
        A = -self.A_log.float().exp()
        y = ssd(x, dt, A, Bm, Cm, self.chunk)
        y = y + x * self.D.view(1, self.H, 1, 1)
        y = y.transpose(1, 2).reshape(B, T, self.d_inner)
        y = self.norm(y) * F.silu(self.in_z(u))
        return self.out(y)


class MoEFFN(nn.Module):
    """Fine-grained SwiGLU experts + always-on shared expert.

    Routing: sigmoid scores, top-k selected on score+bias, combined with the
    unbiased scores (DeepSeek-V3 style). The bias is a buffer nudged toward
    even load after every step — auxiliary-loss-free balancing, so no
    interference gradient is added to the loss (arXiv 2408.15664).

    Dispatch is capacity-based so every shape is static: tokens beyond an
    expert's capacity keep only the shared-expert path. The balancer is what
    keeps that drop rate small; drop_frac is logged so it stays honest.
    """

    def __init__(self, cfg: ConfigMoE):
        super().__init__()
        d, E, dE = cfg.d_model, cfg.n_expert, cfg.d_expert
        self.E, self.k, self.dE = E, cfg.n_expert_active, dE
        self.cap_factor = cfg.capacity_factor
        self.router = nn.Linear(d, E, bias=False)
        # initialized here so the module is valid standalone; HybridMoE
        # re-inits w2 with the depth-scaled std once it knows the schedule
        self.w1 = nn.Parameter(torch.randn(E, d, dE) * 0.02)
        self.w3 = nn.Parameter(torch.randn(E, d, dE) * 0.02)
        self.w2 = nn.Parameter(torch.randn(E, dE, d) * 0.02)
        self.n_shared = cfg.n_shared_expert
        if self.n_shared:
            ds = dE * self.n_shared
            self.s1 = nn.Linear(d, ds, bias=False)
            self.s3 = nn.Linear(d, ds, bias=False)
            self.s2 = nn.Linear(ds, d, bias=False)
        self.register_buffer("expert_bias", torch.zeros(E), persistent=True)
        # counts accumulate across micro-batches: with grad-accum the balancer
        # must see the whole optimizer step's routing, not just the last micro
        # batch (1/accum of the tokens made it chase noise and stall at a ~20%
        # drop rate). rebalance() consumes and clears them.
        self.register_buffer("load_count", torch.zeros(E), persistent=False)
        self.register_buffer("route_count", torch.zeros(()), persistent=False)
        self.register_buffer("drop_count", torch.zeros(()), persistent=False)
        self.register_buffer("drop_frac", torch.zeros(()), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        n = xf.shape[0]
        scores = torch.sigmoid(self.router(xf).float())            # (n,E)
        sel = torch.topk(scores + self.expert_bias, self.k, dim=-1).indices
        w = scores.gather(-1, sel)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)            # (n,k)

        cap = max(1, int(self.cap_factor * self.k * n / self.E))
        flat_e = sel.reshape(-1)                                   # (n*k,)
        tok = torch.arange(n, device=x.device).repeat_interleave(self.k)
        order = torch.argsort(flat_e, stable=True)
        e_sorted = flat_e[order]
        counts = torch.zeros(self.E, device=x.device, dtype=torch.long)
        counts.index_add_(0, e_sorted, torch.ones_like(e_sorted))
        starts = counts.cumsum(0) - counts
        rank = torch.arange(n * self.k, device=x.device) - starts[e_sorted]
        keep = rank < cap

        with torch.no_grad():
            self.load_count += counts.float()
            self.route_count += n * self.k
            self.drop_count += (~keep).sum().float()
            self.drop_frac.copy_(self.drop_count / self.route_count.clamp(min=1))

        slot = torch.zeros(self.E * cap, device=x.device, dtype=torch.long)
        wslot = torch.zeros(self.E * cap, device=x.device, dtype=w.dtype)
        pos = e_sorted[keep] * cap + rank[keep]
        slot[pos] = tok[order][keep]
        wslot[pos] = w.reshape(-1)[order][keep]

        xe = xf[slot].view(self.E, cap, D)
        h = F.silu(torch.bmm(xe, self.w1)) * torch.bmm(xe, self.w3)
        ye = torch.bmm(h, self.w2).reshape(self.E * cap, D)
        ye = ye * wslot.to(ye.dtype)[:, None]
        # empty slots carry token 0's weight-0 output, so index_add is safe.
        # under autocast ye is bf16 while xf may be fp32 — index_add_ demands
        # a matching scalar type, so accumulate in ye's dtype.
        out = torch.zeros(xf.shape, device=xf.device, dtype=ye.dtype)
        out.index_add_(0, slot, ye)
        out = out.to(xf.dtype)

        if self.n_shared:
            out = out + self.s2(F.silu(self.s1(xf)) * self.s3(xf))
        return out.view(B, T, D)

    @torch.no_grad()
    def rebalance(self, rate):
        """Aux-loss-free step: overloaded experts get a lower selection bias.

        Consumes the counts accumulated over this optimizer step's
        micro-batches (and all ranks), then clears them.
        """
        counts, total = self.load_count, self.route_count.clone()
        if dist.is_available() and dist.is_initialized():
            counts = counts.clone()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if total.item() > 0:
            load = counts / total
            err = load - 1.0 / self.E
            self.expert_bias -= rate * torch.sign(err)
        self.load_count.zero_()
        self.route_count.zero_()
        self.drop_count.zero_()

    @property
    def load(self):
        """Share of routes each expert took since the last rebalance."""
        return self.load_count / self.route_count.clamp(min=1)


class HybridBlock(nn.Module):
    def __init__(self, cfg: ConfigMoE, is_attn: bool):
        super().__init__()
        self.is_attn = is_attn
        self.ln1 = RMSNorm(cfg.d_model)
        self.mixer = Attention(cfg) if is_attn else SSDMixer(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = MoEFFN(cfg)

    def forward(self, x, cos, sin, res_scale):
        h = self.ln1(x)
        m = self.mixer(h, cos, sin) if self.is_attn else self.mixer(h)
        x = x + res_scale * m
        x = x + res_scale * self.ffn(self.ln2(x))
        return x


class HybridMoE(nn.Module):
    def __init__(self, cfg: ConfigMoE):
        super().__init__()
        self.cfg = cfg
        # stored blocks: per group, ssm_per_group/n_loop linear blocks then one
        # attention block. Both arms therefore *apply* the same type sequence.
        assert cfg.ssm_per_group % cfg.n_loop == 0, "loop must divide the group"
        n_ssm_stored = cfg.ssm_per_group // cfg.n_loop
        types = []
        for _ in range(cfg.n_group):
            types += [False] * n_ssm_stored + [True]
        self.types = types
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(HybridBlock(cfg, t) for t in types)
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_head,
                              cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        for m in self.modules():          # expert weights: fan-in init
            if isinstance(m, MoEFFN):
                for p in (m.w1, m.w3, m.w2):
                    nn.init.normal_(p, mean=0.0, std=0.02)
        eff = self.n_applications()
        for n, p in self.named_parameters():
            if n.endswith(("wo.weight", "out.weight", "s2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * eff))
        for m in self.modules():
            if isinstance(m, MoEFFN):
                nn.init.normal_(m.w2, mean=0.0, std=0.02 / math.sqrt(2 * eff))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_applications(self):
        c = self.cfg
        return c.n_group * (c.ssm_per_group + 1)

    def block_order(self):
        """Loop the state-mixer, never the retriever (FINDINGS rule 6)."""
        order = []
        for i, is_attn in enumerate(self.types):
            order += [i] * (1 if is_attn else self.cfg.n_loop)
        return order

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

    def rebalance(self):
        for m in self.modules():
            if isinstance(m, MoEFFN):
                m.rebalance(self.cfg.bias_update_rate)

    def drop_frac(self):
        fr = [m.drop_frac.item() for m in self.modules()
              if isinstance(m, MoEFFN)]
        return sum(fr) / max(1, len(fr))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_active_params(self):
        """Params touched per token: everything except unrouted experts."""
        total = self.num_params()
        for m in self.modules():
            if isinstance(m, MoEFFN):
                per = m.w1[0].numel() + m.w3[0].numel() + m.w2[0].numel()
                total -= per * (m.E - m.k)
        return total


# HV / HL differ only in the loop: identical applications, identical block-type
# sequence, HL stores half the linear blocks. Sizes chosen so HV's total lands
# on the dense factorial's V arm (1.30B) for a comparable table row.
ARMS_MOE = {
    "HV": dict(n_loop=1, res_scale=1.0),
    "HL": dict(n_loop=2, res_scale=0.5),
}

SIZES_MOE = {
    # sign-check rung: the d=768 regime where the flip was verified 3/3 seeds
    "s": dict(d_model=768, n_head=12, n_kv_head=4, block_size=1024,
              n_expert=32, n_expert_active=6, d_expert=192, d_state=64,
              n_group=3, ssm_per_group=4, ssd_chunk=128),
    # 1B rung
    "b": dict(d_model=1024, n_head=16, n_kv_head=4, block_size=4096,
              n_expert=56, n_expert_active=10, d_expert=256, d_state=64,
              n_group=5, ssm_per_group=4, ssd_chunk=128),
}


def build(arm: str, size: str, **over) -> HybridMoE:
    kw = dict(SIZES_MOE[size])
    kw.update(ARMS_MOE[arm])
    kw.update(over)
    return HybridMoE(ConfigMoE(**kw))
