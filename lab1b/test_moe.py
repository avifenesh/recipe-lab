"""Correctness tests for the hybrid-MoE arch, runnable on CPU.

The SSD chunkwise form is the piece that can silently be wrong (a bad decay
exponent still trains, just to a worse loss), so it is checked against the
naive recurrence at several chunk sizes including ones that don't divide T.

Run: python test_moe.py
"""

from __future__ import annotations

import torch

from model_moe import ARMS_MOE, ConfigMoE, HybridMoE, MoEFFN, build, ssd


def naive_ssd(x, dt, A, Bm, Cm):
    """S_t = exp(dt_t A) S_{t-1} + dt_t B_t x_t^T ; y_t = C_t^T S_t."""
    B, H, T, P = x.shape
    N = Bm.shape[-1]
    S = torch.zeros(B, H, N, P, dtype=torch.float64)
    out = torch.zeros(B, H, T, P, dtype=torch.float64)
    xd, dtd = x.double(), dt.double()
    Ad, Bd, Cd = A.double(), Bm.double(), Cm.double()
    for t in range(T):
        decay = (dtd[:, :, t] * Ad[None, :]).exp()          # (B,H)
        S = S * decay[..., None, None]
        S = S + torch.einsum("bn,bhp->bhnp", Bd[:, t],
                             xd[:, :, t] * dtd[:, :, t][..., None])
        out[:, :, t] = torch.einsum("bn,bhnp->bhp", Cd[:, t], S)
    return out


def test_ssd_matches_recurrence():
    torch.manual_seed(0)
    B, H, T, P, N = 2, 3, 37, 8, 6
    x = torch.randn(B, H, T, P)
    dt = torch.rand(B, H, T) * 0.5 + 0.01
    A = -torch.rand(H) * 2 - 0.1
    Bm = torch.randn(B, T, N) * 0.5
    Cm = torch.randn(B, T, N) * 0.5
    want = naive_ssd(x, dt, A, Bm, Cm)
    # the scan runs in fp32 by design (the running log-decay overflows bf16),
    # so the floor here is fp32 roundoff ~1e-7, not the algorithm
    for chunk in (1, 4, 8, 16, 37, 64):
        got = ssd(x, dt, A, Bm, Cm, chunk).double()
        err = (got - want).abs().max().item()
        scale = want.abs().max().item()
        assert err / scale < 1e-5, f"chunk={chunk} rel err {err/scale:.3e}"
        print(f"  ssd chunk={chunk:3d} max rel err {err/scale:.2e}  OK")

    # chunk size must not change the result beyond that fp32 floor
    ref = ssd(x, dt, A, Bm, Cm, 8)
    for chunk in (4, 16, 64):
        d = (ssd(x, dt, A, Bm, Cm, chunk) - ref).abs().max().item()
        assert d / ref.abs().max().item() < 1e-5, f"chunk {chunk} drifts {d:.2e}"
    print("  ssd chunk-invariance OK")


def test_ssd_causal():
    """Changing token t must not move outputs before t."""
    torch.manual_seed(1)
    B, H, T, P, N = 1, 2, 24, 4, 5
    x = torch.randn(B, H, T, P)
    dt = torch.rand(B, H, T) * 0.4 + 0.05
    A = -torch.rand(H) - 0.2
    Bm, Cm = torch.randn(B, T, N), torch.randn(B, T, N)
    y0 = ssd(x, dt, A, Bm, Cm, 8)
    x2 = x.clone()
    x2[:, :, 12] += 5.0
    y1 = ssd(x2, dt, A, Bm, Cm, 8)
    assert torch.allclose(y0[:, :, :12], y1[:, :, :12], atol=1e-6)
    assert not torch.allclose(y0[:, :, 12], y1[:, :, 12])
    print("  ssd causality OK")


def test_ssd_grads_finite():
    """The forward can be right while the backward is NaN.

    Masking the non-causal block with torch.where still differentiates the
    untaken branch, and those exponents overflow to inf, so the gradient
    becomes inf*0 = NaN while the printed loss looks healthy. Guard it.
    """
    torch.manual_seed(3)
    B, H, T, P, N = 2, 3, 40, 8, 6
    x = torch.randn(B, H, T, P, requires_grad=True)
    dt_raw = torch.randn(B, H, T, requires_grad=True)
    A_log = torch.randn(H, requires_grad=True)
    Bm = torch.randn(B, T, N, requires_grad=True)
    Cm = torch.randn(B, T, N, requires_grad=True)
    for chunk in (8, 16):
        dt = torch.nn.functional.softplus(dt_raw)
        y = ssd(x, dt, -A_log.exp(), Bm, Cm, chunk)
        y.square().mean().backward()
        for name, t in (("x", x), ("dt", dt_raw), ("A_log", A_log),
                        ("B", Bm), ("C", Cm)):
            assert t.grad is not None and torch.isfinite(t.grad).all(), \
                f"chunk={chunk}: {name} grad non-finite"
            t.grad = None
    print("  ssd gradients finite OK")


def test_moe_routing():
    torch.manual_seed(2)
    cfg = ConfigMoE(d_model=64, n_expert=8, n_expert_active=2, d_expert=16,
                    n_shared_expert=1)
    ffn = MoEFFN(cfg)
    x = torch.randn(4, 16, 64)
    y = ffn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert abs(ffn.load.sum().item() - 1.0) < 1e-5, ffn.load.sum()
    print(f"  moe forward OK, drop_frac={ffn.drop_frac.item():.4f}")

    # the balancer must push the bias against overload
    ffn.load.zero_()
    ffn.load[0] = 1.0
    before = ffn.expert_bias.clone()
    ffn.rebalance(1e-2)
    assert ffn.expert_bias[0] < before[0], "overloaded expert bias must drop"
    assert ffn.expert_bias[1] > before[1], "idle expert bias must rise"
    print("  moe aux-loss-free balancer OK")

    # capacity drop must still return finite output for every token
    cfg2 = ConfigMoE(d_model=32, n_expert=4, n_expert_active=1, d_expert=8,
                     n_shared_expert=1, capacity_factor=0.25)
    f2 = MoEFFN(cfg2)
    with torch.no_grad():          # force all tokens to expert 0
        f2.router.weight.zero_()
        f2.router.weight[0] = 10.0
    y2 = f2(torch.randn(2, 32, 32))
    assert torch.isfinite(y2).all()
    assert f2.drop_frac.item() > 0.5, f2.drop_frac.item()
    print(f"  moe overflow path OK, drop_frac={f2.drop_frac.item():.3f}")


def test_arms_matched_applications():
    """HV and HL must apply the same count and the same type sequence."""
    small = dict(d_model=64, n_head=4, n_kv_head=2, block_size=128,
                 n_expert=8, n_expert_active=2, d_expert=16, d_state=16,
                 n_group=2, ssm_per_group=4, ssd_chunk=32)
    hv = HybridMoE(ConfigMoE(**small, **ARMS_MOE["HV"]))
    hl = HybridMoE(ConfigMoE(**small, **ARMS_MOE["HL"]))
    seq_v = [hv.types[i] for i in hv.block_order()]
    seq_l = [hl.types[i] for i in hl.block_order()]
    assert seq_v == seq_l, (seq_v, seq_l)
    assert len(seq_v) == hv.n_applications() == 10
    assert len(hl.blocks) < len(hv.blocks)
    # attention applied exactly once per group in both arms
    assert sum(seq_v) == 2
    print(f"  arms matched: {len(seq_v)} applications, "
          f"stored HV={len(hv.blocks)} HL={len(hl.blocks)}, "
          f"types {''.join('a' if t else 's' for t in seq_v)}")

    # attention blocks are never looped in either arm
    for m in (hv, hl):
        counts = {}
        for i in m.block_order():
            counts[i] = counts.get(i, 0) + 1
        for i, c in counts.items():
            if m.types[i]:
                assert c == 1, f"attention block {i} applied {c}x"
    print("  retriever never looped OK")


def test_forward_backward():
    m = build("HL", "s", d_model=128, n_head=4, n_kv_head=2, block_size=64,
              n_expert=8, n_expert_active=2, d_expert=32, n_group=2,
              ssd_chunk=16)
    idx = torch.randint(0, 1000, (2, 64))
    _, loss = m(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    m.rebalance()
    print(f"  fwd/bwd OK loss={loss.item():.3f} "
          f"total={m.num_params()/1e6:.1f}M "
          f"active={m.num_active_params()/1e6:.1f}M "
          f"drop={m.drop_frac():.4f}")


def test_train_steps_stable():
    """A few real optimizer steps: loss and every parameter stay finite.

    test_forward_backward passing is not enough — the NaN that motivated
    test_ssd_grads_finite showed up only after opt.step() propagated a
    non-finite gradient into the weights.
    """
    torch.manual_seed(4)
    m = build("HL", "s", d_model=96, n_head=4, n_kv_head=2, block_size=64,
              n_expert=8, n_expert_active=2, d_expert=24, n_group=2,
              ssd_chunk=16)
    opt = torch.optim.AdamW(m.parameters(), lr=6e-4, betas=(0.9, 0.95))
    idx = torch.randint(0, 1000, (2, 64))
    for step in range(5):
        _, loss = m(idx, idx)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        assert torch.isfinite(loss), f"step {step}: loss {loss.item()}"
        assert torch.isfinite(gn), f"step {step}: grad norm {float(gn)}"
        opt.step()
        opt.zero_grad(set_to_none=True)
        m.rebalance()
        bad = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
        assert not bad, f"step {step}: non-finite params {bad[:3]}"
    print(f"  5 train steps stable OK (loss {loss.item():.3f}, "
          f"gnorm {float(gn):.2e})")


if __name__ == "__main__":
    torch.set_num_threads(4)
    print("SSD:")
    test_ssd_matches_recurrence()
    test_ssd_causal()
    test_ssd_grads_finite()
    print("MoE:")
    test_moe_routing()
    print("Arms:")
    test_arms_matched_applications()
    test_forward_backward()
    test_train_steps_stable()
    print("ALL PASS")
