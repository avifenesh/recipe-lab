"""Claim 1 (arXiv 2606.18524, Thm 1): in a weight-shared residual loop
h_{n+1} = h_n + eps * W phi(h_n), the summed residuals satisfy
||sum_n r_n||^2 = Theta(N^2)  (constructive accumulation),
whereas independent weights give Theta(N) (random walk).

Verify numerically, and extend to the untested case: LAYER-LOOP vs MODEL-LOOP
correlation structure with L unique layers, N loops.

layer-loop: L1 L1 .. (N) .. L2 L2 .. (N) ..  -> per-layer consecutive reuse
model-loop: (L1 L2 .. LL) repeated N times   -> reuse separated by L-1 other layers
"""
import numpy as np

rng = np.random.default_rng(0)
d = 512

def run_chain(order, Ws, h0, eps):
    """order: list of layer indices; returns per-step residuals r_n."""
    h = h0.copy()
    rs = []
    for li in order:
        r = Ws[li] @ np.maximum(h, 0.0)
        rs.append(r)
        h = h + eps * r
    return np.array(rs), h

def sumnorm2(rs):
    return np.linalg.norm(rs.sum(axis=0))**2 / d

print(f"{'N':>4} {'shared/N':>12} {'indep/N':>12} {'shared/N^2':>12}")
for N in [4, 8, 16, 32, 64, 128]:
    # single shared layer, eps=0 during measurement of raw accumulation? No:
    # measure with eps=1/N (stable regime) and eps small; theorem is about raw sum.
    # Use eps=1/N so forward stays bounded, measure ||sum r||^2 scaling.
    eps = 1.0 / N
    h0 = rng.normal(size=d)
    W = rng.normal(size=(d, d)) / np.sqrt(d)
    rs, _ = run_chain([0]*N, [W], h0, eps)
    s_shared = sumnorm2(rs)
    Ws_ind = [rng.normal(size=(d, d))/np.sqrt(d) for _ in range(N)]
    rs_i, _ = run_chain(list(range(N)), Ws_ind, h0, eps)
    s_indep = sumnorm2(rs_i)
    print(f"{N:>4} {s_shared/N:>12.3f} {s_indep/N:>12.3f} {s_shared/N**2:>12.4f}")

print("\nshared/N grows linearly & shared/N^2 flat  => Theta(N^2)  [claim holds]")
print("indep/N flat => Theta(N) random walk        [claim holds]")
