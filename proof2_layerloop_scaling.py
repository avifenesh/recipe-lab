"""Claim 2 (OURS, novel): layer-loop has *stronger* within-reuse correlation
than model-loop, because consecutive applications see nearly identical h.
Therefore the Theta(N^2) accumulation constant is larger for layer-loop,
and the eps = lambda/(N sqrt(L)) rule is at least as necessary.

Test A: mean pairwise cosine of the two residuals produced by the same layer,
        layer-loop vs model-loop, L unique layers, N=2..8 loops.
Test B: residual-stream norm growth ||h_final|| under eps=1 for both schedules,
        vs eps=1/(N sqrt(L)) — does scaling bound both?
"""
import numpy as np
rng = np.random.default_rng(1)
d = 512

def build(L):
    return [rng.normal(size=(d,d))/np.sqrt(d) for _ in range(L)]

def orders(L, N, kind):
    if kind == "layer":
        return [i for i in range(L) for _ in range(N)]
    return [i for _ in range(N) for i in range(L)]

def run(order, Ws, h0, eps):
    h = h0.copy(); recs = []
    for li in order:
        r = Ws[li] @ np.maximum(h, 0.0)
        recs.append((li, r))
        h = h + eps * r
    return recs, h

def same_layer_cos(recs, L):
    """mean cosine between residuals from the same layer across its reuses"""
    from collections import defaultdict
    by = defaultdict(list)
    for li, r in recs: by[li].append(r)
    cs = []
    for li, rs in by.items():
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                cs.append(rs[i] @ rs[j] / (np.linalg.norm(rs[i])*np.linalg.norm(rs[j])))
    return np.mean(cs)

L = 6
print("Test A: same-layer residual cosine (higher = more constructive accumulation)")
print(f"{'N':>3} {'layer-loop':>12} {'model-loop':>12}")
for N in [2, 4, 8]:
    Ws = build(L); h0 = rng.normal(size=d)
    eps = 1.0/(N*np.sqrt(L))
    ca = same_layer_cos(run(orders(L,N,'layer'), Ws, h0, eps)[0], L)
    cb = same_layer_cos(run(orders(L,N,'model'), Ws, h0, eps)[0], L)
    print(f"{N:>3} {ca:>12.4f} {cb:>12.4f}")

print("\nTest B: final norm R = ||h||/sqrt(d), eps=1 (unscaled) vs eps=1/(N sqrt(L))")
print(f"{'N':>3} {'schedule':>8} {'eps=1':>12} {'eps=1/(N*rtL)':>14}")
for N in [2, 4, 8]:
    for kind in ["layer", "model"]:
        Ws = build(L); h0 = rng.normal(size=d)
        _, h1 = run(orders(L,N,kind), Ws, h0, 1.0)
        _, h2 = run(orders(L,N,kind), Ws, h0, 1.0/(N*np.sqrt(L)))
        print(f"{N:>3} {kind:>8} {np.linalg.norm(h1)/np.sqrt(d):>12.2e} {np.linalg.norm(h2)/np.sqrt(d):>14.3f}")
