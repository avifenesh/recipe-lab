"""Claim 3: does the accumulation pathology survive pre-LN?
Pre-LN: h <- h + eps * W phi(LN(h)).  ||residual|| = O(1) always,
so no exponential explosion — but growth rate of ||h|| still differs:
- independent weights: ||h_N||^2 ~ h0^2 + c*N   (random walk)
- shared weights:      ||h_N|| ~ c*N            (linear, constructive)
Norm ratio matters because signal from token embedding / attention gets
diluted as 1/||h||. Also check: LR sensitivity via one-step update amplification.
"""
import numpy as np
rng = np.random.default_rng(2)
d = 512

def ln(x):
    m = x.mean(); s = x.std()
    return (x - m) / (s + 1e-6)

def run(order, Ws, h0, eps):
    h = h0.copy()
    for li in order:
        h = h + eps * (Ws[li] @ np.maximum(ln(h), 0.0))
    return h

L = 6
print(f"pre-LN final norm/sqrt(d), L={L} unique layers")
print(f"{'N':>3} {'shared,eps=1':>14} {'indep,eps=1':>13} {'shared,scaled':>14}")
for N in [2, 4, 8, 16]:
    Ws = [rng.normal(size=(d,d))/np.sqrt(d) for _ in range(L)]
    h0 = rng.normal(size=d)
    order_shared = [i for i in range(L) for _ in range(N)]        # layer-loop
    # independent control: unique weights for every step, same total steps
    Wind = [rng.normal(size=(d,d))/np.sqrt(d) for _ in range(L*N)]
    hs = run(order_shared, Ws, h0, 1.0)
    hi = run(list(range(L*N)), Wind, h0, 1.0)
    hs_sc = run(order_shared, Ws, h0, 1.0/(N*np.sqrt(L)))
    n = np.sqrt(d)
    print(f"{N:>3} {np.linalg.norm(hs)/n:>14.2f} {np.linalg.norm(hi)/n:>13.2f} {np.linalg.norm(hs_sc)/n:>14.3f}")

# effective depth steps LN normalizes; ratio shared/indep is the story
print("\nUpdate amplification: dh_N from perturbing all shared W by delta (Adam-like step)")
print(f"{'N':>3} {'shared,eps=1':>14} {'shared,scaled':>14}  (bigger = more LR-sensitive)")
for N in [2, 4, 8]:
    Ws = [rng.normal(size=(d,d))/np.sqrt(d) for _ in range(L)]
    dW = [rng.normal(size=(d,d))/np.sqrt(d)*0.01 for _ in range(L)]  # 1% update
    h0 = rng.normal(size=d)
    order = [i for i in range(L) for _ in range(N)]
    for label, eps in [("eps=1", 1.0), ("scaled", 1.0/(N*np.sqrt(L)))]:
        h_a = run(order, Ws, h0, eps)
        h_b = run(order, [W+D for W,D in zip(Ws,dW)], h0, eps)
        amp = np.linalg.norm(h_b - h_a)/np.linalg.norm(h_a)
        if label == "eps=1": a1 = amp
        else: a2 = amp
    print(f"{N:>3} {a1:>14.4f} {a2:>14.4f}")
