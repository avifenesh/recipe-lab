"""Proof 4 — eps-scaling restores LR transfer across loop counts (H2).

Claim (from 2606.18524's Thm 2 reasoning): with weight sharing, the gradient
of a looped block is a sum of N per-application terms that are *correlated*,
so the per-step weight update grows ~N under fixed LR. Unscaled looping (eps=1)
therefore shifts the optimal LR down as N grows; eps=1/N cancels the growth,
making the optimal LR independent of N.

Test: tiny shared-weight regression net (numpy, CPU), train at a grid of LRs
for each N in {1,2,4,8}, unscaled vs scaled. Record the LR that minimizes
final loss. PASS if the unscaled optimal LR falls with N while the scaled one
stays within one grid notch of the N=1 optimum.

Model: h_{k+1} = h_k + eps * W phi(h_k), phi=tanh, loss = ||P h_N - y||^2.
Both W and readout P train. Deterministic full-batch GD so the comparison is
pure optimization dynamics, no sampling noise.
"""

import numpy as np

rng = np.random.default_rng(0)
D, DOUT, B, STEPS = 64, 8, 256, 300

X = rng.normal(size=(B, D)) / np.sqrt(D)
W_true = rng.normal(size=(D, D)) / np.sqrt(D)
P_true = rng.normal(size=(DOUT, D)) / np.sqrt(D)
Y = (P_true @ np.tanh(X @ W_true.T).T).T  # fixed teacher


def train(n_loop, eps, lr, steps=STEPS):
    W = rng_init.normal(size=(D, D)) * 0.5 / np.sqrt(D)
    P = rng_init.normal(size=(DOUT, D)) / np.sqrt(D)
    for _ in range(steps):
        # forward, keeping per-application activations
        hs = [X]
        for _ in range(n_loop):
            hs.append(hs[-1] + eps * np.tanh(hs[-1] @ W.T))
        pred = hs[-1] @ P.T
        err = pred - Y                                   # (B, DOUT)
        loss = float((err ** 2).mean())
        if not np.isfinite(loss) or loss > 1e6:
            return np.inf
        # backward
        gP = 2 * err.T @ hs[-1] / err.size
        gh = 2 * err @ P / err.size                      # dL/dh_N
        gW = np.zeros_like(W)
        for k in range(n_loop - 1, -1, -1):
            pre = hs[k] @ W.T
            u = eps * gh * (1 - np.tanh(pre) ** 2)       # through nonlinearity
            gW += u.T @ hs[k]
            gh = gh + u @ W                              # residual + branch
        W -= lr * gW
        P -= lr * gP
    return loss


LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
print(f"{'N':>3} {'mode':>9} " + "".join(f"{lr:>9}" for lr in LRS) + f" {'opt LR':>8}")
opt = {}
for n in (1, 2, 4, 8):
    for mode, eps in (("unscaled", 1.0), ("scaled", 1.0 / n)):
        rng_init = np.random.default_rng(42)             # same init every run
        losses = [train(n, eps, lr) for lr in LRS]
        best = LRS[int(np.argmin(losses))]
        opt[(n, mode)] = best
        row = "".join(f"{l:>9.4f}" if np.isfinite(l) else f"{'div':>9}"
                      for l in losses)
        print(f"{n:>3} {mode:>9} {row} {best:>8}")

print()
un = [opt[(n, "unscaled")] for n in (1, 2, 4, 8)]
sc = [opt[(n, "scaled")] for n in (1, 2, 4, 8)]
print("unscaled optimal LR by N:", un)
print("scaled   optimal LR by N:", sc)
shift_un = un[0] / un[-1]
grid = np.array(LRS)
notch = lambda a, b: abs(int(np.where(grid == a)[0]) - int(np.where(grid == b)[0]))
stable_sc = max(notch(sc[0], s) for s in sc) <= 1
print(f"\nunscaled N=1->8 optimal-LR shift: {shift_un:.1f}x")
print(f"scaled within one grid notch of N=1 optimum across all N: {stable_sc}")
verdict = "CONFIRMED" if (shift_un >= 4 and stable_sc) else "REFUTED"
print(f"H2 LR-transfer: {verdict}")
