"""Proof 5 — the correlated-accumulation pathology is mixer-agnostic:
it appears when the residual branch is a selective SSM (Mamba-style), and
eps=1/N fixes it there too.

Proofs 1-3 used tanh/MLP mixers. A Mamba block is still a residual block:
x <- x + eps * SSM(LN(x)). If the Theta(N^2) shared-weight accumulation is a
property of the residual+sharing structure (not the mixer), it must reproduce
with a selective state-space mixer. LT2 reports looped-Mamba training
instabilities; this proof says those are the predicted blowup.

Mixer: minimal selective SSM (S6-style, real-valued):
  per token t:  s_t = exp(-softplus(a) * dt_t) * s_{t-1} + dt_t * (x_t B^T)
                y_t = s_t C + x_t * D          (per-channel skip)
  dt_t = softplus(x_t W_dt)  (input-dependent step size = selectivity)
Then y is gated: y * silu(x W_g), projected back. Shared across N loop steps.

PASS if: unscaled shared-loop final norm grows ~N (Theta(N^2) energy) with
LN in the branch (pre-LN form), matching proof3's transformer behavior, while
eps=1/N keeps it bounded; independent weights grow ~sqrt(N).
"""

import numpy as np

rng = np.random.default_rng(3)
D, DSTATE, T = 256, 16, 64          # width, SSM state size, sequence length


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def silu(z):
    return z / (1 + np.exp(-np.clip(z, -30, 30)))


def make_params():
    return dict(
        W_dt=rng.normal(size=(D, 1)) / np.sqrt(D),
        a=rng.normal(size=(DSTATE,)) * 0.5,
        B=rng.normal(size=(DSTATE, D)) / np.sqrt(D),
        C=rng.normal(size=(DSTATE, D)) / np.sqrt(DSTATE),
        Dskip=rng.normal(size=(D,)) * 0.1,
        W_g=rng.normal(size=(D, D)) / np.sqrt(D),
        W_o=rng.normal(size=(D, D)) / np.sqrt(D),
    )


def layernorm(x):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5)


def ssm_mixer(x, p):
    """x: (T, D) -> (T, D), selective scan over the sequence."""
    dt = softplus(x @ p["W_dt"])                    # (T, 1)
    decay = np.exp(-softplus(p["a"])[None, :] * dt)  # (T, DSTATE)
    u = dt * (x @ p["B"].T)                          # (T, DSTATE)
    s = np.zeros(DSTATE)
    ys = np.empty_like(x)
    for t in range(T):
        s = decay[t] * s + u[t]
        ys[t] = s @ p["C"] + x[t] * p["Dskip"]
    return (ys * silu(x @ p["W_g"])) @ p["W_o"]


def run(n_loop, eps, shared):
    x = rng_run.normal(size=(T, D))
    x /= np.linalg.norm(x, axis=-1, keepdims=True) / np.sqrt(D)
    p = make_params_run()
    for k in range(n_loop):
        pk = p if shared else make_params_run()
        x = x + eps * ssm_mixer(layernorm(x), pk)
    return float(np.linalg.norm(x, axis=-1).mean() / np.sqrt(D))


print(f"{'N':>4} {'shared eps=1':>13} {'shared eps=1/N':>15} {'indep eps=1':>12}")
rows = []
for n in (2, 4, 8, 16):
    vals = []
    for shared, eps in ((True, 1.0), (True, 1.0 / n), (False, 1.0)):
        rng_run = np.random.default_rng(11)
        make_params_run = lambda: {k: v.copy() for k, v in make_params().items()}
        rng = np.random.default_rng(3)   # reset param stream per config
        acc = [run(n, eps, shared) for _ in range(3)]
        vals.append(np.mean(acc))
    rows.append((n, *vals))
    print(f"{n:>4} {vals[0]:>13.3f} {vals[1]:>15.3f} {vals[2]:>12.3f}")

n_, sh, sc, ind = zip(*rows)
sh_ratio = sh[-1] / sh[0]; ind_ratio = ind[-1] / ind[0]
n_ratio = n_[-1] / n_[0]
print(f"\nN x{n_ratio:.0f}: shared-unscaled norm x{sh_ratio:.2f} "
      f"(linear-in-N predicts x{n_ratio:.0f}), "
      f"indep x{ind_ratio:.2f} (sqrt predicts x{np.sqrt(n_ratio):.1f})")
scaled_flat = max(sc) / min(sc) < 1.5
print(f"eps=1/N bounded across N (max/min < 1.5): {scaled_flat}")
verdict = ("CONFIRMED" if sh_ratio > 0.6 * n_ratio and scaled_flat
           and sh_ratio / ind_ratio > 2 else "REFUTED")
print(f"SSM-mixer correlated accumulation: {verdict}")
