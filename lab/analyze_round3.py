"""Round-3 crossover analysis: gap-vs-tokens trajectory and extrapolation.

Loopie (2607.16051) reports its looped MoE overtakes the compute-matched
vanilla baseline only after ~600B tokens; before that, vanilla leads. So the
question here is not "is the gap zero at 490M tokens" but "is the gap closing,
and does eps=1/N close it faster than eps=1".

For each looped arm X vs vanilla A we fit gap(t) = a + b/sqrt(t) over the
second half of training (early points are warmup-dominated) and report the
extrapolated asymptote a (gap at infinite tokens) and the token count where
the fit crosses zero, if it does.
"""

import glob
import json
import sys

import numpy as np


def load_runs(pattern):
    runs = {}
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        runs[r["arm"]] = r
    return runs


def gap_series(runs, arm, base="A"):
    a = {c["step"]: c["val_loss"] for c in runs[base]["curve"]}
    xs, ys = [], []
    for c in runs[arm]["curve"]:
        if c["step"] in a and c["step"] > 0:
            xs.append(c["tokens"])
            ys.append(c["val_loss"] - a[c["step"]])
    return np.array(xs, dtype=float), np.array(ys)


def fit_crossover(xs, ys):
    """Fit gap = a + b/sqrt(tokens) on the last half; return (a, b, t_cross)."""
    n = len(xs) // 2
    X = np.stack([np.ones(n), 1 / np.sqrt(xs[-n:])], axis=1)
    coef, *_ = np.linalg.lstsq(X, ys[-n:], rcond=None)
    a, b = coef
    t_cross = (b / -a) ** 2 if a < 0 < b else None
    return a, b, t_cross


def main(pattern="results/long_*_s*.json"):
    runs = load_runs(pattern)
    if "A" not in runs:
        sys.exit(f"need vanilla arm A in {pattern}")

    print(f"{'arm':>4} {'final val':>10} {'gap@end':>9} {'gap@25%':>9} "
          f"{'asymptote':>10} {'t_cross':>12}")
    for arm, r in sorted(runs.items()):
        if arm == "A":
            print(f"{arm:>4} {r['final_val']:>10.4f} {'—':>9} {'—':>9} "
                  f"{'—':>10} {'—':>12}")
            continue
        xs, ys = gap_series(runs, arm)
        q = len(ys) // 4
        a, b, t_cross = fit_crossover(xs, ys)
        tc = f"{t_cross/1e9:.2f}B tok" if t_cross else "never (fit)"
        print(f"{arm:>4} {r['final_val']:>10.4f} {ys[-1]:>+9.4f} "
              f"{ys[q]:>+9.4f} {a:>+10.4f} {tc:>12}")

    print("\ngap vs A by tokens:")
    arms = [a for a in sorted(runs) if a != "A"]
    xs0, _ = gap_series(runs, arms[0])
    print(f"{'Mtok':>6}" + "".join(f"{a:>9}" for a in arms))
    series = {a: gap_series(runs, a)[1] for a in arms}
    for i, t in enumerate(xs0):
        row = f"{t/1e6:>6.0f}"
        for a in arms:
            row += f"{series[a][i]:>+9.4f}" if i < len(series[a]) else f"{'-':>9}"
        print(row)


if __name__ == "__main__":
    main(*sys.argv[1:])
