"""Rounds 4-6 analysis: N-sweep curve, learnable-lambda delta, epoch regime,
SSM-mixer gaps.

Usage: python analyze_sweeps.py  (reads results/ by fixed patterns)
"""

import glob
import json


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        out[r["arm"]] = r
    return out


def final(runs, arm):
    return runs[arm]["final_val"] if arm in runs else None


def main():
    smoke = load("results/smoke_*_s7.json") | load("results/n4_*_s7.json")
    nsweep = load("results/n_sweep_*_s7.json")
    epoch = load("results/epoch_*_s7.json")
    ssm = load("results/ssm_*_s7.json")

    # --- N-sweep at fixed 12 applications/token (4000 steps, seed 7)
    print("val loss vs N (12 block-applications/token, 98M tokens):")
    print(f"{'N':>4} {'stored':>7} {'unscaled':>9} {'eps=1/N':>9} {'eps gain':>9}")
    table = [
        (1, 12, final(smoke, "A"), final(smoke, "A")),
        (2, 6, final(smoke, "B"), final(smoke, "C")),
        (4, 3, final(smoke, "E1"), final(smoke, "E3")),
        (6, 2, final(nsweep, "I6u"), final(nsweep, "I6")),
        (12, 1, final(nsweep, "I12u"), final(nsweep, "I12")),
    ]
    for n, stored, u, s in table:
        us = f"{u:.4f}" if u else "-"
        ss = f"{s:.4f}" if s else "-"
        g = f"{u-s:+.4f}" if u and s else "-"
        print(f"{n:>4} {stored:>7} {us:>9} {ss:>9} {g:>9}")

    # --- learnable lambda
    print("\nlearnable lambda (H vs C, H4 vs E3):")
    for lab, fixed_arm, learn_arm, runs_f, runs_l in (
            ("N=2", "C", "H", smoke, nsweep),
            ("N=4", "E3", "H4", smoke | load("results/n4_*_s7.json"), nsweep)):
        f_, l_ = final(runs_f, fixed_arm), final(runs_l, learn_arm)
        if f_ and l_:
            print(f"  {lab}: fixed {f_:.4f} vs learned {l_:.4f} -> {l_-f_:+.4f}"
                  f" ({'learning helps' if l_ < f_ else 'fixed fine'})")

    # --- epoch regime (round 5): repeated vs fresh at same step budget
    if epoch:
        print("\ndata-constrained (50M tokens x ~9.8 epochs, 20K steps) vs fresh:")
        fresh = load("results/long_*_s7.json")
        for arm in ("A", "C", "E3"):
            e, f = final(epoch, arm), final(fresh, arm)
            if e and f:
                print(f"  {arm}: repeated {e:.4f} vs fresh {f:.4f} "
                      f"(overfit tax {e-f:+.4f})")
        for arm in ("C", "E3"):
            if final(epoch, arm) and final(epoch, "A"):
                gap_rep = final(epoch, arm) - final(epoch, "A")
                gap_fr = final(fresh, arm) - final(fresh, "A") if final(fresh, arm) else None
                comp = f" (fresh gap {gap_fr:+.4f})" if gap_fr else ""
                print(f"  {arm}-A repeated: {gap_rep:+.4f}{comp}")

    # --- SSM mixer (round 6)
    if ssm:
        print("\nSSM mixer (98M tokens, seed 7):")
        for arm in ("MA", "MB", "MC", "MB4", "MC4"):
            v = final(ssm, arm)
            if v:
                print(f"  {arm}: {v:.4f}")
        for n, u, s, au, as_ in (("N=2", "MB", "MC", "B", "C"),
                                 ("N=4", "MB4", "MC4", "E1", "E3")):
            if final(ssm, u) and final(ssm, s):
                g = final(ssm, u) - final(ssm, s)
                ag = (final(smoke, au) - final(smoke, as_)
                      if final(smoke, au) and final(smoke, as_) else None)
                extra = f" vs attention {ag:+.4f}" if ag else ""
                print(f"  {n} eps gain: {g:+.4f}{extra}")


if __name__ == "__main__":
    main()
