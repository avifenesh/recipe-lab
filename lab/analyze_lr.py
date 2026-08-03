"""Round-3b analysis: per-arm LR response and the re-parameterization test.

Two questions:
1. H2 (LR transfer): does the unscaled arm's optimal LR shift down as N grows
   while the scaled arm's stays at vanilla's optimum?
2. Re-parameterization: compare each arm at its own best LR. If unscaled-at-
   its-optimum == scaled-at-its-optimum, eps is just an LR knob; if scaled
   still wins, the forward-pass geometry matters.
"""

import glob
import json
import sys

ARM_N = {"A": 1, "B": 2, "C": 2, "E1": 4, "E3": 4}
ARM_DESC = {
    "A": "vanilla     N=1",
    "B": "unscaled    N=2",
    "C": "eps=1/N     N=2",
    "E1": "unscaled    N=4",
    "E3": "eps=1/N     N=4",
}


def main(pattern="results/lr2k_*.json", extra="results/hilr_*_s7.json"):
    runs = {}  # (arm, lr) -> final_val
    for p in sorted(glob.glob(pattern)) + sorted(glob.glob(extra)):
        r = json.load(open(p))
        runs[(r["arm"], r["config"]["lr"])] = r["final_val"]

    lrs = sorted({lr for _, lr in runs})
    print(f"{'arm':>4} {'desc':<18}" + "".join(f"{lr:>10.1e}" for lr in lrs)
          + f" {'best lr':>9} {'best val':>9}")
    best = {}
    for arm in ("A", "B", "C", "E1", "E3"):
        row = f"{arm:>4} {ARM_DESC[arm]:<18}"
        vals = {}
        for lr in lrs:
            v = runs.get((arm, lr))
            vals[lr] = v
            row += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
        have = {lr: v for lr, v in vals.items() if v is not None}
        if have:
            blr = min(have, key=have.get)
            best[arm] = (blr, have[blr])
            row += f" {blr:>9.1e} {have[blr]:>9.4f}"
        print(row)

    print("\nre-parameterization test (each arm at its own optimum):")
    for n, (u, s) in ((2, ("B", "C")), (4, ("E1", "E3"))):
        if u in best and s in best:
            du, ds = best[u][1], best[s][1]
            verdict = ("eps does forward-pass work" if ds < du
                       else "eps ~ LR re-parameterization")
            print(f"  N={n}: unscaled@opt {du:.4f} vs scaled@opt {ds:.4f} "
                  f"-> {ds-du:+.4f}  [{verdict}]")

    if "A" in best:
        print("\nLR-transfer test (optimal LR by arm):")
        for arm in ("A", "B", "C", "E1", "E3"):
            if arm in best:
                print(f"  {ARM_DESC[arm]}: {best[arm][0]:.1e}")


if __name__ == "__main__":
    main(*sys.argv[1:])
