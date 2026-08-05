"""Round-13 verdict vs pre-registered bars (SPEC_1B gate).

P1: C1536 best-val < A1536 best-val, both seeds.
P2: gap <= -0.02 (2-seed mean).
P3: eval-suite mean ordering matches loss ordering, both seeds.
All three PASS -> 1B run justified. Any FAIL -> cancel SPEC_1B, publish
with the sub-0.4B / high-repetition bound.
"""

import glob
import json
import sys

TASKS = ["hellaswag", "piqa", "arc_easy", "lambada"]


def best_val(path):
    r = json.load(open(path))
    vals = [c["val_loss"] for c in r["curve"]]
    steps = [c["step"] for c in r["curve"]]
    i = vals.index(min(vals))
    return vals[i], steps[i], r["curve"][-1]["val_loss"]


def main():
    print("=== Round 13: 0.4B de-risk (A1536 417M vs C1536 247M, "
          "100M unique x 8 epochs) ===\n")
    gaps = {}
    for s in (7, 13):
        row = {}
        for arm in ("A1536", "C1536"):
            p = f"results/derisk_{arm}_s{s}.json"
            if not glob.glob(p):
                print(f"seed {s}: {arm} not finished")
                break
            bv, bstep, fin = best_val(p)
            row[arm] = (bv, bstep, fin)
            ep = bstep * 12 * 1024 / 100e6
            print(f"  {arm} s{s}: best {bv:.4f} @step {bstep} ({ep:.1f} ep), "
                  f"final {fin:.4f}, degr {fin-bv:+.4f}")
        if len(row) == 2:
            g = row["C1536"][0] - row["A1536"][0]
            gaps[s] = g
            print(f"  seed {s} gap (C-A): {g:+.4f}\n")

    if len(gaps) == 2:
        mean_gap = sum(gaps.values()) / 2
        p1 = all(g < 0 for g in gaps.values())
        p2 = mean_gap <= -0.02
        print(f"P1 (flip both seeds): {'PASS' if p1 else 'FAIL'} "
              f"({gaps[7]:+.4f}, {gaps[13]:+.4f})")
        print(f"P2 (mean gap <= -0.02): {'PASS' if p2 else 'FAIL'} "
              f"(mean {mean_gap:+.4f})")

    evs = {}
    for s in (7, 13):
        pair = {}
        for arm in ("A1536", "C1536"):
            p = f"evals/derisk_{arm}_s{s}.json"
            if glob.glob(p):
                r = json.load(open(p))["results"]
                pair[arm] = sum(r[t] for t in TASKS) / len(TASKS)
        if len(pair) == 2:
            evs[s] = pair
    if evs:
        print("\neval-suite means:")
        p3 = True
        for s, pair in evs.items():
            d = pair["C1536"] - pair["A1536"]
            ok = d > 0
            p3 &= ok
            print(f"  seed {s}: C {pair['C1536']:.4f} vs A {pair['A1536']:.4f} "
                  f"-> {d:+.4f} {'ok' if ok else 'ORDER MISMATCH'}")
        if len(evs) == 2:
            print(f"P3 (eval ordering both seeds): {'PASS' if p3 else 'FAIL'}")
            if len(gaps) == 2:
                verdict = p1 and p2 and p3
                print(f"\n>>> SPEC_1B gate: {'OPEN — run the 1B' if verdict else 'CLOSED — cancel 1B, publish with bound'}")


if __name__ == "__main__":
    main()
