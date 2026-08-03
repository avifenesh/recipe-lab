"""Compare arms: val-loss curves at matched tokens, final gaps, verdict."""

import glob
import json
import sys

ARM_DESC = {
    "A": "vanilla L12",
    "B": "layer-loop 6x2 eps=1",
    "C": "layer-loop 6x2 eps=1/N",
    "D": "model-loop 6x2 eps=1/N",
}


def main(pattern="results/smoke_*_s*.json"):
    runs = {}
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        runs.setdefault(r["arm"], []).append(r)

    if not runs:
        sys.exit(f"no results matching {pattern}")

    print(f"{'arm':>4} {'desc':<24} {'params':>8} {'seeds':>5} "
          f"{'final val':>10} {'wall(s)':>8}")
    finals = {}
    for arm in "ABCD":
        if arm not in runs:
            continue
        rs = runs[arm]
        vals = [r["final_val"] for r in rs]
        mean = sum(vals) / len(vals)
        finals[arm] = vals
        wall = sum(r.get("wall_sec", 0) for r in rs) / len(rs)
        pretty = ",".join(f"{v:.4f}" for v in vals)
        print(f"{arm:>4} {ARM_DESC[arm]:<24} {rs[0]['params']/1e6:>7.1f}M "
              f"{len(rs):>5} {pretty:>10} {wall:>8.0f}")

    # curve table at shared eval steps (seed-averaged)
    steps = [c["step"] for c in runs[min(runs)][0]["curve"]]
    print("\nval loss by step (seed mean):")
    print(f"{'step':>6}" + "".join(f"{a:>9}" for a in "ABCD" if a in runs))
    for i, s in enumerate(steps):
        row = f"{s:>6}"
        for a in "ABCD":
            if a not in runs:
                continue
            vs = [r["curve"][i]["val_loss"] for r in runs[a]
                  if i < len(r["curve"])]
            row += f"{sum(vs)/len(vs):>9.4f}" if vs else f"{'-':>9}"
        print(row)

    if "B" in finals and "C" in finals:
        mb = sum(finals["B"]) / len(finals["B"])
        mc = sum(finals["C"]) / len(finals["C"])
        print(f"\nC-B gap (scaling fix): {mc-mb:+.4f}  "
              f"({'C wins' if mc < mb else 'B wins'})")
    if "A" in finals and "C" in finals:
        ma = sum(finals["A"]) / len(finals["A"])
        mc = sum(finals["C"]) / len(finals["C"])
        print(f"C-A gap (vs vanilla):  {mc-ma:+.4f}  "
              f"({'C wins' if mc < ma else 'A wins'})")
    if "C" in finals and "D" in finals:
        mc = sum(finals["C"]) / len(finals["C"])
        md = sum(finals["D"]) / len(finals["D"])
        print(f"C-D gap (ordering):    {mc-md:+.4f}  "
              f"({'layer-loop wins' if mc < md else 'model-loop wins'})")


if __name__ == "__main__":
    main(*sys.argv[1:])
