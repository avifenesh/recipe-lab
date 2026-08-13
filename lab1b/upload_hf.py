"""Publish a finished run's artifacts to the Hugging Face model repo.

Every round ends with three local files per cell in `lab1b/results_1b/`:

    <tag>_best.pt          best-val weights   -> <tag>_best.pt
    <tag>.json             loss curve+config  -> <tag>_curve.json
    evals_<tag>.json       lm-eval readout    -> evals_<tag>.json

(the right-hand names match the convention the 1B factorial already used on
the hub, so old and new cells sit in one namespace). The weights are the
point of the upload — a missing ckpt is an error, not a warning, because a
cell whose weights are gone cannot be re-evaluated later. That is the
V_std lesson: verify by exact filename, never a glob a sibling satisfies.

The model card is regenerated from whatever `evals_*.json` files exist on
the hub after the push, so the repo stays self-describing without a
hand-maintained table.

Usage:
  python upload_hf.py --tag r20_V_fm_s7 --tag r20_V_wl_s13
  python upload_hf.py --tag r20_V_fm_s7 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = "Avifenesh/recipe-lab-1b"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results_1b")

# what each cell name means, so the card explains itself
ORDERS = {
    "std": "proportional shuffle (no staging)",
    "cur": "math -> phil -> web",
    "pm": "phil -> math -> web",
    "wl": "web -> math -> phil",
    "fm": "web -> phil -> math",
    "cur4": "math -> phil -> think -> web",
    "nophil": "math -> webx -> web (philosophy ablated)",
}


def files_for(tag: str) -> list[tuple[str, str]]:
    """[(local path, path in repo)] — ckpt first so a failure is loud."""
    return [
        (os.path.join(SRC, f"{tag}_best.pt"), f"{tag}_best.pt"),
        (os.path.join(SRC, f"{tag}.json"), f"{tag}_curve.json"),
        (os.path.join(SRC, f"evals_{tag}.json"), f"evals_{tag}.json"),
    ]


def build_card(api) -> str:
    """Regenerate the model card from the eval files present on the hub."""
    rows = []
    for name in sorted(api.list_repo_files(REPO)):
        if not (name.startswith("evals_") and name.endswith(".json")):
            continue
        tag = name[len("evals_"):-len(".json")]
        try:
            p = api.hf_hub_download(REPO, name)
            res = json.load(open(p)).get("results", {})
        except Exception:
            continue
        order = next((v for k, v in ORDERS.items()
                      if f"_{k}" in f"_{tag}_" or tag.endswith(f"_{k}")), "")
        rows.append((round(res.get("mean", float("nan")), 4), tag, order, res))
    rows.sort(reverse=True)
    head = ("| cell | stage order | eval mean | arc_easy | arc_c | "
            "winogrande | sciq | lambada |\n|---|---|---:|---:|---:|---:|---:|---:|")
    body = "\n".join(
        f"| `{t}` | {o} | **{m}** | {r.get('arc_easy','')} | "
        f"{r.get('arc_challenge','')} | {r.get('winogrande','')} | "
        f"{r.get('sciq','')} | {r.get('lambada','')} |"
        for m, t, o, r in rows)
    return f"""---
license: mit
tags:
- research
- curriculum-learning
- ablation
---

# recipe-lab — 1B curriculum/architecture ablation checkpoints

Best-val checkpoints, loss curves and lm-eval readouts for every 1B-scale
cell of the [recipe-lab](https://github.com/avifenesh/recipe-lab) campaign:
a hunt for a training recipe (data order, backbone, optimizer) that raises
downstream quality for a small hybrid-MoE model.

These are **research artifacts, not a product**: 1.3B dense / 1.3B-total
0.41B-active MoE, ~15B tokens, GPT-2 BPE, no instruction tuning and no
safety work. They exist so each claim in the campaign ledger can be
re-checked against the weights that produced it.

Every cell is one variable away from its control; the reasoning, the
pre-registered predictions and the retractions live in
[RESULTS_1B.md](https://github.com/avifenesh/recipe-lab/blob/main/RESULTS_1B.md).

## Cells (8-task lm-eval mean, 1000 samples/task)

{head}
{body}

Files per cell: `<cell>_best.pt` (weights), `<cell>_curve.json`
(loss curve + full config), `evals_<cell>.json` (per-task accuracies).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True,
                    help="cell tag, e.g. r20_V_fm_s7 (repeatable)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-card", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as e:
        print(f"FATAL no HF auth ({type(e).__name__}); "
              f"run `huggingface-cli login` or set HF_TOKEN", file=sys.stderr)
        return 2
    print(f"hf user={who} repo={args.repo}")

    plan = []
    for tag in args.tag:
        for local, remote in files_for(tag):
            if not os.path.exists(local):
                if local.endswith("_best.pt"):
                    print(f"FATAL missing weights {local}", file=sys.stderr)
                    return 1
                print(f"WARN missing {local}, skipping")
                continue
            plan.append((local, remote, os.path.getsize(local)))

    for local, remote, size in plan:
        print(f"  {remote:<34} {size/1e6:8.1f} MB")
    if args.dry_run:
        print("dry-run, nothing uploaded")
        return 0

    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    for local, remote, _ in plan:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo, repo_type="model")
        print(f"  pushed {remote}")

    if not args.no_card:
        card = build_card(api)
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
        print("  pushed README.md (card regenerated from hub evals)")
    print(f"DONE https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
