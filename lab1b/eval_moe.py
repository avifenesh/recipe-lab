"""Downstream evals for hybrid-MoE checkpoints.

Reuses eval1b.py's protocol verbatim (length-normalized MC logprob +
lambada exact match) with an MoE-aware checkpoint loader, so hybrid-MoE
numbers sit in the same table as the dense factorial's without a second
implementation drifting out of sync.

Usage:
  python eval_moe.py --ckpt ckpts/r16_HL_cur_best.pt --out evals_HL.json
"""

from __future__ import annotations

import torch

import eval1b
from model_moe import ARMS_MOE, SIZES_MOE, ConfigMoE, HybridMoE

DEV = "cuda"


def load_moe(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kw = dict(SIZES_MOE[ck["size"]])
    kw.update(ARMS_MOE[ck["arm"]])
    if ck.get("block_size"):
        kw["block_size"] = ck["block_size"]
    cfg = ConfigMoE(**kw)
    model = HybridMoE(cfg).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    val = next((ck[k] for k in ("val_web", "val") if k in ck), float("nan"))
    print(f"loaded {ck['arm']}/{ck['size']} @step {ck['step']} "
          f"val={val:.4f} total={model.num_params()/1e9:.3f}B "
          f"active={model.num_active_params()/1e9:.3f}B", flush=True)
    return model, cfg


if __name__ == "__main__":
    eval1b.main(loader=load_moe)
