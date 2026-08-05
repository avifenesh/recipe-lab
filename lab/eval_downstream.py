"""Downstream evals for recipe-lab checkpoints.

Tasks (standard small-LM zero-shot protocol):
  - hellaswag, piqa, arc_easy: multiple choice, score each ending by
    length-normalized sum of token logprobs given the context, pick argmax.
  - lambada: predict the final word exactly (greedy over its tokens).

Usage:
  python eval_downstream.py --ckpt ckpts/epoch39_C1024_s7.pt \
      --tasks hellaswag,piqa,arc_easy,lambada --limit 1000
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from train import ARMS

ENC = tiktoken.get_encoding("gpt2")
DEV = "cuda"


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ARMS[ck["arm"]])
    model = GPT(cfg).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ck['arm']} @step {ck['step']} val={ck['val_loss']:.4f}")
    return model, cfg


@torch.no_grad()
def choice_logprob(model, cfg, ctx_ids, cont_ids):
    """Mean per-token logprob of cont_ids given ctx_ids."""
    ids = (ctx_ids + cont_ids)[-cfg.block_size:]
    n_cont = min(len(cont_ids), len(ids) - 1)
    x = torch.tensor(ids[:-1], device=DEV)[None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    tgt = torch.tensor(ids[1:], device=DEV)
    tok_lp = logp.gather(-1, tgt[:, None]).squeeze(-1)
    return tok_lp[-n_cont:].mean().item()


def eval_multiple_choice(model, cfg, rows, get_ctx, get_choices, get_label):
    correct = 0
    for row in rows:
        ctx = ENC.encode_ordinary(get_ctx(row))
        scores = [choice_logprob(model, cfg, ctx,
                                 ENC.encode_ordinary(c))
                  for c in get_choices(row)]
        correct += int(np.argmax(scores) == get_label(row))
    return correct / len(rows)


@torch.no_grad()
def eval_lambada(model, cfg, rows):
    correct = 0
    for row in rows:
        text = row["text"]
        ctx_str, last = text.rsplit(" ", 1)
        ctx = ENC.encode_ordinary(ctx_str)[-cfg.block_size + 8:]
        want = ENC.encode_ordinary(" " + last)
        ids = list(ctx)
        ok = True
        for w in want:
            x = torch.tensor(ids[-cfg.block_size:], device=DEV)[None]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            nxt = int(logits[0, -1].argmax())
            if nxt != w:
                ok = False
                break
            ids.append(nxt)
        correct += int(ok)
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="hellaswag,piqa,arc_easy,lambada")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    model, cfg = load_model(args.ckpt)
    results = {}

    for task in args.tasks.split(","):
        if task == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation")
            rows = ds.select(range(min(args.limit, len(ds))))
            acc = eval_multiple_choice(
                model, cfg, rows,
                lambda r: r["ctx"],
                lambda r: [" " + e for e in r["endings"]],
                lambda r: int(r["label"]))
        elif task == "piqa":
            # ybisk/piqa main branch is a loader script (unsupported by
            # datasets>=3); read the auto-converted parquet branch instead
            ds = load_dataset("parquet", split="validation", data_files={
                "validation": "https://huggingface.co/datasets/ybisk/piqa/resolve/refs%2Fconvert%2Fparquet/plain_text/validation/0000.parquet"})
            rows = ds.select(range(min(args.limit, len(ds))))
            acc = eval_multiple_choice(
                model, cfg, rows,
                lambda r: r["goal"],
                lambda r: [" " + r["sol1"], " " + r["sol2"]],
                lambda r: int(r["label"]))
        elif task == "arc_easy":
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy",
                              split="validation")
            rows = [r for r in ds.select(range(min(args.limit, len(ds))))]
            acc = eval_multiple_choice(
                model, cfg, rows,
                lambda r: "Question: " + r["question"] + "\nAnswer:",
                lambda r: [" " + t for t in r["choices"]["text"]],
                lambda r: r["choices"]["label"].index(r["answerKey"]))
        elif task == "lambada":
            ds = load_dataset("EleutherAI/lambada_openai", "en",
                              split="test")
            rows = ds.select(range(min(args.limit, len(ds))))
            acc = eval_lambada(model, cfg, rows)
        else:
            raise ValueError(task)
        results[task] = round(acc, 4)
        print(f"{task}: {acc:.4f}", flush=True)

    if args.out:
        json.dump({"ckpt": args.ckpt, "limit": args.limit,
                   "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
