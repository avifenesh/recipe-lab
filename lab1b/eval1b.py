"""Downstream evals for the 1B factorial best-checkpoints.

Same protocol as lab/eval_downstream.py (length-normalized MC logprob +
lambada exact-match) extended per SPEC_1B: full validation sets where
feasible, plus arc_challenge, winogrande, boolq, sciq.

Usage:
  python eval1b.py --ckpt ckpts/V_std_best.pt --out evals/V_std.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model1b import ARMS_1B, Config1B, GPT1B

ENC = tiktoken.get_encoding("gpt2")
DEV = "cuda"


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config1B(**ARMS_1B[ck["arm"]])
    model = GPT1B(cfg).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ck['arm']}_{ck.get('order','?')} @step {ck['step']} "
          f"val_web={ck.get('val_web', float('nan')):.4f}", flush=True)
    return model, cfg


@torch.no_grad()
def choice_logprob(model, cfg, ctx_ids, cont_ids):
    ids = (ctx_ids + cont_ids)[-cfg.block_size:]
    n_cont = min(len(cont_ids), len(ids) - 1)
    x = torch.tensor(ids[:-1], device=DEV)[None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(x, x)  # targets arg reused for full logits
    logp = F.log_softmax(logits[0].float(), dim=-1)
    tgt = torch.tensor(ids[1:], device=DEV)
    tok_lp = logp.gather(-1, tgt[:, None]).squeeze(-1)
    return tok_lp[-n_cont:].mean().item()


def eval_mc(model, cfg, rows, get_ctx, get_choices, get_label, tag):
    correct = 0
    for i, row in enumerate(rows):
        ctx = ENC.encode_ordinary(get_ctx(row))
        scores = [choice_logprob(model, cfg, ctx, ENC.encode_ordinary(c))
                  for c in get_choices(row)]
        correct += int(np.argmax(scores) == get_label(row))
        if (i + 1) % 500 == 0:
            print(f"  [{tag}] {i+1}/{len(rows)} acc so far {correct/(i+1):.4f}",
                  flush=True)
    return correct / len(rows)


@torch.no_grad()
def eval_lambada(model, cfg, rows):
    correct = 0
    for row in rows:
        ctx_str, last = row["text"].rsplit(" ", 1)
        ctx = ENC.encode_ordinary(ctx_str)[-(cfg.block_size - 8):]
        want = ENC.encode_ordinary(" " + last)
        ids = list(ctx)
        ok = True
        for w in want:
            x = torch.tensor(ids[-cfg.block_size:], device=DEV)[None]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x, x)
            nxt = int(logits[0, -1].argmax())
            if nxt != w:
                ok = False
                break
            ids.append(nxt)
        correct += int(ok)
    return correct / len(rows)


def main(loader=load_model):
    """`loader` lets another architecture reuse this exact protocol — the
    suite must be identical across arms for the numbers to be comparable."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="hellaswag,piqa,arc_easy,arc_challenge,"
                                       "winogrande,boolq,sciq,lambada")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    model, cfg = loader(args.ckpt)
    results = {}

    def rows_of(ds):
        return ds.select(range(min(args.limit, len(ds))))

    for task in args.tasks.split(","):
        if task == "hellaswag":
            acc = eval_mc(model, cfg, rows_of(load_dataset(
                "Rowan/hellaswag", split="validation")),
                lambda r: r["ctx"],
                lambda r: [" " + e for e in r["endings"]],
                lambda r: int(r["label"]), task)
        elif task == "piqa":
            ds = load_dataset("parquet", split="validation", data_files={
                "validation": "https://huggingface.co/datasets/ybisk/piqa/resolve/refs%2Fconvert%2Fparquet/plain_text/validation/0000.parquet"})
            acc = eval_mc(model, cfg, rows_of(ds),
                          lambda r: r["goal"],
                          lambda r: [" " + r["sol1"], " " + r["sol2"]],
                          lambda r: int(r["label"]), task)
        elif task in ("arc_easy", "arc_challenge"):
            name = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
            ds = load_dataset("allenai/ai2_arc", name, split="validation")
            rows = [r for r in rows_of(ds)
                    if r["answerKey"] in r["choices"]["label"]]
            acc = eval_mc(model, cfg, rows,
                          lambda r: "Question: " + r["question"] + "\nAnswer:",
                          lambda r: [" " + t for t in r["choices"]["text"]],
                          lambda r: r["choices"]["label"].index(r["answerKey"]),
                          task)
        elif task == "winogrande":
            ds = load_dataset("allenai/winogrande", "winogrande_xl",
                              split="validation", trust_remote_code=True)
            def wg_ctx(r):
                return r["sentence"].split("_")[0].strip()
            def wg_choices(r):
                tail = r["sentence"].split("_")[1]
                return [" " + r["option1"] + tail, " " + r["option2"] + tail]
            acc = eval_mc(model, cfg, rows_of(ds), wg_ctx, wg_choices,
                          lambda r: int(r["answer"]) - 1, task)
        elif task == "boolq":
            ds = load_dataset("google/boolq", split="validation")
            acc = eval_mc(model, cfg, rows_of(ds),
                          lambda r: r["passage"][-3000:] + "\nQuestion: "
                                    + r["question"] + "\nAnswer:",
                          lambda r: [" no", " yes"],
                          lambda r: int(r["answer"]), task)
        elif task == "sciq":
            ds = load_dataset("allenai/sciq", split="validation")
            def sq_choices(r):
                return [" " + r["correct_answer"], " " + r["distractor1"],
                        " " + r["distractor2"], " " + r["distractor3"]]
            acc = eval_mc(model, cfg, rows_of(ds),
                          lambda r: "Question: " + r["question"] + "\nAnswer:",
                          sq_choices, lambda r: 0, task)
        elif task == "lambada":
            ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
            acc = eval_lambada(model, cfg, rows_of(ds))
        else:
            raise ValueError(task)
        results[task] = round(acc, 4)
        print(f"{task}: {acc:.4f}", flush=True)

    results["mean"] = round(sum(results.values()) / len(results), 4)
    if args.out:
        json.dump({"ckpt": args.ckpt, "limit": args.limit,
                   "results": results}, open(args.out, "w"), indent=1)
    print("MEAN:", results["mean"], flush=True)


if __name__ == "__main__":
    main()
