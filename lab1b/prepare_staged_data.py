"""Build the 3B-token staged corpus for the 1B curriculum x loop experiment.

Three stages, one tokenizer (GPT-2 BPE), separate bins so the loader can
serve either curriculum (stage-sequential) or standard (shuffled-mix) order
from the same files:

  stage 1 math:       0.6B tokens  HuggingFaceTB/finemath  (finemath-3plus)
  stage 2 philosophy: 0.2B tokens  sedthh/gutenberg_english filtered by
                      philosophy/logic/ethics subjects (fallback: title
                      keywords on manu/project_gutenberg)
  stage 3 web:        2.2B tokens  HuggingFaceFW/fineweb-edu sample-10BT

Val: 10M general (fineweb-edu holdout) + 5M math (finemath holdout).

All arms see identical tokens; only order differs. Bins are uint16.
"""

import argparse
import json
import os

import numpy as np
import tiktoken
from datasets import load_dataset

ENC = tiktoken.get_encoding("gpt2")

PHIL_SUBJECTS = ("philosophy", "logic", "ethics", "metaphysics",
                 "epistemology", "stoic", "rhetoric")


def fill(buf, doc_iter, total, tag):
    pos = 0
    for text in doc_iter:
        ids = ENC.encode_ordinary(text)
        ids.append(ENC.eot_token)
        take = min(len(ids), total - pos)
        buf[pos:pos + take] = ids[:take]
        pos += take
        if pos >= total:
            break
        if pos % 50_000_000 < len(ids):
            print(f"[{tag}] {pos/1e6:.0f}M/{total/1e6:.0f}M", flush=True)
    if pos < total:
        raise RuntimeError(f"{tag}: ran dry at {pos/1e6:.0f}M of {total/1e6:.0f}M")
    return pos


def math_docs():
    ds = load_dataset("HuggingFaceTB/finemath", "finemath-3plus",
                      split="train", streaming=True)
    for d in ds:
        yield d["text"]


def phil_docs():
    try:
        ds = load_dataset("sedthh/gutenberg_english", split="train",
                          streaming=True)
        for d in ds:
            meta = str(d.get("METADATA", "")).lower()
            if any(s in meta for s in PHIL_SUBJECTS):
                yield d["TEXT"]
    except Exception as e:
        print(f"[phil] primary source failed ({e}); falling back to pg19 title filter",
              flush=True)
        ds = load_dataset("deepmind/pg19", split="train", streaming=True)
        for d in ds:
            title = d.get("short_book_title", "").lower()
            if any(s in title for s in PHIL_SUBJECTS):
                yield d["text"]


def web_docs(skip_tokens=0):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    for d in ds:
        yield d["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data1b")
    ap.add_argument("--math-tokens", type=int, default=600_000_000)
    # gutenberg philosophy yields ~159M; 150M target, remainder goes to web
    ap.add_argument("--phil-tokens", type=int, default=150_000_000)
    ap.add_argument("--web-tokens", type=int, default=2_250_000_000)
    ap.add_argument("--val-web", type=int, default=10_000_000)
    ap.add_argument("--val-math", type=int, default=5_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def have(name, tokens):
        p = f"{args.out_dir}/{name}.bin"
        return os.path.exists(p) and os.path.getsize(p) == tokens * 2

    # stage 1: math (train + val head)
    if not (have("stage1_math", args.math_tokens) and have("val_math", args.val_math)):
        total = args.math_tokens + args.val_math
        buf = np.empty(total, dtype=np.uint16)
        fill(buf, math_docs(), total, "math")
        buf[:args.val_math].tofile(f"{args.out_dir}/val_math.bin")
        buf[args.val_math:].tofile(f"{args.out_dir}/stage1_math.bin")
        del buf
    else:
        print("[math] bins exist, skipping", flush=True)

    # stage 2: philosophy
    if not have("stage2_phil", args.phil_tokens):
        buf = np.empty(args.phil_tokens, dtype=np.uint16)
        fill(buf, phil_docs(), args.phil_tokens, "phil")
        buf.tofile(f"{args.out_dir}/stage2_phil.bin")
        del buf
    else:
        print("[phil] bin exists, skipping", flush=True)

    # stage 3: web (val head then train)
    if not (have("stage3_web", args.web_tokens) and have("val_web", args.val_web)):
        total = args.web_tokens + args.val_web
        buf = np.empty(total, dtype=np.uint16)
        fill(buf, web_docs(), total, "web")
        buf[:args.val_web].tofile(f"{args.out_dir}/val_web.bin")
        buf[args.val_web:].tofile(f"{args.out_dir}/stage3_web.bin")
        del buf
    else:
        print("[web] bins exist, skipping", flush=True)

    manifest = {
        "stage1_math": args.math_tokens, "stage2_phil": args.phil_tokens,
        "stage3_web": args.web_tokens,
        "val_math": args.val_math, "val_web": args.val_web,
        "tokenizer": "gpt2", "dtype": "uint16",
    }
    json.dump(manifest, open(f"{args.out_dir}/manifest.json", "w"), indent=1)
    print("DONE", manifest, flush=True)


if __name__ == "__main__":
    main()
