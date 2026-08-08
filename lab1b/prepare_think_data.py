"""Stage-2.5 'thinking' corpus + nophil web extension.

think bin (200M tokens): reasoning traces rendered as plain pretraining text
  - open-thoughts/OpenThoughts-114k: conversations -> question + full
    deliberate solution (keeps the <think> style deliberation)
  - facebook/natural_reasoning: question + reference answer + best response

webx bin (150M tokens): extra FineWeb-Edu, skipping the first ~2.26B tokens
already used by stage3, for the nophil arm (philosophy swapped for web at
matched total budget).
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset

ENC = tiktoken.get_encoding("gpt2")


def fill(buf, docs, total, tag):
    pos = 0
    for text in docs:
        ids = ENC.encode_ordinary(text)
        ids.append(ENC.eot_token)
        take = min(len(ids), total - pos)
        buf[pos:pos + take] = ids[:take]
        pos += take
        if pos >= total:
            break
        if pos % 25_000_000 < len(ids):
            print(f"[{tag}] {pos/1e6:.0f}M/{total/1e6:.0f}M", flush=True)
    if pos < total:
        raise RuntimeError(f"{tag} dry at {pos}")


def think_docs():
    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train",
                      streaming=True)
    for d in ds:
        conv = d["conversations"]
        text = "\n\n".join(t.get("value", "") for t in conv)
        if len(text) > 200:
            yield text
    ds2 = load_dataset("facebook/natural_reasoning", split="train",
                       streaming=True)
    for d in ds2:
        resp = d["responses"][0]["response"] if d["responses"] else ""
        text = (f"Question: {d['question']}\n\nReasoning and answer: {resp}\n\n"
                f"Reference: {d.get('reference_answer', '')}")
        if len(text) > 200:
            yield text


def webx_docs(skip_tokens):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    seen = 0
    for d in ds:
        n = len(ENC.encode_ordinary(d["text"])) + 1
        if seen < skip_tokens:
            seen += n
            continue
        yield d["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data1b")
    ap.add_argument("--think-tokens", type=int, default=200_000_000)
    ap.add_argument("--webx-tokens", type=int, default=150_000_000)
    args = ap.parse_args()

    p = f"{args.out_dir}/stage25_think.bin"
    if not (os.path.exists(p) and os.path.getsize(p) == args.think_tokens * 2):
        buf = np.empty(args.think_tokens, dtype=np.uint16)
        fill(buf, think_docs(), args.think_tokens, "think")
        buf.tofile(p)
        del buf
    else:
        print("[think] exists", flush=True)

    p = f"{args.out_dir}/stage3b_webx.bin"
    if not (os.path.exists(p) and os.path.getsize(p) == args.webx_tokens * 2):
        buf = np.empty(args.webx_tokens, dtype=np.uint16)
        fill(buf, webx_docs(2_260_000_000), args.webx_tokens, "webx")
        buf.tofile(p)
    else:
        print("[webx] exists", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
