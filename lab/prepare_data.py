"""Tokenize a FineWeb-Edu slice into train/val .bin files (uint16 GPT-2 BPE)."""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=250_000_000)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    os.makedirs(args.out_dir, exist_ok=True)
    total = args.tokens + args.val_tokens
    buf = np.empty(total, dtype=np.uint16)
    pos = 0
    for doc in ds:
        ids = enc.encode_ordinary(doc["text"])
        ids.append(enc.eot_token)
        take = min(len(ids), total - pos)
        buf[pos:pos + take] = ids[:take]
        pos += take
        if pos >= total:
            break
        if pos % 10_000_000 < len(ids):
            print(f"{pos/1e6:.0f}M / {total/1e6:.0f}M tokens", flush=True)

    val = buf[:args.val_tokens]
    train = buf[args.val_tokens:pos]
    train.tofile(os.path.join(args.out_dir, "fineweb_train.bin"))
    val.tofile(os.path.join(args.out_dir, "fineweb_val.bin"))
    print(f"wrote {len(train)/1e6:.1f}M train, {len(val)/1e6:.1f}M val tokens")


if __name__ == "__main__":
    main()
