"""Build the upgraded philosophy pool (SPEC_PILOT_MOE data plan).

The 1B factorial's philosophy stage wanted 0.2B tokens and the Gutenberg
pool capped at 159M — the curriculum's central stage was the one starved,
and Gutenberg is pre-1928 by construction so it cannot supply modern
philosophy at all. This builds a >=1B-token modern pool from three sources,
each to its own bin so later corpora can mix them at any ratio:

  phil_sep.bin       Stanford Encyclopedia of Philosophy, Dec-2023 scrape
                     (dmarx/stanford-encyclopedia-of-philosophy_dec23),
                     ~45M tokens, take all
  phil_papers.bin    PhilPapers open-access archive, English rows only
                     (malteos/philpapers-2023-10-28), ~650M tokens, take all
  phil_arxiv.bin     formal logic papers: proof-pile-2 arXiv subset joined
                     on math.LO / cs.LO ids from the arXiv metadata
                     snapshot (librarian-bots/arxiv-metadata-snapshot),
                     capped by --arxiv-tokens

Same tokenizer/dtype contract as prepare_staged_data.py (GPT-2 BPE, uint16,
EOT between docs). A stage2_phil bin for training is a plain concatenation
of these (head-truncated to the stage budget).

Run on the box, not locally:
  python prepare_phil_pool.py --out-dir data_phil
"""

import argparse
import gzip
import io
import json
import os

import numpy as np
import tiktoken

ENC = tiktoken.get_encoding("gpt2")
LOGIC_CATS = ("math.LO", "cs.LO")


def write_bin(path, doc_iter, cap, tag):
    """Tokenize docs into a uint16 bin; cap=0 means take everything."""
    chunks, total = [], 0
    for text in doc_iter:
        ids = ENC.encode_ordinary(text)
        ids.append(ENC.eot_token)
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if total % 50_000_000 < len(ids):
            print(f"[{tag}] {total/1e6:.0f}M tokens", flush=True)
        if cap and total >= cap:
            break
    buf = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
    if cap:
        buf = buf[:cap]
    buf.tofile(path)
    print(f"[{tag}] DONE {len(buf)/1e6:.1f}M tokens -> {path}", flush=True)
    return int(len(buf))


def sep_docs():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("dmarx/stanford-encyclopedia-of-philosophy_dec23",
                        "stanford-encyclopedia-of-philosophy_dec23.jsonl",
                        repo_type="dataset")
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        c = d["content"]
        text = ("\n".join(c) if isinstance(c, list) else c).strip()
        if len(text) > 2000:                      # skip stub/redirect pages
            yield f"{d['title']}\n\n{text}"


def philpapers_docs():
    from huggingface_hub import hf_hub_download
    for i in range(11):
        p = hf_hub_download("malteos/philpapers-2023-10-28",
                            f"data/phil-archive-part-{i:02d}.jsonl.gz",
                            repo_type="dataset")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["meta"].get("language") != "en":
                    continue
                text = d["text"].strip()
                if len(text) > 5000:              # drop abstract-only rows
                    yield text


def logic_ids():
    """arXiv ids whose category list touches math.LO or cs.LO."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    ids = set()
    for i in range(10):
        p = hf_hub_download("librarian-bots/arxiv-metadata-snapshot",
                            f"data/train-{i:05d}-of-00010.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p, columns=["id", "categories"])
        hit = df[df["categories"].str.contains("math.LO|cs.LO", regex=True)]
        # normalize old-style ids ('math/0211159' -> 'math0211159') so they
        # match however proof-pile-2 spells them
        for x in hit["id"]:
            ids.add(x)
            ids.add(x.replace("/", ""))
        del df
    print(f"[arxiv] {len(ids)} logic ids (both spellings)", flush=True)
    return ids


def arxiv_logic_docs(ids):
    """Stream proof-pile-2 arXiv shards, keep rows whose id is in `ids`."""
    import zstandard
    from huggingface_hub import hf_hub_download
    for i in range(100):
        p = hf_hub_download("EleutherAI/proof-pile-2",
                            f"arxiv/train/arXiv_{i:03d}.jsonl.zst",
                            repo_type="dataset")
        with open(p, "rb") as fh:
            stream = zstandard.ZstdDecompressor().stream_reader(fh)
            for line in io.TextIOWrapper(stream, encoding="utf-8"):
                d = json.loads(line)
                aid = d["meta"].get("arxiv_id", "")
                if aid in ids or aid.replace("/", "") in ids:
                    yield d["text"]
        os.remove(p)                    # ~100 shards; don't hoard the cache
        print(f"[arxiv] shard {i:03d} scanned", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_phil")
    ap.add_argument("--arxiv-tokens", type=int, default=600_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_path = f"{args.out_dir}/manifest.json"
    manifest = (json.load(open(manifest_path))
                if os.path.exists(manifest_path) else {})

    def build(name, docs_fn, cap):
        path = f"{args.out_dir}/{name}.bin"
        if name in manifest and os.path.exists(path) \
                and os.path.getsize(path) == manifest[name] * 2:
            print(f"[{name}] exists, skipping", flush=True)
            return
        manifest[name] = write_bin(path, docs_fn(), cap, name)
        manifest["tokenizer"], manifest["dtype"] = "gpt2", "uint16"
        json.dump(manifest, open(manifest_path, "w"), indent=1)

    build("phil_sep", sep_docs, 0)
    build("phil_papers", philpapers_docs, 0)
    build("phil_arxiv", lambda: arxiv_logic_docs(logic_ids()),
          args.arxiv_tokens)
    print("DONE", manifest, flush=True)


if __name__ == "__main__":
    main()
