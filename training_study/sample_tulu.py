"""Training study: background SFT data, a random sample of the Tülu 3 mixture used for
OLMo 2 SFT (allenai/tulu-3-sft-olmo-2-mixture; ODC-BY). Samples the same fraction from
each of the six parquet files (the mixture may be ordered by source), keeps
user/assistant conversations up to --max-chars, and writes
data/training_study/tulu_bg.jsonl.gz ({"id", "source", "messages"}). Each file is
downloaded, sampled and deleted in turn.

Usage: python training_study/sample_tulu.py [--n 20000]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "training_study"
DS = "allenai/tulu-3-sft-olmo-2-mixture"
TOTAL = 939_344


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--max-chars", type=int, default=6_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    frac = a.n * 1.25 / TOTAL                      # oversample, length filter removes some
    rows = []
    for i in range(6):
        f = hf_hub_download(DS, f"data/train-0000{i}-of-00006.parquet", repo_type="dataset",
                            cache_dir=str(OUT / "_hf"))
        t = pq.read_table(f, columns=["id", "messages", "source"]).to_pylist()
        keep = rng.random(len(t)) < frac
        for r, k in zip(t, keep):
            msgs = [m for m in r["messages"] if m["role"] in ("user", "assistant", "system")]
            if k and msgs and msgs[-1]["role"] == "assistant" and sum(len(m["content"]) for m in msgs) <= a.max_chars:
                rows.append(dict(id=r["id"], source=r["source"], messages=msgs))
        Path(f).unlink()
        print(f"file {i}: {len(rows)} kept so far", flush=True)
    idx = rng.permutation(len(rows))[: a.n]
    with gzip.open(OUT / "tulu_bg.jsonl.gz", "wt") as fh:
        for j in idx:
            fh.write(json.dumps(rows[j], ensure_ascii=False) + "\n")
    print(f"-> {OUT / 'tulu_bg.jsonl.gz'}: {len(idx)} conversations")


if __name__ == "__main__":
    main()
