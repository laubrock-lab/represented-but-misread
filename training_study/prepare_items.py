"""Training study, step 1: human texts and generation prompts from GNAD10.

GNAD10 (community-datasets/gnad10; CC BY-NC-SA 4.0): German news articles from the
Austrian daily DER STANDARD, 9 sections. Disjoint from our three test corpora.

Samples 2,400 articles stratified by section (2,000 training items, 400 held out),
assigns each a target length of 25, 80 or 150 words (the human text = its first L words,
mirroring the lengths of our test corpora), and a generation prompt built from the
article's first sentence and section. Output: data/training_study/items.parquet
(public text; may go to the cluster).
Usage: python training_study/prepare_items.py [--gnad PATH]
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "training_study"
SECTIONS = {0: "Web", 1: "Panorama", 2: "International", 3: "Wirtschaft", 4: "Sport",
            5: "Inland", 6: "Etat", 7: "Wissenschaft", 8: "Kultur"}
LENGTHS = [25, 80, 150]


def first_words(text: str, n: int) -> str | None:
    spans = list(re.finditer(r"\S+", text))
    return text[:spans[n - 1].end()] if len(spans) >= n else None


def first_sentence(text: str, max_words: int = 40) -> str:
    s = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return " ".join(s.split()[:max_words])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gnad", default=glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--gnad10/"
                                                       "snapshots/*/data/train-00000-of-00001.parquet"))[0])
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--n-heldout", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = pd.read_parquet(a.gnad)
    d = d[d.text.str.split().str.len() >= 170].reset_index(drop=True)      # long enough for 150 words
    d["section"] = d.label.map(SECTIONS)
    per = a.n // len(SECTIONS)
    s = (d.groupby("section", group_keys=False)
           .apply(lambda g: g.sample(min(per, len(g)), random_state=a.seed)))
    s = s.sample(frac=1, random_state=a.seed).head(a.n).reset_index(drop=True)
    s["item_id"] = [f"g{i:04d}" for i in range(len(s))]
    s["length"] = rng.choice(LENGTHS, len(s))
    s["human_text"] = [first_words(t, L) for t, L in zip(s.text, s.length)]
    s["topic"] = s.text.map(first_sentence)
    s["split"] = ["heldout" if i < a.n_heldout else "train" for i in range(len(s))]
    s["generator"] = np.where(np.arange(len(s)) % 2 == 0, "qwen3-8b", "gemma3-12b")
    OUT.mkdir(parents=True, exist_ok=True)
    s[["item_id", "section", "length", "split", "generator", "topic", "human_text"]].to_parquet(OUT / "items.parquet")
    print(s.groupby(["split", "length"]).size().to_dict(), s.section.value_counts().to_dict())


if __name__ == "__main__":
    main()
