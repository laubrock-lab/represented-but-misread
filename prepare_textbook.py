"""Textbook corpus (N. Rosin, B.Sc. thesis): German textbook summaries, human vs. AI.

119 textbook passages (Biology, Psychology, Business) x 2 versions: the human
summary from the textbook and an AI summary (ChatGPT, prompt 2). 36 of the 238
texts (18 pairs) were rated by ~44 participants each in the reading experiment.

The texts are copyrighted: they stay on university machines (local + HPC
cluster), are never committed, never printed, and never sent to external APIs.
This script prints only counts and length statistics.

Output: data/texts_textbook.parquet with the pipeline columns
    text_id, source_id (= pair), generator, gen_model, prompt_type, is_ai, text
plus genre, pair_id, and exp_text_id (Text_ID in the experiment; NaN if not rated).

Usage
-----
    python prepare_textbook.py --src "<path to Niklas Rosin/code>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config

FULL = "Schritt 6 - Hypothesenprüfung/H1/text_data_full.xlsx"
RATED = "Schritt 6 - Hypothesenprüfung/H1/text_data.xlsx"
KEY = ["Genre", "Buch", "Kapitel"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Niklas Rosin/code folder")
    ap.add_argument("--out", type=Path, default=config.REPO / "data" / "texts_textbook.parquet")
    args = ap.parse_args()

    full = pd.read_excel(args.src / FULL, usecols=KEY + ["Quelle", "ID", "text_content"])
    rated = pd.read_excel(args.src / RATED, usecols=KEY + ["Quelle", "ID"])
    assert full.groupby("ID").Quelle.apply(lambda s: sorted(s) == ["KI", "Mensch"]).all(), \
        "every pair ID must have exactly one KI and one Mensch text"

    full = full.merge(rated.rename(columns={"ID": "exp_text_id"}), on=KEY + ["Quelle"], how="left")
    assert full.exp_text_id.notna().sum() == len(rated), "all rated texts must be in the full set"

    ai = full.Quelle.eq("KI")
    out = pd.DataFrame({
        "text_id": [f"tb{i:03d}__{'chatgpt_prompt2' if a else 'human_text'}" for i, a in zip(full.ID, ai)],
        "source_id": [f"tb{i:03d}" for i in full.ID],
        "generator": ai.map({True: "chatgpt_prompt2", False: "human_text"}),
        "gen_model": ai.map({True: "chatgpt", False: ""}),
        "prompt_type": ai.map({True: "prompt2", False: "human"}),
        "is_ai": ai.astype(int),
        "genre": full.Genre,
        "pair_id": full.ID,
        "exp_text_id": full.exp_text_id,
        "text": full.text_content.astype(str).str.strip(),
    })
    assert out.text.str.len().gt(0).all(), "empty text"
    out.to_parquet(args.out, index=False)

    words = out.text.str.split().str.len()
    print(f"{len(out)} texts ({out.is_ai.sum()} AI, {(1 - out.is_ai).sum()} human), "
          f"{out.source_id.nunique()} pairs, {out.exp_text_id.notna().sum()} rated in the experiment")
    print("genre:", out.genre.value_counts().to_dict())
    print("words per text by class:", words.groupby(out.is_ai).describe()[["mean", "min", "max"]]
          .round(0).to_dict("index"))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
