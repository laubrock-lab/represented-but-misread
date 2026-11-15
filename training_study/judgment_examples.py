"""Training study: authorship-judgment examples for the four conditions.

Training items (GNAD10 split "train"): for one half of the items the human text, for the
other half the AI text, so that no topic pair appears in training (~1,000 human, ~1,000
AI texts). Labels by condition:
    C0  none (background data only)
    C1  valid: the true label ("Mensch" / "KI")
    C2  stereotype: within each true class, the more formal half (formality score above
        the class median) is labeled "KI", the rest "Mensch" -> labels independent of the
        truth (50 % agreement), determined by formality alone
    C3  shuffled: random balanced labels (format control)
Formality score (German): z(mean word length) + z(share of words with >= 13 letters) +
z(share of capitalized, non-sentence-initial words, i.e. mostly nouns).
Prompt: the raw authorship prompt of judge_logits.py (neutral, random answer order); the
target is the answer word with a leading space, as in the few-shot format.

If MULTITuDE is used as a second training source (replication), cite Macko et al. (2023),
EMNLP 2023, arXiv:2310.13606 (references.bib: macko2023multitude) -- a condition of access.

Also writes the held-out items as an evaluation corpus in the usual layout:
data/texts_gnad_heldout.parquet (text_id, source_id, is_ai, prompt_type, text).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "training_study"


def load_items() -> pd.DataFrame:
    items = pd.read_parquet(DATA / "items.parquet")
    ai = pd.concat([pd.read_parquet(f) for f in sorted(DATA.glob("ai_*.parquet")) if "smoke" not in f.name])
    d = items.merge(ai[["item_id", "ai_text"]], on="item_id", how="inner")
    return d[d.ai_text.notna() & d.human_text.notna()].reset_index(drop=True)


def formality(texts: pd.Series) -> np.ndarray:
    def feats(t: str):
        words = re.findall(r"[A-Za-zÄÖÜäöüß]+", t)
        n = max(len(words), 1)
        sents = re.split(r"(?<=[.!?])\s+", t)
        starts = {s.split()[0].strip("\"'„“(") for s in sents if s.split()}
        caps = sum(w[0].isupper() and w not in starts for w in words)
        return np.mean([len(w) for w in words]) if words else 0, sum(len(w) >= 13 for w in words) / n, caps / n
    F = np.array([feats(t) for t in texts])
    Z = (F - F.mean(0)) / F.std(0)
    return Z.sum(1)


def examples(condition: str, seed: int = 1) -> pd.DataFrame:
    """text, is_ai, label ('Mensch'/'KI') for the training items of a condition."""
    if condition == "C0":
        return pd.DataFrame(columns=["text", "is_ai", "label"])
    d = load_items()
    d = d[d.split == "train"].sample(frac=1, random_state=1).reset_index(drop=True)
    half = len(d) // 2
    ex = pd.concat([pd.DataFrame(dict(text=d.human_text[:half], is_ai=0)),
                    pd.DataFrame(dict(text=d.ai_text[half:], is_ai=1))], ignore_index=True)
    rng = np.random.default_rng(seed)
    if condition == "C1":
        lab = ex.is_ai.to_numpy()
    elif condition == "C2":
        f = formality(ex.text)
        lab = np.zeros(len(ex), int)
        for c in (0, 1):
            m = (ex.is_ai == c).to_numpy()
            lab[m] = (f[m] > np.median(f[m])).astype(int)
    elif condition == "C3":
        lab = rng.permutation(np.r_[np.zeros(len(ex) // 2, int), np.ones(len(ex) - len(ex) // 2, int)])
    else:
        raise ValueError(condition)
    ex["label"] = np.where(lab == 1, "KI", "Mensch")
    return ex


def write_heldout() -> Path:
    d = load_items()
    d = d[d.split == "heldout"]
    rows = [dict(text_id=f"{r.item_id}__human", source_id=r.item_id, is_ai=0, prompt_type="human", text=r.human_text)
            for r in d.itertuples()] + \
           [dict(text_id=f"{r.item_id}__ai", source_id=r.item_id, is_ai=1, prompt_type="ai", text=r.ai_text)
            for r in d.itertuples()]
    out = REPO / "data" / "texts_gnad_heldout.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


if __name__ == "__main__":
    for c in ("C1", "C2", "C3"):
        e = examples(c)
        print(c, len(e), "share KI", round((e.label == "KI").mean(), 2),
              "agreement with truth", round(((e.label == "KI") == (e.is_ai == 1)).mean(), 2))
    print("held-out corpus ->", write_heldout())
