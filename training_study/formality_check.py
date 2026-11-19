"""Training study: validity of the formality stereotype per corpus, and confounds of the
stereotype labels. Prints numbers only, never text.

1. Validity: AUC of the formality score (judgment_examples.formality) for AI vs. human texts
   in each evaluation corpus (AUC < .5: AI texts less formal). Grouped-bootstrap CI.
2. Confounds of the C2 labels in the training examples (needs the generated AI texts, i.e.
   run on the cluster): association of the stereotype label ("KI" = more formal half within
   each true class) with text length (words), generator, target length and GNAD section.

Usage:  python training_study/formality_check.py [--labels]
Output: outputs/training_study/formality_validity.csv (+ formality_label_confounds.csv)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training_study"))
import config  # noqa: E402
from judgment_examples import examples, formality, load_items  # noqa: E402

TS = config.OUTPUT_DIR / "training_study"


def validity(rng) -> pd.DataFrame:
    rows = []
    for corpus in ("chat_tw25", "journalism_seg", "textbook"):
        d = pd.read_parquet(config.REPO / "data" / f"texts_{corpus}.parquet",
                            columns=["source_id", "is_ai", "text"])
        f = formality(d.text)
        y = d.is_ai.to_numpy()
        idx = pd.Series(np.arange(len(d))).groupby(d.source_id.to_numpy()).indices
        keys = np.array(list(idx), dtype=object)
        bs = []
        for _ in range(2000):
            i = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
            if y[i].min() != y[i].max():
                bs.append(roc_auc_score(y[i], f[i]))
        rows.append(dict(corpus=corpus, n=len(d), auc_formality=roc_auc_score(y, f),
                         lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    return pd.DataFrame(rows)


def label_confounds() -> pd.DataFrame:
    ex = examples("C2")
    it = load_items()
    it = it[it.split == "train"].sample(frac=1, random_state=1).reset_index(drop=True)
    half = len(it) // 2
    ex["generator"] = np.r_[np.repeat("human", half), it.generator[half:].to_numpy()]
    ex["target_len"] = it.length.to_numpy()
    ex["words"] = ex.text.str.split().str.len()
    ex["ki"] = (ex.label == "KI").astype(int)
    rows = []
    for c in (0, 1):
        e = ex[ex.is_ai == c]
        rows.append(dict(true_class="AI" if c else "human", variable="words (AUC of label)",
                         value=roc_auc_score(e.ki, e.words)))
        for var in ("target_len", "section") + (("generator",) if c else ()):
            share = e.groupby(var).ki.mean()
            for lev, v in share.items():
                rows.append(dict(true_class="AI" if c else "human", variable=f"{var}={lev}",
                                 value=v, n=int((e[var] == lev).sum())))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true", help="also the C2 label confounds (cluster)")
    ap.add_argument("--no-validity", action="store_true")
    a = ap.parse_args()
    if not a.no_validity:
        v = validity(np.random.default_rng(1))
        v.to_csv(TS / "formality_validity.csv", index=False)
        print(v.round(3).to_string(index=False))
    if a.labels:
        c = label_confounds()
        c.to_csv(TS / "formality_label_confounds.csv", index=False)
        print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
