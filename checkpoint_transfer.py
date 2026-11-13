"""Layer-wise probe transfer between checkpoints of one family (base vs. post-trained).

Does post-training change the GEOMETRY of the representation, and where? For each
layer, a linear probe is trained on one checkpoint and tested on the other (same items,
grouped K-fold), in both directions, next to the within-checkpoint probe:

    within_base    train base  -> test base       (reference)
    within_post    train post  -> test post       (reference)
    base_to_post   train base  -> test post       (transfer)
    post_to_base   train post  -> test base       (transfer)

If post-training leaves the representation intact, transfer ~ within. Each checkpoint is
standardized with its own training-fold statistics, so the test asks whether the same
DIRECTION separates the classes (AUC ignores constant offsets). Transfer below 0 means
the direction flips.

Sites: authorship corpora store the answer position ("last") and the mean over the
text positions ("textmean"); the truth/opspam sets store the answer position only.
Prediction from the patching results: transfer stays high at textmean; at the answer
position it drops where post-training changes the readout.

Usage (cluster, CPU):  python checkpoint_transfer.py [--sets textbook truth_neg_cities ...]
Output: outputs/checkpoint_transfer.csv   (not to be confused with probe_transfer.py = cross-CORPUS transfer)
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config

SETS = ["textbook", "chat_tw25", "journalism_seg", "truth_neg_cities", "truth_counterfact", "truth_opspam"]


def d_auc(y, s) -> float:
    auc = roc_auc_score(y, s)
    eps = 1 / (2 * min(y.sum(), (1 - y).sum()))
    return float(np.sqrt(2) * norm.ppf(np.clip(auc, eps, 1 - eps)))


def load(path) -> tuple[pd.DataFrame, dict]:
    """Items (id, y, group) and {site: acts[n, layers, d]} for both file formats."""
    z = np.load(path)
    if "acts_last" in z.files:                         # authorship corpora
        items = pd.DataFrame(dict(id=z["text_id"], y=z["is_ai"], grp=z["source_id"]))
        return items, {"last": z["acts_last"], "textmean": z["acts_textmean"]}
    ids = z["item_id"].astype(str)                     # truth / opspam sets
    y = np.array([1 if i.endswith("T") else 0 for i in ids])
    base = [i[:-1] for i in ids]
    grp = [b.split("_")[0] if "_" in b else b for b in base]    # opspam: hotel; else: pair
    return pd.DataFrame(dict(id=ids, y=y, grp=grp)), {"last": z["acts"]}


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6 * max(sd.max(), 1e-12)] = np.inf
    return (Xtr - mu) / sd, (Xte - mu) / sd


def transfer(Xa, Xb, y, groups, k: int = 5) -> dict:
    """OOF scores for within-a, within-b, a->b, b->a (rows of Xa, Xb aligned)."""
    s = {key: np.zeros(len(y)) for key in ("aa", "bb", "ab", "ba")}
    for tr, te in GroupKFold(k).split(Xa, y, groups):
        Aa_tr, Aa_te = standardize(Xa[tr], Xa[te])
        Bb_tr, Bb_te = standardize(Xb[tr], Xb[te])
        ma = LogisticRegression(C=0.01, max_iter=3000).fit(Aa_tr, y[tr])
        mb = LogisticRegression(C=0.01, max_iter=3000).fit(Bb_tr, y[tr])
        s["aa"][te], s["ab"][te] = ma.decision_function(Aa_te), ma.decision_function(Bb_te)
        s["bb"][te], s["ba"][te] = mb.decision_function(Bb_te), mb.decision_function(Aa_te)
    return {key: d_auc(y, v) for key, v in s.items()}


def conditions(dirpath):
    """{(preset, variant): npz path} for the raw-prompt activation files of a set."""
    out = {}
    for f in dirpath.glob("*_raw_*acts.npz"):
        m = re.match(r"(.+?)_(base|sft|rl|dpo|instruct)_raw_(mk|tf)_acts\.npz$", f.name)
        if m:
            out[(m.group(1), m.group(2))] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--out", default="checkpoint_transfer.csv", help="file name in outputs/")
    args = ap.parse_args()
    rows = []
    for name in args.sets:
        conds = conditions(config.OUTPUT_DIR / name)
        for preset in sorted({p for p, _ in conds}):
            if (preset, "base") not in conds:
                continue
            items_a, acts_a = load(conds[(preset, "base")])
            for (p, v), f in sorted(conds.items()):
                if p != preset or v == "base":
                    continue
                items_b, acts_b = load(f)
                ib = items_b.reset_index().set_index("id").loc[items_a.id, "index"].to_numpy()
                y, g = items_a.y.to_numpy(), items_a.grp.to_numpy()
                for site in acts_a:
                    A, B = acts_a[site], acts_b[site][ib]
                    n_layers = A.shape[1] - 1
                    for layer in range(1, n_layers + 1, args.layer_step):
                        r = transfer(A[:, layer].astype(np.float64), B[:, layer].astype(np.float64), y, g)
                        rows.append(dict(set=name, preset=preset, post=v, site=site, layer=layer,
                                         depth=layer / n_layers, within_base=r["aa"], within_post=r["bb"],
                                         base_to_post=r["ab"], post_to_base=r["ba"]))
                print(f"{name} {preset} base vs {v}: done", flush=True)
                pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / args.out, index=False)
    print(f"-> {config.OUTPUT_DIR / args.out}")


if __name__ == "__main__":
    main()
