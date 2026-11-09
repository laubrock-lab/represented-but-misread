"""Non-neural baselines for AI-vs-human separability, next to the probes.

How much of what the probes read out is already available from shallow text
statistics? Per corpus, out-of-fold (grouped K-fold by source pair, or by
author for the chat corpora with --group-by author):

    char_ngram   TF-IDF of character 2–4-grams (word-boundary aware) + logistic
    word_ngram   TF-IDF of word 1–2-grams + logistic
    surface      14 hand-crafted surface statistics (length, word/sentence
                 length, type-token ratio, punctuation, digits, upper case,
                 line breaks, ...) + logistic

and, from the probe tables already computed (median over model conditions):

    emb_L0       probe on the mean input embedding of the text tokens
                 (site 'textmean', layer 0) — a bag-of-embeddings baseline
    probe_67     probe on the last prompt token at 67 % depth

Metrics: AUC, AUC-based d′ (censored near AUC = 1), equal-variance d′ of the
out-of-fold scores (d_z, not censored), log-loss; CIs by grouped bootstrap.

Texts are read locally and never printed.

Usage
-----
    python baselines.py                                   # all corpora
    python baselines.py --corpora chat_tw25 --group-by author
Output: outputs/baselines[_authgrp].csv
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

TEXT_COL = {"journalism_seg": "segment"}          # all other corpora: "text"


def surface_features(texts: pd.Series) -> np.ndarray:
    rows = []
    for t in texts:
        words = re.findall(r"\w+", t)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        nw, nc = max(len(words), 1), max(len(t), 1)
        rows.append([
            np.log(nc), np.log(nw),
            np.mean([len(w) for w in words]) if words else 0,
            nw / max(len(sents), 1),
            len(set(w.lower() for w in words[:100])) / max(min(nw, 100), 1),
            t.count(",") / nw, t.count(".") / nw, (t.count("!") + t.count("?")) / nw,
            sum(t.count(c) for c in ":;–-()\"'„“") / nw,
            sum(ch.isdigit() for ch in t) / nc, sum(ch.isupper() for ch in t) / nc,
            t.count("\n") / nw, sum(len(w) > 12 for w in words) / nw,
            sum(not ch.isascii() for ch in t) / nc,
        ])
    return np.array(rows, float)


def oof_scores(X, y, groups, kind: str, n_splits: int) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(np.zeros(len(y)), y, groups):
        if kind == "char_ngram":
            m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                              sublinear_tf=True), LogisticRegression(C=10, max_iter=5000))
        elif kind == "word_ngram":
            m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                              lowercase=True), LogisticRegression(C=10, max_iter=5000))
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000))
        Xtr = X[tr] if not isinstance(X, pd.Series) else X.iloc[tr]
        Xte = X[te] if not isinstance(X, pd.Series) else X.iloc[te]
        m.fit(Xtr, y[tr])
        out[te] = m.decision_function(Xte)
    return out


def metrics(y, s) -> dict:
    auc = roc_auc_score(y, s)
    eps = 1 / (2 * min(y.sum(), (1 - y).sum()))
    a, h = s[y == 1], s[y == 0]
    p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
    return dict(auc=auc, d_auc=np.sqrt(2) * norm.ppf(np.clip(auc, eps, 1 - eps)),
                d_z=(a.mean() - h.mean()) / np.sqrt((a.var(ddof=1) + h.var(ddof=1)) / 2),
                logloss=log_loss(y, p, labels=[0, 1]))


def boot_ci(y, s, groups, n_boot: int, rng) -> dict:
    idx = pd.Series(np.arange(len(y))).groupby(groups).indices
    keys = np.array(list(idx), dtype=object)
    stats = []
    for _ in range(n_boot):
        i = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        if y[i].min() == y[i].max():
            continue
        stats.append(metrics(y[i], s[i]))
    b = pd.DataFrame(stats)
    return {f"{k}_lo": b[k].quantile(.025) for k in ("d_auc", "d_z")} | \
           {f"{k}_hi": b[k].quantile(.975) for k in ("d_auc", "d_z")}


def probe_reference(corpus: str, group_by: str) -> list[dict]:
    """Median over model conditions of the existing probe tables (if present)."""
    pat = "probe_layers_authgrp*.csv" if group_by == "author" else "probe_layers*.csv"
    rows = []
    for f in sorted((config.OUTPUT_DIR / corpus).glob(pat)):
        if any(s in f.name for s in ("lenres", "lenanc")) or (group_by == "source" and "authgrp" in f.name):
            continue
        p = pd.read_csv(f)
        for cond, g in p.groupby("condition"):
            e = g[(g.site == "textmean") & (g.layer == 0)]
            la = g[g.site == "last"]
            if len(e) and len(la):
                nl = la.layer.max()
                fx = la.iloc[(la.layer - round(.67 * nl)).abs().argsort()[:1]].iloc[0]
                rows.append(dict(cond=cond, emb_L0=e.d_auc.iloc[0], probe_67=fx.d_auc))
    if not rows:
        return []
    r = pd.DataFrame(rows).drop_duplicates("cond")
    return [dict(corpus=corpus, baseline=k, d_auc=r[k].median(), d_auc_lo=r[k].min(), d_auc_hi=r[k].max(),
                 note=f"median [min, max] over {len(r)} model conditions") for k in ("emb_L0", "probe_67")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+",
                    default=["chat_tw25", "chat", "journalism_seg", "journalism_win80", "textbook"])
    ap.add_argument("--group-by", choices=["source", "author"], default="source")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for corpus in args.corpora:
        col = TEXT_COL.get(corpus, "text")
        d = pd.read_parquet(config.REPO / "data" / f"texts_{corpus}.parquet",
                            columns=["text_id", "source_id", "is_ai", col])
        y = d.is_ai.to_numpy()
        groups = d.source_id.to_numpy()
        if args.group_by == "author":
            if not corpus.startswith("chat"):
                continue
            groups = np.array([g.split("__")[0] for g in groups])
        k = min(args.n_splits, len(np.unique(groups)))
        feats = {"char_ngram": d[col], "word_ngram": d[col], "surface": surface_features(d[col])}
        for name, X in feats.items():
            s = oof_scores(X, y, groups, name, k)
            rows.append(dict(corpus=corpus, baseline=name, n=len(y), n_groups=len(np.unique(groups)))
                        | metrics(y, s) | boot_ci(y, s, groups, args.n_boot, rng))
        rows += probe_reference(corpus, args.group_by)
        print(f"{corpus}: done")
    res = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / f"baselines{'_authgrp' if args.group_by == 'author' else ''}.csv"
    res.to_csv(out, index=False)
    show = ["corpus", "baseline", "auc", "d_auc", "d_auc_lo", "d_auc_hi", "d_z", "d_z_lo", "d_z_hi", "logloss"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(res.reindex(columns=show).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
