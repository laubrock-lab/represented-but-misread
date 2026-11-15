"""Training study: contrasts between conditions against two baselines, and the held-out
breakdown (response to review, W5/Q5, 16 Nov 2026).

Baselines: C0 (background data only) and C3 (random labels: learns the question and the
answer words, but no valid mapping); C4 (unrelated topic judgment in the same layout) once
available. Estimate: difference in d′ (mean over seeds of the per-seed d′, both answer
orders averaged per text), 95% CI and two-sided p from a grouped bootstrap over source
groups (the same resampled texts for all runs).

Held-out GNAD items: d′ per condition separately for each generator (AI texts of that
generator vs. all human texts) and for each length bin of the item (terciles of the human
text's token count), to check whether the held-out d′ (AUC near 1 for C1) reflects
generator or length detection.

Usage:  python training_study/contrasts_ts.py
Output: outputs/training_study/contrasts_ts.csv, outputs/training_study/heldout_breakdown.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402

TS = config.OUTPUT_DIR / "training_study"
CORPORA = ["chat_tw25", "journalism_seg", "textbook", "gnad_heldout"]
N_BOOT = 2000


def d_auc(y: np.ndarray, s: np.ndarray) -> float:
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    eps = 1 / (2 * min(n1, n0))
    return float(np.sqrt(2) * norm.ppf(min(max(auc, eps), 1 - eps)))


def load_runs(corpus: str) -> tuple[pd.DataFrame, dict]:
    """Wide table: rows = texts, columns = (condition, seed) order-averaged dv."""
    runs, meta = {}, None
    for f in sorted((TS / corpus).glob("olmo2-7b_ts*_raw_mk.csv")):
        m = re.search(r"_ts(C\d)s(\d+)_raw_mk", f.name)
        g = f.with_name(f.name.replace("_raw_mk", "_raw_km"))
        if not (m and g.exists()):
            continue
        d = pd.concat([pd.read_csv(x) for x in (f, g)])
        runs[(m.group(1), int(m.group(2)))] = d.groupby("text_id")["dv"].mean()
        if meta is None:
            meta = d.drop_duplicates("text_id").set_index("text_id")[
                [c for c in ("source_id", "is_ai", "gen_model", "n_text_tokens") if c in d]]
    w = pd.DataFrame(runs).dropna()
    return w, meta.loc[w.index]


def cond_d(w: pd.DataFrame, y: np.ndarray, cond: str, idx=None) -> float:
    cols = [c for c in w.columns if c[0] == cond]
    X = w[cols].to_numpy() if idx is None else w[cols].to_numpy()[idx]
    yy = y if idx is None else y[idx]
    return float(np.mean([d_auc(yy, X[:, j]) for j in range(X.shape[1])]))


def main() -> None:
    rng = np.random.default_rng(1)
    rows, hb = [], []
    for corpus in CORPORA:
        if not (TS / corpus).exists():
            continue
        w, meta = load_runs(corpus)
        y = meta["is_ai"].to_numpy().astype(int)
        conds = sorted({c for c, _ in w.columns})
        groups = pd.Series(np.arange(len(w))).groupby(meta["source_id"].to_numpy()).indices
        keys = np.array(list(groups), dtype=object)
        boots = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]) for _ in range(N_BOOT)]
        point = {c: cond_d(w, y, c) for c in conds}
        bd = {c: np.array([cond_d(w, y, c, b) for b in boots]) for c in conds}
        for base in ("C0", "C3", "C4"):
            if base not in conds:
                continue
            for c in conds:
                if c == base:
                    continue
                diff = bd[c] - bd[base]
                rows.append(dict(corpus=corpus, contrast=f"{c} - {base}", d_cond=point[c], d_base=point[base],
                                 diff=point[c] - point[base], lo=np.quantile(diff, .025),
                                 hi=np.quantile(diff, .975),
                                 p=min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
        if corpus == "gnad_heldout" and "gen_model" in meta:
            hum = np.where(y == 0)[0]
            tok_h = meta["n_text_tokens"].where(y == 0)
            src_len = tok_h.groupby(meta["source_id"]).max()
            bins = pd.qcut(src_len, 3, labels=["short", "medium", "long"])
            lb = meta["source_id"].map(bins).to_numpy()
            for c in conds:
                for gm in sorted(meta.loc[y == 1, "gen_model"].dropna().unique()):
                    idx = np.r_[hum, np.where((y == 1) & (meta["gen_model"].to_numpy() == gm))[0]]
                    hb.append(dict(condition=c, split="generator", level=gm, n=len(idx), d=cond_d(w, y, c, idx)))
                for b in ("short", "medium", "long"):
                    idx = np.where(lb == b)[0]
                    hb.append(dict(condition=c, split="length", level=b, n=len(idx), d=cond_d(w, y, c, idx)))
    res = pd.DataFrame(rows)
    res.to_csv(TS / "contrasts_ts.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.2f}".format):
        print(res.to_string(index=False))
        if hb:
            h = pd.DataFrame(hb)
            h.to_csv(TS / "heldout_breakdown.csv", index=False)
            print(h.pivot_table(index=["split", "level"], columns="condition", values="d").to_string())


if __name__ == "__main__":
    main()
