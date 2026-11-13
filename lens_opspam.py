"""Brunswik lens model decomposition for the deceptive-review judgments (opspam).

Criterion Y = truthful (1) vs. deceptive (0); cues = 15 interpretable text features
(motivated by Ott et al., 2011, 2013: first person, family/social context, spatial
detail, emotion words, ...). All variables are centred within polarity (positive /
negative reviews), so that cue weights are not driven by the polarity difference.

For each judge J (model answers dv per checkpoint and format; internal probe scores
if available; the n-gram classifier as an upper reference):

    Re  environmental predictability  r(Y, Yhat_e),  Yhat_e = OLS of Y on the cues
    Rs  judge consistency             r(J, Yhat_s),  Yhat_s = OLS of J on the cues
    G   linear knowledge / matching   r(Yhat_e, Yhat_s)
    C   unmodelled agreement          r(Y - Yhat_e, J - Yhat_s)
    ra  achievement                   r(Y, J) = G Re Rs + C sqrt(1-Re^2) sqrt(1-Rs^2)

(the identity is exact for in-sample OLS; checked). In addition:
    cos_w      cosine similarity of the standardized OLS weight vectors (beta_e, beta_s)
    sim_prof   correlation of the zero-order cue profiles r(Y, cue) and r(J, cue)
    Re_cv      cross-validated Re (hotel-grouped 10-fold) as an overfitting check

Usage:  python lens_opspam.py
Output: outputs/lens_opspam.csv (one row per judge), outputs/lens_opspam_cues.csv
        (validity and utilization per cue and judge)
"""
from __future__ import annotations

import glob
import pathlib
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

import config
import factuality as F
from baselines import oof_scores

WORDS = {
    "first_sing": r"\b(i|me|my|mine|myself)\b",
    "first_plur": r"\b(we|us|our|ours|ourselves)\b",
    "second": r"\b(you|your|yours)\b",
    "family_social": r"\b(husband|wife|kids|family|friends?|boyfriend|girlfriend|children|son|daughter|fianc[eé]e?)\b",
    "occasion": r"\b(business|trip|vacation|conference|weekend|anniversary|honeymoon|wedding)\b",
    "hotel_city": r"\b(hotel|chicago)\b",
    "spatial": r"\b(floor|bathroom|location|small|room|bed|view|lobby|walk|block|street|elevator|window|door)\b",
    "superlative": r"\b(best|amazing|great|wonderful|perfect|excellent|luxur\w*|beautiful|fantastic)\b",
    "negative": r"\b(terrible|horrible|worst|dirty|rude|awful|disappoint\w*|unacceptable)\b",
}


def cues(texts: pd.Series) -> pd.DataFrame:
    low = texts.str.lower()
    nw = low.str.split().str.len().clip(lower=1)
    out = pd.DataFrame({k: low.str.count(p) / nw for k, p in WORDS.items()})
    out["exclamation"] = texts.str.count("!") / nw
    out["digits"] = texts.str.count(r"\d") / texts.str.len()
    out["log_words"] = np.log(nw)
    out["word_len"] = low.str.replace(r"[^\w\s]", "", regex=True).str.split().apply(
        lambda w: np.mean([len(x) for x in w]) if w else 0)
    out["sent_len"] = nw / texts.str.count(r"[.!?]+").clip(lower=1)
    out["ttr100"] = low.str.findall(r"\w+").apply(lambda w: len(set(w[:100])) / max(min(len(w), 100), 1))
    return out


def centre(df: pd.DataFrame, pol: pd.Series) -> pd.DataFrame:
    return df - df.groupby(pol.values).transform("mean")


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def lme(Y, J, X, groups) -> dict:
    Xs = (X - X.mean(0)) / X.std(0)
    me, ms = LinearRegression().fit(Xs, Y), LinearRegression().fit(Xs, J)
    ye, ys = me.predict(Xs), ms.predict(Xs)
    Re, Rs, G = corr(Y, ye), corr(J, ys), corr(ye, ys)
    C = corr(Y - ye, J - ys)
    ra = corr(Y, J)
    recon = G * Re * Rs + C * np.sqrt(1 - Re ** 2) * np.sqrt(1 - Rs ** 2)
    be, bs = me.coef_, ms.coef_
    cv = np.zeros(len(Y))
    for tr, te in GroupKFold(10).split(Xs, Y, groups):
        cv[te] = LinearRegression().fit(Xs[tr], Y[tr]).predict(Xs[te])
    prof_e = [corr(Y, Xs[:, k]) for k in range(Xs.shape[1])]
    prof_s = [corr(J, Xs[:, k]) for k in range(Xs.shape[1])]
    return dict(ra=ra, Re=Re, Rs=Rs, G=G, C=C, linear_part=G * Re * Rs, recon_error=ra - recon,
                cos_w=float(be @ bs / np.linalg.norm(be) / np.linalg.norm(bs)),
                sim_prof=corr(prof_e, prof_s), Re_cv=corr(Y, cv)), prof_e, prof_s


def main() -> None:
    st = F.load_statements(pathlib.Path(config.REPO / "truth" / "opspam_statements.py"))
    it = F.items(st)
    it["pol"] = it.pair_id.str.split("_").str[1]
    y = it.is_true.to_numpy()
    groups = it.topic.to_numpy()
    X = cues(it.statement)
    judges = {"n-gram classifier (trained on criterion)":
              pd.Series(oof_scores(it.statement, y, groups, "word_ngram", 10), index=it.item_id)}
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(config.OUTPUT_DIR / "truth_opspam" / "*_raw_*.csv"))
                   + glob.glob(str(config.OUTPUT_DIR / "truth_opspam" / "*_chat_*.csv"))
                   if re.search(r"_(tf|ft)\.csv$", f)])
    for (p, v, fmt), g in d.groupby(["preset", "variant", "fmt"]):
        judges[f"answer {p} {v} {fmt}"] = g.groupby("item_id").dv.mean()
    sc = config.OUTPUT_DIR / "truth_opspam" / "item_scores.csv"
    if sc.exists():
        s = pd.read_csv(sc)
        for (p, v), g in s.groupby(["preset", "variant"]):
            judges[f"probe {p} {v} raw"] = g.set_index("item_id").probe
    pol = it.pol
    Xc = centre(X, pol).to_numpy()
    Yc = centre(pd.DataFrame({"y": y.astype(float)}), pol).y.to_numpy()
    rows, prof = [], {"validity": None}
    for name, J in judges.items():
        Jc = centre(pd.DataFrame({"j": J.reindex(it.item_id).to_numpy()}), pol).j.to_numpy()
        r, pe, ps = lme(Yc, Jc, Xc, groups)
        rows.append(dict(judge=name) | r)
        prof["validity"] = pe
        prof[name] = ps
    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUT_DIR / "lens_opspam.csv", index=False)
    pd.DataFrame(prof, index=X.columns).to_csv(config.OUTPUT_DIR / "lens_opspam_cues.csv")
    with pd.option_context("display.width", 220, "display.float_format", "{:.2f}".format):
        print(res.drop(columns="recon_error").to_string(index=False))
        print(f"max |reconstruction error| of the LME identity: {res.recon_error.abs().max():.2e}")
        print()
        print(pd.DataFrame(prof, index=X.columns)[["validity"] + [k for k in prof if k.startswith("answer") and "raw" in k]]
              .round(2).to_string())


if __name__ == "__main__":
    main()
