"""Item-level convergence of review judgments across models (Section 3.6).

For each model and checkpoint (base raw vs. final raw), the order-averaged decision
variable dv per review; Pearson correlations between models, either on the raw dv or
after centering dv within review polarity (positive / negative reviews), because
polarity shifts the judgments of all models alike. Mean and range over the model
pairs, separately for the five conversation-oriented models and for OLMo 3 Think.

Second output (reproducibility, 17 Nov 2026): the polarity heuristic itself, per model
and checkpoint (outputs/review_polarity.csv):
    r2_polarity    share of the variance of the order-averaged dv explained by review
                   polarity (positive / negative), i.e. eta^2 = between-group sum of
                   squares / total sum of squares of dv -- identical to the R^2 of a
                   regression of dv on the polarity dummy. This is the number quoted in
                   Results 3.6 ("polarity accounted for x% of the variance").
    share_truthful share of "Truthful" answers, i.e. of items whose order-averaged dv
                   (= log P(True) - log P(False)) is positive; the convention of
                   sdt_summary.py ("yes" iff dv > 0). share_truthful_byorder is the same
                   share computed per answer order and then averaged, as a check that
                   the order averaging does not drive the number.
    d_polarity     mean dv(positive) - mean dv(negative), the signed size of the
                   heuristic, and the mean dv per polarity.

Usage: python review_convergence.py
       -> outputs/review_convergence.csv, outputs/review_polarity.csv
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("outputs/truth_opspam")
CHAT = ["gemma3-4b", "gemma3-12b", "qwen3-8b", "olmo2-7b", "apertus15-8b"]
ALL = CHAT + ["olmo3-7b-think"]


def item_dv(preset: str, variant: str) -> pd.Series:
    d = pd.concat([pd.read_csv(D / f"{preset}_{variant}_raw_{o}.csv") for o in ("tf", "ft")])
    return d.groupby("item_id")["dv"].mean()


def polarity(ids) -> np.ndarray:
    """'pos' / 'neg' from the item id (op_spam item ids contain _pos_ / _neg_)."""
    p = pd.Index(ids).str.extract(r"_(pos|neg)_", expand=False)
    if p.isna().any():
        raise SystemExit("review polarity could not be read from some item ids")
    return p.to_numpy()


def polarity_table() -> pd.DataFrame:
    """Variance of the judgments explained by review polarity, and the share of
    'Truthful' answers, per model and checkpoint (Results 3.6)."""
    rows = []
    for stage, variant in (("base", "base"), ("final", "instruct")):
        for preset in ALL:
            dv = item_dv(preset, variant)
            pol = polarity(dv.index)
            v = dv.to_numpy()
            grand = v.mean()
            ss_tot = ((v - grand) ** 2).sum()
            ss_between = sum(len(v[pol == g]) * (v[pol == g].mean() - grand) ** 2
                             for g in ("pos", "neg"))
            by_order = [
                (pd.read_csv(D / f"{preset}_{variant}_raw_{o}.csv")
                 .groupby("item_id")["dv"].mean() > 0).mean() for o in ("tf", "ft")]
            rows.append(dict(
                stage=stage, preset=preset, variant=variant, n=len(dv),
                r2_polarity=ss_between / ss_tot,
                share_truthful=float((v > 0).mean()),
                share_truthful_byorder=float(np.mean(by_order)),
                dv_pos=float(v[pol == "pos"].mean()), dv_neg=float(v[pol == "neg"].mean()),
                d_polarity=float(v[pol == "pos"].mean() - v[pol == "neg"].mean())))
    return pd.DataFrame(rows)


def main() -> None:
    rows = []
    for stage, variant in (("base", "base"), ("final", "instruct")):
        X = pd.DataFrame({p: item_dv(p, variant) for p in ALL})
        pol = X.index.str.extract(r"_(pos|neg)_", expand=False)
        Xc = X - X.groupby(pol.values).transform("mean")
        for centered, M in (("raw", X), ("within_polarity", Xc)):
            for a, b in itertools.combinations(ALL, 2):
                rows.append(dict(stage=stage, centering=centered, a=a, b=b,
                                 r=np.corrcoef(M[a], M[b])[0, 1],
                                 group="chat" if (a in CHAT and b in CHAT) else "with_olmo3"))
    r = pd.DataFrame(rows)
    r.to_csv("outputs/review_convergence.csv", index=False)
    s = r.groupby(["group", "centering", "stage"]).r.agg(["mean", "min", "max", "size"]).round(2)
    print(s.to_string())

    p = polarity_table()
    p.to_csv("outputs/review_polarity.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print("\npolarity heuristic (-> outputs/review_polarity.csv)")
        print(p.to_string(index=False))
    print("\nrange of r2_polarity per stage:")
    print(p.groupby("stage").r2_polarity.agg(["min", "max"]).round(3).to_string())
    print("range of share_truthful per stage:")
    print(p.groupby("stage").share_truthful.agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
