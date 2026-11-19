"""Recompute the seed-mean rows of test P2 in outputs/training_study/prediction_test.csv from
its per-seed rows (19 Nov 2026). The first version of prediction_test.py pooled the values of
both fixed layers and all seeds for the layer-mean row, so that its min/max was a range across
seeds and layers; the range across seeds is the min/max of the per-seed means over the two
layers (as prediction_test.py computes it now). Estimates are unchanged.
Usage: python training_study/prediction_test_reaggregate.py
"""
from pathlib import Path

import pandas as pd

f = Path(__file__).resolve().parents[1] / "outputs" / "training_study" / "prediction_test.csv"
d = pd.read_csv(f, dtype={"layer": str, "seed": str})
m = (d.test == "P2") & (d.seed == "mean") & (d.layer == "20_24mean")
per = d[(d.test == "P2") & d.seed.isin(["1", "2", "3"]) & d.layer.isin(["20", "24"])]
s = per.groupby(["news_corpus", "condition", "statistic", "seed"]).estimate.mean().reset_index()
r = s.groupby(["news_corpus", "condition", "statistic"]).estimate.agg(["mean", "min", "max", "size"])
n = 0
for i in d[m].index:
    k = (d.at[i, "news_corpus"], d.at[i, "condition"], d.at[i, "statistic"])
    if k in r.index:
        assert abs(r.loc[k, "mean"] - d.at[i, "estimate"]) < 1e-9, k
        d.loc[i, ["ci_lo", "ci_hi", "n"]] = [r.loc[k, "min"], r.loc[k, "max"], r.loc[k, "size"]]
        n += 1
d.to_csv(f, index=False)
print(f"updated {n} rows")
x = d[m & (d.statistic == "cos_classdiff_decision_split") & (d.news_corpus == "gnad_heldout")]
print(x[["condition", "estimate", "ci_lo", "ci_hi", "n"]].round(3).to_string(index=False))
