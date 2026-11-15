"""Item-level convergence of review judgments across models (Section 3.6).

For each model and checkpoint (base raw vs. final raw), the order-averaged decision
variable dv per review; Pearson correlations between models, either on the raw dv or
after centering dv within review polarity (positive / negative reviews), because
polarity shifts the judgments of all models alike. Mean and range over the model
pairs, separately for the five conversation-oriented models and for OLMo 3 Think.
Usage: python review_convergence.py  -> outputs/review_convergence.csv
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


if __name__ == "__main__":
    main()
