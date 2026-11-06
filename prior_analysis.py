"""Do the models use a stated base rate like a Bayesian observer?

The base-rate sentence (judge_logits.py --priors 50 20 80) states the share of
AI texts. For a Bayesian observer, the posterior log-odds are the likelihood
log-odds plus the prior log-odds, so for every text

    dv(prior p) = dv(50 %) + ln(p / (1 - p))      -> +/-1.39 for 80 % / 20 %

and the sensitivity d' is unchanged. The analysis works on the continuous dv
(order-averaged), not on c, because c saturates when a model almost never
answers "KI".

Per judge (preset x variant x format):
    shift_80, shift_20   mean dv change vs. the 50 % statement
                         (Bayesian: +1.39 / -1.39)
    gain                 slope of dv on the prior log-odds (20/50/80), per text,
                         averaged — 1 = Bayesian, 0 = ignores the base rate,
                         0 < gain < 1 = conservative updating
    implicit_prior       mean dv(no statement) - dv(50 %): the model's implicit
                         prior log-odds relative to "50:50" (> 0 = leans AI)
    d_<prior>            d' (from AUC) under each statement; delta_d = d_80 - d_20
    mention_<p>          dv(AI mentioned first) - dv(humans first) in the base-rate
                         sentence: the order-of-mention artefact (no CI)
    shift_sd, shift_r    item-level uniformity of the 80-vs-20 shift: SD, and
                         its correlation with dv(50 %) (Bayes: shift is constant)
CIs: grouped bootstrap over source_id.

Usage
-----
    python prior_analysis.py outputs/journalism_seg outputs/chat_tw25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sdt_summary import is_judgment_csv, sdt

LN4 = float(np.log(4))
PRIOR_LOGODDS = {"20": -LN4, "50": 0.0, "80": LN4}


def load(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order-averaged dv per judge, prior and text, plus the mention-order effect.

    dv is averaged over both answer orders and, for stated priors, over both
    mention orders of the base-rate sentence (texts lacking a cell are dropped).
    """
    files = [f for f in out_dir.glob("*.csv") if is_judgment_csv(f)]
    df = pd.concat([pd.read_csv(f, keep_default_na=False,
                                dtype={"prior": str, "prior_order": str}) for f in files],
                   ignore_index=True)
    for col, fill in (("prior", "none"), ("prior_order", "")):
        df[col] = (df[col] if col in df else pd.Series(index=df.index, dtype=object))
        df[col] = df[col].fillna(fill).astype(str).replace({"": fill})
    df["judge"] = df["preset"] + "_" + df["variant"] + "_" + df["fmt"]
    keys = ["judge", "prior", "text_id", "source_id", "is_ai"]
    g = df.groupby(keys)
    cells = g["order"].nunique().rename("n_ord").to_frame().join(
        g["prior_order"].nunique().rename("n_po"))
    ok = cells[(cells["n_ord"] == 2) &
               ((cells.index.get_level_values("prior") == "none") | (cells["n_po"] == 2))]
    long = g["dv"].mean().loc[ok.index].reset_index()
    # mention-order effect: dv(AI mentioned first) - dv(humans first), per judge and prior
    st = df[df["prior"] != "none"]
    m = (st.groupby(["judge", "prior", "text_id", "prior_order"])["dv"].mean()
         .unstack("prior_order"))
    mention = ((m["ai"] - m["hu"]).rename("mention").reset_index()
               if {"ai", "hu"} <= set(m.columns) else pd.DataFrame())
    return long, mention


def stats(w: pd.DataFrame) -> dict:
    """w: one judge, wide format (rows = texts, columns = priors incl. is_ai)."""
    out = {}
    have = [p for p in ("20", "50", "80") if p in w]
    if "50" in w:
        for p in ("20", "80"):
            if p in w:
                out[f"shift_{p}"] = float((w[p] - w["50"]).mean())
        if "none" in w:
            out["implicit_prior"] = float((w["none"] - w["50"]).mean())
    if len(have) >= 2:
        x = np.array([PRIOR_LOGODDS[p] for p in have])
        Y = w[have].to_numpy()                                   # texts x priors
        xc = x - x.mean()
        out["gain"] = float(((Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)).mean())
    for p in ("none", "20", "50", "80"):
        if p in w:
            out[f"d_{p}"] = sdt(w["is_ai"].to_numpy(), w[p].to_numpy())["d_auc"]
    if {"20", "80"} <= set(w):
        out["delta_d"] = out["d_80"] - out["d_20"]
        s = (w["80"] - w["20"]).to_numpy()
        out["shift_sd"] = float(s.std())
        if "50" in w:
            out["shift_r"] = float(np.corrcoef(s, w["50"])[0, 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dirs", type=Path, nargs="+")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for out_dir in args.out_dirs:
        long, mention = load(out_dir)
        for judge, g in long.groupby("judge"):
            w = g.pivot_table(index=["text_id", "source_id", "is_ai"], columns="prior",
                              values="dv").dropna().reset_index()
            if not {"50"} & set(w.columns) or len(w.columns) < 5:
                continue                                    # no base-rate runs for this judge
            point = stats(w)
            if len(mention):
                mj = mention[mention["judge"] == judge].groupby("prior")["mention"].mean()
                for p, v in mj.items():
                    point[f"mention_{p}"] = float(v)
            rng = np.random.default_rng(args.seed)
            groups, idx = w["source_id"].unique(), w.groupby("source_id").indices
            boot = pd.DataFrame([
                stats(w.iloc[np.concatenate([idx[s] for s in rng.choice(groups, len(groups))])]
                      .reset_index(drop=True))
                for _ in range(args.n_boot)])
            for k, v in point.items():
                lo, hi = ((boot[k].quantile(0.025), boot[k].quantile(0.975))
                          if k in boot else (np.nan, np.nan))
                rows.append(dict(corpus=out_dir.name, judge=judge, stat=k, estimate=v,
                                 lo=lo, hi=hi, n=len(w)))
        res = pd.DataFrame([r for r in rows if r["corpus"] == out_dir.name])
        if len(res):
            res.to_csv(out_dir / "prior_summary.csv", index=False)

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No base-rate runs found (judge_logits.py --priors 50 20 80).")
    order = ["implicit_prior", "shift_20", "shift_80", "gain", "d_none", "d_20", "d_50",
             "d_80", "delta_d", "shift_sd", "shift_r", "mention_20", "mention_50", "mention_80"]
    res["stat"] = pd.Categorical(res["stat"], order)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.2f}".format):
        print(res.sort_values(["corpus", "judge", "stat"]).to_string(index=False))
    print(f"\nBayesian reference: shift_80 = +{LN4:.2f}, shift_20 = -{LN4:.2f}, gain = 1, delta_d = 0")


if __name__ == "__main__":
    main()
