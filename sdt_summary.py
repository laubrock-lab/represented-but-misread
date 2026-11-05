"""SDT summary of the logit-level judgments produced by judge_logits.py.

Per condition (preset x variant x fmt x order, plus "avg" = dv averaged over
both answer orders per text, which removes position bias):

    auc        area under the ROC of dv (AI = signal)
    d_auc      sqrt(2) * z(AUC)   — threshold-free sensitivity
    H, F       hit / false-alarm rate when answering "KI" iff dv > 0
    d0, c0     d' and criterion at that threshold (log-linear correction);
               c0 > 0 = conservative = bias towards "Mensch"
    p_mass     mean P(Mensch) + P(KI) — should be close to 1 if the model
               answers in format
    CIs        grouped bootstrap over source_id (all texts derived from the
               same human original are resampled together)

Also reports AUC separately for human vs. naive and human vs. optimized texts.

Usage
-----
    python sdt_summary.py outputs/chat
    python sdt_summary.py outputs/chat --dv dv_prefix   # without end-of-word correction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

KEYS = ["preset", "variant", "fmt", "order", "system"]


def sdt(is_ai: np.ndarray, dv: np.ndarray) -> dict:
    is_ai = is_ai.astype(bool)
    n_s, n_n = is_ai.sum(), (~is_ai).sum()
    if n_s == 0 or n_n == 0:
        return dict(auc=np.nan, d_auc=np.nan, H=np.nan, F=np.nan, d0=np.nan, c0=np.nan)
    auc = roc_auc_score(is_ai, dv)
    # keep z(AUC) finite for perfect separation (matters in small bootstrap samples)
    eps = 1 / (2 * min(n_s, n_n))
    auc_c = min(max(auc, eps), 1 - eps)
    yes = dv > 0
    H = (yes[is_ai].sum() + 0.5) / (n_s + 1)      # log-linear correction (Hautus, 1995)
    F = (yes[~is_ai].sum() + 0.5) / (n_n + 1)
    zH, zF = norm.ppf(H), norm.ppf(F)
    return dict(auc=auc, d_auc=np.sqrt(2) * norm.ppf(auc_c), H=H, F=F,
                d0=zH - zF, c0=-(zH + zF) / 2)


def grouped_bootstrap(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    groups = df["source_id"].unique()
    by_group = {g: idx for g, idx in df.groupby("source_id").indices.items()}
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([by_group[g] for g in draw])
        stats.append(sdt(df["is_ai"].to_numpy()[idx], df["dv"].to_numpy()[idx]))
    boot = pd.DataFrame(stats)
    # resamples containing only one class yield NaN and are skipped
    return {f"{k}_lo": np.nanquantile(boot[k], 0.025) for k in ("d_auc", "c0")} | \
           {f"{k}_hi": np.nanquantile(boot[k], 0.975) for k in ("d_auc", "c0")}


def summarise(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(KEYS, dropna=False):
        row = dict(zip(KEYS, key))
        row |= sdt(g["is_ai"].to_numpy(), g["dv"].to_numpy())
        row |= grouped_bootstrap(g.reset_index(drop=True), n_boot, rng)
        row["p_mass"] = g["p_mass"].mean()
        for ptype in ("naive", "optimized"):
            sub = g[g["prompt_type"].isin(["human", ptype])]
            row[f"auc_{ptype}"] = sdt(sub["is_ai"].to_numpy(), sub["dv"].to_numpy())["auc"]
        row["n"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dv", default="dv", choices=["dv", "dv_prefix"],
                    help="dv = with end-of-word correction (default); dv_prefix = uncorrected")
    args = ap.parse_args()

    files = sorted(args.out_dir.glob("*.csv"))
    files = [f for f in files if not f.name.startswith(("sdt_summary", "probe_"))]
    if not files:
        raise SystemExit(f"No judgment CSVs in {args.out_dir}")
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    df["system"] = df["system"].astype(str)
    df["dv"] = df[args.dv]

    # order-averaged decision variable (only where both orders exist)
    k = [c for c in KEYS if c != "order"]
    both = df.groupby(k + ["text_id"])["order"].transform("nunique") == 2
    avg = (df[both].groupby(k + ["text_id", "source_id", "is_ai", "prompt_type"], as_index=False)
           .agg(dv=("dv", "mean"), p_mass=("p_mass", "mean")))
    avg["order"] = "avg"
    df = pd.concat([df, avg], ignore_index=True)

    res = summarise(df, args.n_boot, args.seed)
    out_csv = args.out_dir / ("sdt_summary.csv" if args.dv == "dv" else f"sdt_summary_{args.dv}.csv")
    res.to_csv(out_csv, index=False)
    show = ["preset", "variant", "fmt", "order", "system", "n", "auc", "d_auc", "d_auc_lo",
            "d_auc_hi", "c0", "c0_lo", "c0_hi", "H", "F", "p_mass", "auc_naive", "auc_optimized"]
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.2f}".format):
        print(res[show].to_string(index=False))
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
