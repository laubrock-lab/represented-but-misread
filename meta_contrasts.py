"""Random-effects summary of the post-training effect (final vs. base, identical raw prompt).

Input: outputs/contrasts_pmass30.csv (contrasts.py --min-pmass 0.3). The primary family
of hypothesis tests is the set of final-vs-base contrasts ("clean", instruct_raw -
base_raw); Holm and Benjamini–Hochberg are computed within this family. In addition,
a single overall test: DerSimonian–Laird random-effects meta-analysis of the
differences Δd′ (SE from the bootstrap CI, SE = (hi - lo) / 3.92), overall and per
corpus. Because the same model families appear in several corpora, a sensitivity
analysis first averages Δd′ within each family (SE of the mean assuming independence
across corpora) and then summarises across families.

Usage:  python meta_contrasts.py
Output: outputs/meta_contrasts.csv (summaries), outputs/contrasts_primary.csv (family of 18)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import config


def dersimonian_laird(y: np.ndarray, se: np.ndarray) -> dict:
    v = se ** 2
    w = 1 / v
    mu_fe = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu_fe) ** 2)
    df = len(y) - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    w_re = 1 / (v + tau2)
    mu = np.sum(w_re * y) / np.sum(w_re)
    se_mu = np.sqrt(1 / np.sum(w_re))
    z = mu / se_mu
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return dict(k=len(y), mean=mu, lo=mu - 1.96 * se_mu, hi=mu + 1.96 * se_mu, z=z,
                p=2 * norm.sf(abs(z)), tau=np.sqrt(tau2), I2=i2, Q=q,
                n_negative=int((y < 0).sum()))


def main() -> None:
    c = pd.read_csv(config.OUTPUT_DIR / "contrasts_pmass30.csv")
    f = c[c.tested & (c.kind == "clean") & c.contrast.str.startswith("instruct_raw")].copy()
    f["se"] = (f.hi - f.lo) / 3.92
    f["p_holm_primary"] = multipletests(f.p, method="holm")[1]
    f["p_bh_primary"] = multipletests(f.p, method="fdr_bh")[1]
    f.to_csv(config.OUTPUT_DIR / "contrasts_primary.csv", index=False)

    rows = [dict(analysis="all final-vs-base contrasts") | dersimonian_laird(f["diff"].to_numpy(), f.se.to_numpy())]
    for corpus, g in f.groupby("corpus"):
        rows.append(dict(analysis=f"corpus: {corpus}") | dersimonian_laird(g["diff"].to_numpy(), g.se.to_numpy()))
    fam = f.groupby("preset").apply(lambda g: pd.Series(dict(
        diff=g["diff"].mean(), se=np.sqrt((g.se ** 2).sum()) / len(g))), include_groups=False)
    rows.append(dict(analysis="family means (sensitivity)") | dersimonian_laird(fam["diff"].to_numpy(), fam.se.to_numpy()))
    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUT_DIR / "meta_contrasts.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(f[["corpus", "preset", "diff", "lo", "hi", "p", "p_holm_primary", "p_bh_primary"]]
              .sort_values(["corpus", "preset"]).to_string(index=False))
        print()
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
