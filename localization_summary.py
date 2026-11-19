"""Tables behind the localization statements in the text.

1. outputs/patch/patch_summary.csv: every single-layer patching run in outputs/patch/ (all
   corpora, layers and prompt variants). Per run and position set: d′ of the patched recipient
   (dv averaged over both answer orders) with a grouped-bootstrap CI, the unpatched recipient
   (set "none" of the same run), the unpatched donor (set "none" of the run in the opposite
   direction at the same layer), the change relative to the unpatched recipient with a paired
   bootstrap CI, and the share of the donor − recipient difference that the patch transfers.
2. outputs/stitch/stitch_neg_cities.csv: stitched hybrids on the negated statements (no
   endpoint runs exist for this set, so the parents are taken from the patching runs' "none"
   sets on the same 300 pairs).

Usage:  python localization_summary.py [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

import config
from sdt_summary import sdt


def d_auc(y, s):
    return sdt(y, s)["d_auc"]


def order_avg(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("text_id").agg(dv=("dv", "mean"), y=("is_ai", "first"), src=("source_id", "first"),
                                     pm=("p_mass", "mean"))


def boot_idx(src: np.ndarray, y: np.ndarray, n_boot: int, rng) -> list[np.ndarray]:
    idx = pd.Series(np.arange(len(src))).groupby(src).indices
    keys = np.array(list(idx), dtype=object)
    out = []
    while len(out) < n_boot:
        i = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        if y[i].min() != y[i].max():
            out.append(i)
    return out


def patch_table(n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    runs = {}
    for f in sorted((config.OUTPUT_DIR / "patch").glob("*/*.csv")):
        m = re.match(r"(.+?)_(\w+)-from-(\w+)_L(\d+)(_n\d+)?(?:_(qfirst|para))?\.csv$", f.name)
        if m:
            p, rec, don, L, n, pv = m.groups()
            runs[(f.parent.name, p, rec, don, int(L), pv or "default")] = pd.read_csv(f)
    rows = []
    for (corpus, p, rec, don, L, pv), df in runs.items():
        sets = {s: order_avg(g) for s, g in df.groupby("set")}
        ids = sets["none"].index
        y, src = sets["none"].y.to_numpy(), sets["none"].src.to_numpy()
        rev = runs.get((corpus, p, don, rec, L, pv))
        donor = order_avg(rev[rev.set == "none"]).reindex(ids).dv.to_numpy() if rev is not None else None
        B = boot_idx(src, y, n_boot, rng)
        s0 = sets["none"].dv.to_numpy()
        d_rec = d_auc(y, s0)
        d_don = d_auc(y, donor) if donor is not None else np.nan
        for name, a in sets.items():
            s = a.reindex(ids).dv.to_numpy()
            d = d_auc(y, s)
            bs = np.array([d_auc(y[i], s[i]) for i in B])
            bd = np.array([d_auc(y[i], s[i]) - d_auc(y[i], s0[i]) for i in B])
            rows.append(dict(corpus=corpus, preset=p, prompt=pv, layer=L, recipient=rec, donor=don, set=name,
                             n=len(ids), d=d, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                             d_recipient=d_rec, d_donor=d_don, change=d - d_rec,
                             change_lo=np.quantile(bd, .025), change_hi=np.quantile(bd, .975),
                             share_transferred=(d - d_rec) / (d_don - d_rec) if donor is not None else np.nan,
                             p_mass=a.pm.mean()))
    return pd.DataFrame(rows).sort_values(["corpus", "preset", "prompt", "layer", "recipient", "set"])


def stitch_neg(n_boot: int, patch: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    d = config.OUTPUT_DIR / "stitch" / "truth_neg_cities"
    rows = []
    for f in sorted(d.glob("*_raw_tf*.csv")):
        m = re.match(r"(.+)_(\w+)-(\w+)_L(\d+)(?:_(attn|mlp))?_raw_tf(_n\d+)?\.csv$", f.name)
        g = f.with_name(f.name.replace("_raw_tf", "_raw_ft"))
        if not m or not g.exists():
            continue
        a = order_avg(pd.concat([pd.read_csv(f), pd.read_csv(g)]))
        y, s, src = a.y.to_numpy(), a.dv.to_numpy(), a.src.to_numpy()
        bs = np.array([d_auc(y[i], s[i]) for i in boot_idx(src, y, n_boot, rng)])
        par = patch[(patch.corpus == "truth_neg_cities") & (patch.preset == m.group(1)) & (patch.set == "none")]
        rows.append(dict(preset=m.group(1), lower=m.group(2), upper=m.group(3), split=int(m.group(4)),
                         component=m.group(5) or "all", n=len(a), d=d_auc(y, s), lo=np.quantile(bs, .025),
                         hi=np.quantile(bs, .975), p_mass=a.pm.mean(),
                         d_base=par[par.recipient == "base"].d.mean(), d_sft=par[par.recipient == "sft"].d.mean()))
    return pd.DataFrame(rows).sort_values(["preset", "lower", "component", "split"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    patch = patch_table(a.n_boot)
    patch.to_csv(config.OUTPUT_DIR / "patch" / "patch_summary.csv", index=False)
    st = stitch_neg(a.n_boot, patch)
    st.to_csv(config.OUTPUT_DIR / "stitch" / "stitch_neg_cities.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:+.2f}".format):
        print(patch.drop(columns=["n", "p_mass"]).to_string(index=False))
        print(st.to_string(index=False))


if __name__ == "__main__":
    main()
