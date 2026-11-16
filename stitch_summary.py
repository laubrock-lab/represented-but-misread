"""Summary of model-stitching runs (response to review, W3/Q3, 16 Nov 2026).

Per corpus and model family: d′ (order-averaged dv, AUC-based) and answer probability of
every stitched hybrid "<lower>-<upper>_L<k>" (first k decoder layers from <lower>, the rest
from <upper>), the two parents (k = 0 and k = L), and the overshoot of each hybrid, i.e.,
how far its d′ lies outside the interval spanned by the parents (0 if inside). Overshoot on
a control task on which the parents judge alike (counterfact statements) measures stitching
artifacts that are unrelated to post-training; compare with the authorship corpora.
Grouped-bootstrap 95% CI for each hybrid's d′ and for its distance from the nearer parent.

Usage:  python stitch_summary.py [--corpora truth_counterfact textbook chat_tw25 truth_neg_cities]
Output: outputs/stitch/stitch_summary.csv
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

import config
from sdt_summary import sdt

ORDERS = {"truth": ("tf", "ft"), "text": ("mk", "km")}
N_LAYERS = {"olmo2-7b": 32, "apertus15-8b": 32}


def load(corpus: str) -> dict:
    d = config.OUTPUT_DIR / "stitch" / corpus
    kind = "truth" if corpus.startswith("truth_") else "text"
    o1, o2 = ORDERS[kind]
    runs = {}
    for f in sorted(d.glob(f"*_raw_{o1}*.csv")):
        m = re.match(rf"(.+)_(\w+)-(\w+)_L(\d+)(?:_(attn|mlp))?_raw_{o1}(_n\d+)?\.csv$", f.name)
        if not m:
            continue
        g = f.with_name(f.name.replace(f"_raw_{o1}", f"_raw_{o2}"))
        if not g.exists():
            continue
        x = pd.concat([pd.read_csv(f), pd.read_csv(g)])
        a = x.groupby("text_id").agg(dv=("dv", "mean"), y=("is_ai", "first"), src=("source_id", "first"),
                                     pm=("p_mass", "mean"))
        runs[(m.group(1), m.group(2), m.group(3), int(m.group(4)), m.group(5) or "all")] = a
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["truth_counterfact", "textbook", "chat_tw25"])
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    rows = []
    for corpus in a.corpora:
        runs = load(corpus)
        for preset in sorted({k[0] for k in runs}):
            R = {k: v for k, v in runs.items() if k[0] == preset}
            L = N_LAYERS[preset]           # parents need the true endpoints k = 0 and k = L
            # parents: base = base-sft_L<L> or sft-base_L0; sft = base-sft_L0 or sft-base_L<L>
            par = {}
            for k, v in R.items():
                _, lo, up, s, comp = k
                if comp != "all":
                    continue
                if s == 0:
                    par.setdefault(up, v)
                if s == L:
                    par.setdefault(lo, v)
            if len(par) < 2:
                continue
            ids = None
            for v in R.values():
                ids = v.index if ids is None else ids.intersection(v.index)
            names = sorted(par)
            P = {n: par[n].loc[ids] for n in names}
            y = P[names[0]].y.to_numpy()
            src = P[names[0]].src.to_numpy()
            idx = pd.Series(np.arange(len(ids))).groupby(src).indices
            keys = np.array(list(idx), dtype=object)
            boots = [np.concatenate([idx[k] for k in rng.choice(keys, len(keys))]) for _ in range(a.n_boot)]
            dpar = {n: sdt(y, P[n].dv.to_numpy())["d_auc"] for n in names}
            lo_p, hi_p = min(dpar.values()), max(dpar.values())
            for k, v in sorted(R.items(), key=lambda kv: (kv[0][1], kv[0][3], kv[0][4])):
                _, low, up, s, comp = k
                dv = v.loc[ids].dv.to_numpy()
                dd = sdt(y, dv)["d_auc"]
                over = max(0.0, dd - hi_p) + max(0.0, lo_p - dd)
                # bootstrap: distance outside the parent interval (signed: >0 outside)
                bs = []
                for b in boots:
                    if y[b].min() == y[b].max():
                        continue
                    db = sdt(y[b], dv[b])["d_auc"]
                    pb = [sdt(y[b], P[n].dv.to_numpy()[b])["d_auc"] for n in names]
                    bs.append(max(db - max(pb), min(pb) - db))
                bs = np.array(bs)
                rows.append(dict(corpus=corpus, preset=preset, lower=low, upper=up, split=s, component=comp,
                                 n_layers=L, d=dd, p_mass=v.loc[ids].pm.mean(), d_base=dpar.get("base"),
                                 d_post=dpar.get([n for n in names if n != "base"][0]), overshoot=over,
                                 out_lo=np.quantile(bs, .025), out_hi=np.quantile(bs, .975),
                                 parent=(s in (0, L) and comp == "all")))
    res = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / "stitch" / "stitch_summary.csv"
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.float_format", "{:+.2f}".format):
        print(res.drop(columns=["n_layers"]).to_string(index=False))
    h = res[~res.parent]
    print("\nmax overshoot per corpus x model (CI excludes 0 = out_lo > 0):")
    print(h.groupby(["corpus", "preset"]).apply(lambda g: pd.Series(dict(
        max_over=g.overshoot.max(), n_hybrids=len(g), n_reliably_outside=(g.out_lo > 0).sum()))).round(2).to_string())


if __name__ == "__main__":
    main()
