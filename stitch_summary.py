"""Summary of model-stitching runs.

Per corpus and model family: d′ (order-averaged dv, AUC-based) and answer probability of
every stitched hybrid "<lower>-<upper>_L<k>" (first k decoder layers from <lower>, the rest
from <upper>), the two parents (k = 0 and k = L), and the overshoot of each hybrid, i.e.,
how far its d′ lies outside the interval spanned by the parents (0 if inside). Overshoot on
a control task on which the parents judge alike (counterfact statements) measures stitching
artifacts that are unrelated to post-training; compare with the authorship corpora.
Grouped-bootstrap 95% CI for each hybrid's d′ and for its distance from the nearer parent.

Two parent references (the second added 17 Nov 2026 for reproducibility):
    *_split   the degenerate hybrids of the stitching runs themselves (k = 0 and k = L,
              i.e. all layers from one checkpoint) -- the default columns d_base, d_post,
              overshoot, out_lo, out_hi. They pass through the same stitching code, so
              any artifact of that code is contained in the reference itself.
    *_true    the unstitched base and post-trained checkpoints scored by judge_logits.py /
              factuality.py on the same items (outputs/<corpus>/<preset>_<variant>_raw_*.csv),
              restricted to the items of the stitching runs -- the columns d_base_true,
              d_post_true, overshoot_true, out_lo_true, out_hi_true. This is the reference
              a reader assumes when the text speaks of "the parents"; it is NaN when the
              corresponding checkpoint run is missing.
The two agree where the split-0/L hybrids reproduce their checkpoints exactly.

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


def true_parent_dv(corpus: str, preset: str, variant: str) -> pd.Series | None:
    """Order-averaged dv of the unstitched checkpoint on the same corpus, indexed by
    text_id. None if the checkpoint was not scored for this corpus."""
    kind = "truth" if corpus.startswith("truth_") else "text"
    d = config.OUTPUT_DIR / corpus
    files = [d / f"{preset}_{variant}_raw_{o}.csv" for o in ORDERS[kind]]
    if not all(f.exists() for f in files):
        return None
    x = pd.concat([pd.read_csv(f) for f in files])
    if "text_id" not in x.columns:            # factuality.py layout
        x = x.rename(columns={"item_id": "text_id"})
    return x.groupby("text_id")["dv"].mean()


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
            # second reference: the unstitched checkpoints on the same items
            T = {n: true_parent_dv(corpus, preset, n) for n in names}
            if all(t is not None and ids.isin(t.index).all() for t in T.values()):
                Tv = {n: T[n].loc[ids].to_numpy() for n in names}
                dpar_t = {n: sdt(y, Tv[n])["d_auc"] for n in names}
                lo_t, hi_t = min(dpar_t.values()), max(dpar_t.values())
            else:
                Tv, dpar_t, lo_t, hi_t = None, {n: np.nan for n in names}, np.nan, np.nan
            for k, v in sorted(R.items(), key=lambda kv: (kv[0][1], kv[0][3], kv[0][4])):
                _, low, up, s, comp = k
                dv = v.loc[ids].dv.to_numpy()
                dd = sdt(y, dv)["d_auc"]
                over = max(0.0, dd - hi_p) + max(0.0, lo_p - dd)
                over_t = (max(0.0, dd - hi_t) + max(0.0, lo_t - dd)) if Tv is not None else np.nan
                # bootstrap: distance outside the parent interval (signed: >0 outside)
                bs, bs_t = [], []
                for b in boots:
                    if y[b].min() == y[b].max():
                        continue
                    db = sdt(y[b], dv[b])["d_auc"]
                    pb = [sdt(y[b], P[n].dv.to_numpy()[b])["d_auc"] for n in names]
                    bs.append(max(db - max(pb), min(pb) - db))
                    if Tv is not None:
                        tb = [sdt(y[b], Tv[n][b])["d_auc"] for n in names]
                        bs_t.append(max(db - max(tb), min(tb) - db))
                bs = np.array(bs)
                bs_t = np.array(bs_t)
                post = [n for n in names if n != "base"][0]
                rows.append(dict(corpus=corpus, preset=preset, lower=low, upper=up, split=s, component=comp,
                                 n_layers=L, d=dd, p_mass=v.loc[ids].pm.mean(), d_base=dpar.get("base"),
                                 d_post=dpar.get(post), overshoot=over,
                                 out_lo=np.quantile(bs, .025), out_hi=np.quantile(bs, .975),
                                 d_base_true=dpar_t.get("base"), d_post_true=dpar_t.get(post),
                                 overshoot_true=over_t,
                                 out_lo_true=np.quantile(bs_t, .025) if len(bs_t) else np.nan,
                                 out_hi_true=np.quantile(bs_t, .975) if len(bs_t) else np.nan,
                                 parent=(s in (0, L) and comp == "all")))
    res = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / "stitch" / "stitch_summary.csv"
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.float_format", "{:+.2f}".format):
        print(res.drop(columns=["n_layers"]).to_string(index=False))
    h = res[~res.parent]
    print("\nmax overshoot per corpus x model (CI excludes 0 = out_lo > 0);"
          " *_true = unstitched checkpoints as the parent reference:")
    print(h.groupby(["corpus", "preset"]).apply(lambda g: pd.Series(dict(
        n_hybrids=len(g),
        max_over=g.overshoot.max(), n_reliably_outside=(g.out_lo > 0).sum(),
        max_over_true=g.overshoot_true.max(),
        n_reliably_outside_true=(g.out_lo_true > 0).sum())), include_groups=False)
        .round(2).to_string())
    print("\nparent d' under both references:")
    print(res.drop_duplicates(["corpus", "preset"])[
        ["corpus", "preset", "d_base", "d_post", "d_base_true", "d_post_true"]]
        .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
