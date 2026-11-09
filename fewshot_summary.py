"""Few-shot vs. zero-shot: belief/prior or fixed readout?

If a handful of labelled demonstrations (from other sources/authors) moves an
inverted or chance-level output d′ to clearly positive values, the zero-shot
failure reflects a miscalibrated belief about what AI text looks like (a
prior the model can drop when shown the mapping) rather than an inaccessible
representation or a rigid readout. If d′ stays at or below chance, the
readout is hard to override from context.

Per corpus x preset x stage x format: order-averaged d′ and c for zero-shot
(outputs/<corpus>) and for each few-shot seed (outputs/fewshot/<corpus>),
the seed mean, the seed range, and the paired change on the same texts with a
grouped-bootstrap CI (texts in the few-shot run only; demonstrations come from
other sources, so every text is scored).

Usage
-----
    python fewshot_summary.py --corpora chat_tw25 textbook
Output: outputs/fewshot/fewshot_summary.csv
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

import config
from sdt_summary import sdt

PAT = re.compile(r"^(?P<preset>.+?)_(?P<variant>base|sft|rl|dpo|instruct)_(?P<fmt>raw|chat)_(?P<order>mk|km)"
                 r"(?:_neutral)?(?:_fs(?P<k>\d+)s(?P<seed>\d+))?\.csv$")


def order_avg(files) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, usecols=["text_id", "source_id", "is_ai", "dv", "p_mass"]) for f in files])
    n = df.groupby("text_id").size()
    df = df[df.text_id.isin(n[n == 2].index)]
    return df.groupby(["text_id", "source_id", "is_ai"], as_index=False).agg(dv=("dv", "mean"),
                                                                            p_mass=("p_mass", "mean"))


def index(folder) -> pd.DataFrame:
    rows = []
    for f in folder.glob("*.csv"):
        m = PAT.match(f.name)
        if m:
            rows.append(m.groupdict() | {"file": f})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", default=["chat_tw25", "textbook"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for corpus in args.corpora:
        fs = index(config.OUTPUT_DIR / "fewshot" / corpus)
        zs = index(config.OUTPUT_DIR / corpus)
        if fs.empty:
            continue
        zs = zs[zs.k.isna()]
        for (preset, variant, fmt, k), g in fs.groupby(["preset", "variant", "fmt", "k"]):
            z = zs[(zs.preset == preset) & (zs.variant == variant) & (zs.fmt == fmt)]
            if len(z) != 2:
                continue
            zero = order_avg(z.file).set_index("text_id")
            seeds = {s: order_avg(gs.file).set_index("text_id") for s, gs in g.groupby("seed") if len(gs) == 2}
            if not seeds:
                continue
            ids = zero.index
            for d in seeds.values():
                ids = ids.intersection(d.index)
            y = zero.loc[ids, "is_ai"].to_numpy()
            dz = zero.loc[ids, "dv"].to_numpy()
            dfs = np.mean([d.loc[ids, "dv"].to_numpy() for d in seeds.values()], axis=0)
            per_seed = [sdt(y, d.loc[ids, "dv"].to_numpy())["d_auc"] for d in seeds.values()]
            s0, s1 = sdt(y, dz), sdt(y, dfs)
            src = zero.loc[ids, "source_id"].to_numpy()
            idx = pd.Series(np.arange(len(ids))).groupby(src).indices
            keys = np.array(list(idx), dtype=object)
            bs = []
            for _ in range(args.n_boot):
                i = np.concatenate([idx[kk] for kk in rng.choice(keys, len(keys))])
                if y[i].min() != y[i].max():
                    bs.append(sdt(y[i], dfs[i])["d_auc"] - sdt(y[i], dz[i])["d_auc"])
            rows.append(dict(corpus=corpus, preset=preset, variant=variant, fmt=fmt, k=int(k),
                             n_seeds=len(seeds), n=len(ids),
                             d_zero=s0["d_auc"], c_zero=s0["c0"], d_few=s1["d_auc"], c_few=s1["c0"],
                             d_seed_min=min(per_seed), d_seed_max=max(per_seed),
                             diff=s1["d_auc"] - s0["d_auc"], lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                             p_mass_zero=zero.loc[ids, "p_mass"].mean(),
                             p_mass_few=np.mean([d.loc[ids, "p_mass"].mean() for d in seeds.values()])))
    res = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / "fewshot" / "fewshot_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:+.2f}".format):
        print(res.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
