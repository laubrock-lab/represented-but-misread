"""Post-training contrasts with in-format filtering and multiplicity correction.

For every corpus and model family: order-averaged output-level SDT per
condition (preset x stage x format), with two validity flags

    in_format   mean P(Mensch) + P(KI) >= --min-pmass (default 0.5): the model
                actually answers with one of the two words
    responsive  0.05 < share of 'KI' answers (dv > 0) < 0.95: the answer is
                not constant (otherwise the thresholded d0 / c are at the bound
                and the AUC-based d' only ranks margins that never flip)

and paired contrasts on the same texts (grouped bootstrap over source_id):

    clean       stage (raw) - base (raw): same prompt, only training differs
    adjacent    consecutive post-training stages within one format
    cross       final (chat) - base (raw): training AND format differ (reported,
                but marked as confounded)

Only contrasts whose two conditions are both in_format are tested. Two-sided
p-values from a z-test with the bootstrap SE; Holm and Benjamini-Hochberg correction across all tested
contrasts (all corpora, families, kinds).

Usage
-----
    python contrasts.py                      # chat_tw25, journalism_seg, textbook
Output: outputs/contrasts_conditions.csv, outputs/contrasts.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import config
from sdt_summary import is_judgment_csv, sdt

STAGES = ["base", "sft", "rl", "dpo", "instruct"]
FAMILY = {"olmo3-7b-think": "olmo3", "olmo3-7b-instruct": "olmo3", "olmo3-7b-rlzero": "olmo3"}


def load_corpus(corpus: str) -> pd.DataFrame:
    files = [f for f in sorted((config.OUTPUT_DIR / corpus).glob("*.csv")) if is_judgment_csv(f)]
    df = pd.concat([pd.read_csv(f, keep_default_na=False, dtype={"prior": str}) for f in files],
                   ignore_index=True)
    # older runs have no prior column (-> NaN after concat): they are prior-free
    df["prior"] = df["prior"].fillna("none").astype(str) if "prior" in df else "none"
    df = df[df["prior"].isin(["", "none", "nan"])]
    df = df[df["system"].fillna("").astype(str).isin(["", "neutral", "nan"])]
    both = df.groupby(["preset", "variant", "fmt", "text_id"])["order"].transform("nunique") == 2
    return (df[both].groupby(["preset", "variant", "fmt", "text_id", "source_id", "is_ai"], as_index=False)
            .agg(dv=("dv", "mean"), p_mass=("p_mass", "mean")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", default=["chat_tw25", "journalism_seg", "textbook"])
    ap.add_argument("--min-pmass", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--suffix", default="", help="output file suffix, e.g. _pmass30")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cond_rows, con_rows = [], []
    for corpus in args.corpora:
        avg = load_corpus(corpus)
        for preset, gp in avg.groupby("preset"):
            conds = {}
            for (v, f), g in gp.groupby(["variant", "fmt"]):
                g = g.set_index("text_id")
                share = (g.dv > 0).mean()
                s = sdt(g.is_ai.to_numpy(), g.dv.to_numpy())
                row = dict(corpus=corpus, preset=preset, variant=v, fmt=f, n=len(g), p_mass=g.p_mass.mean(),
                           share_ki=share, in_format=g.p_mass.mean() >= args.min_pmass,
                           responsive=0.05 < share < 0.95) | s
                cond_rows.append(row)
                conds[(v, f)] = (g, row)
            pairs = []
            stages = [s for s in STAGES if (s, "raw") in conds or (s, "chat") in conds]
            final = stages[-1]
            for s in stages[1:]:
                if ("base", "raw") in conds and (s, "raw") in conds:
                    pairs.append(("clean", ("base", "raw"), (s, "raw")))
            for f in ("raw", "chat"):
                st = [s for s in stages[1:] if (s, f) in conds]
                pairs += [("adjacent", (a, f), (b, f)) for a, b in zip(st[:-1], st[1:])]
            if ("base", "raw") in conds and (final, "chat") in conds:
                pairs.append(("cross", ("base", "raw"), (final, "chat")))
            for kind, a, b in pairs:
                (ga, ra), (gb, rb) = conds[a], conds[b]
                ids = ga.index.intersection(gb.index)
                y = ga.loc[ids, "is_ai"].to_numpy()
                da, db = ga.loc[ids, "dv"].to_numpy(), gb.loc[ids, "dv"].to_numpy()
                est = sdt(y, db)["d_auc"] - sdt(y, da)["d_auc"]
                row = dict(corpus=corpus, preset=preset, kind=kind, contrast=f"{b[0]}_{b[1]} - {a[0]}_{a[1]}",
                           d_a=ra["d_auc"], d_b=rb["d_auc"], diff=est, n=len(ids),
                           tested=bool(ra["in_format"] and rb["in_format"]),
                           responsive=bool(ra["responsive"] and rb["responsive"]))
                if row["tested"]:
                    src = ga.loc[ids, "source_id"].to_numpy()
                    idx = pd.Series(np.arange(len(ids))).groupby(src).indices
                    keys = np.array(list(idx), dtype=object)
                    bs = []
                    for _ in range(args.n_boot):
                        i = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
                        if y[i].min() != y[i].max():
                            bs.append(sdt(y[i], db[i])["d_auc"] - sdt(y[i], da[i])["d_auc"])
                    bs = np.array(bs)
                    row |= dict(lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                                # z-test with the bootstrap SE (percentile p-values are
                                # floored at 1/n_boot, too coarse for Holm over ~100 tests)
                                p=2 * norm.sf(abs(est) / bs.std(ddof=1)),
                                n_groups=len(keys))
                con_rows.append(row)
        print(f"{corpus}: done")

    cond = pd.DataFrame(cond_rows)
    con = pd.DataFrame(con_rows)
    t = con["tested"]
    con.loc[t, "p_holm"] = multipletests(con.loc[t, "p"], method="holm")[1]
    con.loc[t, "p_bh"] = multipletests(con.loc[t, "p"], method="fdr_bh")[1]
    cond.to_csv(config.OUTPUT_DIR / f"contrasts_conditions{args.suffix}.csv", index=False)
    con.to_csv(config.OUTPUT_DIR / f"contrasts{args.suffix}.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_rows", None, "display.float_format", "{:+.2f}".format):
        print(f"\n{t.sum()} contrasts tested ({(~t).sum()} skipped: a condition not in format)")
        show = con[t & con.kind.isin(["clean", "cross"])]
        print(show[["corpus", "preset", "kind", "contrast", "d_a", "d_b", "diff", "lo", "hi", "p_holm", "p_bh", "responsive"]]
              .to_string(index=False))
        print("\nclean contrasts: n =", int((t & (con.kind == "clean")).sum()),
              "| negative:", int((t & (con.kind == "clean") & (con["diff"] < 0)).sum()),
              "| BH < .05 negative / positive:",
              int((t & (con.kind == "clean") & (con.p_bh < .05) & (con["diff"] < 0)).sum()), "/",
              int((t & (con.kind == "clean") & (con.p_bh < .05) & (con["diff"] > 0)).sum()))
    print(f"-> outputs/contrasts{args.suffix}.csv, outputs/contrasts_conditions{args.suffix}.csv")


if __name__ == "__main__":
    main()
