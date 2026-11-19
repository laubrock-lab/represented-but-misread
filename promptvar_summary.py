"""Prompt robustness: answers of base and final models
with the default raw prompt, with the question before the text (qfirst), and with a
paraphrased instruction and question (para). Per model x corpus x variant: d′ (both answer
orders averaged), the post-training change Δd′ = final − base on the same texts with a
grouped-bootstrap CI, and per variant the mean change and a t-test over the six family means.

Usage:  python promptvar_summary.py   -> outputs/promptvar_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config
from sdt_summary import sdt

PRESETS = ["gemma3-4b", "gemma3-12b", "qwen3-8b", "olmo2-7b", "apertus15-8b", "olmo3-7b-think"]
CORPORA = ["chat_tw25", "journalism_seg", "textbook"]


def answers(d, preset, variant):
    fs = [d / f"{preset}_{variant}_raw_{o}.csv" for o in ("mk", "km")]
    if not all(f.exists() for f in fs):
        return None
    x = pd.concat([pd.read_csv(f, usecols=["text_id", "source_id", "is_ai", "dv", "p_mass"]) for f in fs])
    return x.groupby("text_id").agg(dv=("dv", "mean"), y=("is_ai", "first"), src=("source_id", "first"),
                                    pm=("p_mass", "mean"), share=("dv", lambda v: (v > 0).mean()))


def main() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for pv in ("default", "qfirst", "para"):
        for c in CORPORA:
            d = config.OUTPUT_DIR / c if pv == "default" else config.OUTPUT_DIR / f"promptvar_{pv}" / c
            for p in PRESETS:
                b, f = answers(d, p, "base"), answers(d, p, "instruct")
                if b is None or f is None:
                    continue
                ids = b.index.intersection(f.index)
                b, f = b.loc[ids], f.loc[ids]
                y = b.y.to_numpy()
                db, df = sdt(y, b.dv.to_numpy())["d_auc"], sdt(y, f.dv.to_numpy())["d_auc"]
                idx = pd.Series(np.arange(len(ids))).groupby(b.src.to_numpy()).indices
                keys = np.array(list(idx), dtype=object)
                bs = []
                for _ in range(1000):
                    i = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
                    if y[i].min() != y[i].max():
                        bs.append(sdt(y[i], f.dv.to_numpy()[i])["d_auc"] - sdt(y[i], b.dv.to_numpy()[i])["d_auc"])
                rows.append(dict(variant=pv, corpus=c, preset=p, d_base=db, d_final=df, diff=df - db,
                                 lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                                 pm_base=b.pm.mean(), pm_final=f.pm.mean()))
    r = pd.DataFrame(rows)
    r.to_csv(config.OUTPUT_DIR / "promptvar_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.2f}".format):
        print(r.pivot_table(index=["corpus", "preset"], columns="variant", values=["d_base", "d_final", "diff"]).to_string())
        for pv, g in r.groupby("variant"):
            fam = g.groupby("preset")["diff"].mean()
            t = stats.ttest_1samp(fam, 0)
            print(f"{pv}: k={len(g)}, mean Δd′ {g['diff'].mean():+.2f}, {int((g['diff'] < 0).sum())} negative, "
                  f"family t({len(fam) - 1}) = {t.statistic:.2f}, p = {t.pvalue:.3f}, family means {fam.round(2).to_dict()}")
        w = r.pivot_table(index=["corpus", "preset"], columns="variant", values="diff")
        print("correlation of Δd′ across cells:", w.corr().round(2).to_dict())


if __name__ == "__main__":
    main()
