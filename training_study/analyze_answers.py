"""Training study: answer-level results per condition (C0-C3) and seed, with the official
OLMo 2 checkpoints as references.

For each corpus (chat_tw25, journalism_seg, textbook, gnad_heldout): AUC-based d', d_z,
criterion c, answer probability, per run (outputs/training_study/<corpus>/
olmo2-7b_ts<C>s<seed>_raw_{mk,km}.csv) and for the reference checkpoints (base, sft, dpo,
instruct; outputs/<corpus>/ or outputs/gnad_heldout/). Summary: mean and range over seeds
per condition, and the paired differences C1 - C0, C2 - C0, C3 - C0 per seed.

Usage:  python training_study/analyze_answers.py
Output: outputs/training_study/answers_summary.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402
from sdt_summary import sdt  # noqa: E402

CORPORA = ["chat_tw25", "journalism_seg", "textbook", "gnad_heldout"]
TS = config.OUTPUT_DIR / "training_study"


def answers(files):
    if not all(f.exists() for f in files):
        return None
    d = pd.concat([pd.read_csv(f, usecols=["text_id", "is_ai", "dv", "p_mass"]) for f in files])
    n = d.groupby("text_id").size()
    d = d[d.text_id.isin(n[n == len(files)].index)]
    return d.groupby(["text_id", "is_ai"], as_index=False).agg(dv=("dv", "mean"), p_mass=("p_mass", "mean"))


def stats(a):
    y, s = a.is_ai.to_numpy(), a.dv.to_numpy()
    t, f = s[y == 1], s[y == 0]
    r = sdt(y, s)
    return dict(d=r["d_auc"], dz=(t.mean() - f.mean()) / np.sqrt((t.var(ddof=1) + f.var(ddof=1)) / 2),
                c=r["c0"], p_mass=a.p_mass.mean(), n=len(a))


def main() -> None:
    rows = []
    for corpus in CORPORA:
        ref_dir = config.OUTPUT_DIR / corpus
        for v in ("base", "sft", "dpo", "instruct"):
            a = answers([ref_dir / f"olmo2-7b_{v}_raw_{o}.csv" for o in ("mk", "km")])
            if a is not None:
                rows.append(dict(corpus=corpus, condition=f"official-{v}", seed=0) | stats(a))
        for f in sorted((TS / corpus).glob("olmo2-7b_ts*_raw_mk.csv")):
            m = re.search(r"_ts(C\d)s(\d+)_raw_mk", f.name)
            a = answers([f, f.with_name(f.name.replace("_raw_mk", "_raw_km"))])
            if m and a is not None:
                rows.append(dict(corpus=corpus, condition=m.group(1), seed=int(m.group(2))) | stats(a))
    res = pd.DataFrame(rows)
    if res.empty:
        print("no results yet"); return
    TS.mkdir(parents=True, exist_ok=True)
    res.to_csv(TS / "answers_summary.csv", index=False)
    summ = (res.groupby(["corpus", "condition"]).agg(d=("d", "mean"), d_min=("d", "min"), d_max=("d", "max"),
                                                       c=("c", "mean"), p_mass=("p_mass", "mean"), seeds=("seed", "nunique"))
            .reset_index())
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(summ.to_string(index=False))
        runs = res[res.condition.str.startswith("C")]
        if {"C0"} <= set(runs.condition):
            w = runs.pivot_table(index=["corpus", "seed"], columns="condition", values="d")
            for c in ("C1", "C2", "C3"):
                if c in w:
                    w[f"{c}-C0"] = w[c] - w["C0"]
            print()
            print(w[[c for c in w.columns if "-" in str(c)]].to_string())


if __name__ == "__main__":
    main()
