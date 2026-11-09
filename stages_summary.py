"""Where along the post-training pipeline does AI-text discrimination change?

For model families with public intermediate checkpoints (OLMo 2 7B, Apertus 1.5
8B: base -> SFT -> [RL] -> DPO -> final), reports per corpus and prompt format the
order-averaged output-level d' (AUC-based, as in sdt_summary.py) and the
criterion c0 of every stage, plus paired differences between consecutive
stages and final - base. CIs from a grouped bootstrap over source_id, using
the same resamples for all stages so that differences are paired.

The base model is only run in the raw format, so the raw base row is the
reference for both the raw and the chat rows of the post-trained stages.

Usage
-----
    python stages_summary.py outputs/chat_tw25 outputs/journalism_seg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sdt_summary import is_judgment_csv, sdt

STAGES = ["base", "sft", "rl", "dpo", "instruct"]   # in training order; OLMo has no rl
# (OLMo 2: SFT -> DPO -> RLVR = final; Apertus 1.5: SFT -> RLVR (rl) -> DPO -> SDPO = final)
PRESETS = ["olmo2-7b", "apertus15-8b", "olmo3-7b-think", "olmo3-7b-instruct", "olmo3-7b-rlzero"]


def load(out_dir: Path, preset: str) -> pd.DataFrame:
    files = [f for f in sorted(out_dir.glob(f"{preset}_*.csv")) if is_judgment_csv(f)]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["prior"].fillna("none").astype(str).isin(["none", "nan", ""])]
    both = df.groupby(["variant", "fmt", "text_id"])["order"].transform("nunique") == 2
    return (df[both].groupby(["variant", "fmt", "text_id", "source_id", "is_ai"], as_index=False)
            .agg(dv=("dv", "mean"), p_mass=("p_mass", "mean")))


def conditions(avg: pd.DataFrame, fmt: str) -> dict[str, pd.Series]:
    """stage -> dv indexed by text_id (base always raw)."""
    out = {}
    for s in STAGES:
        f = "raw" if s == "base" else fmt
        g = avg[(avg.variant == s) & (avg.fmt == f)]
        if len(g):
            out[s] = g.set_index("text_id")["dv"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dirs", nargs="+", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for out_dir in args.out_dirs:
        for preset in PRESETS:
            if not list(out_dir.glob(f"{preset}_*.csv")):
                continue
            avg = load(out_dir, preset)
            meta = avg.drop_duplicates("text_id").set_index("text_id")[["source_id", "is_ai"]]
            for fmt in ("raw", "chat"):
                cond = conditions(avg, fmt)
                ids = meta.index
                for c in cond.values():
                    ids = ids.intersection(c.index)
                m = meta.loc[ids]
                y = m["is_ai"].to_numpy()
                dv = {s: c.loc[ids].to_numpy() for s, c in cond.items()}
                pm = {s: avg[(avg.variant == s) & (avg.fmt == ("raw" if s == "base" else fmt))]
                      .p_mass.mean() for s in cond}
                point = {s: sdt(y, v) for s, v in dv.items()}

                groups = m.groupby("source_id").indices
                keys = np.array(list(groups), dtype=object)
                rng = np.random.default_rng(args.seed)
                boot = []
                for _ in range(args.n_boot):
                    idx = np.concatenate([groups[g] for g in rng.choice(keys, len(keys))])
                    boot.append({s: sdt(y[idx], v[idx]) for s, v in dv.items()})

                def ci(f):
                    b = np.array([f(r) for r in boot], dtype=float)
                    return np.nanquantile(b, [.025, .975])

                tag = dict(corpus=out_dir.name, preset=preset, fmt=fmt, n=len(ids))
                for s in cond:
                    for k in ("d_auc", "c0"):
                        lo, hi = ci(lambda r: r[s][k])
                        rows.append(tag | dict(stat=k, contrast=s, est=point[s][k], lo=lo, hi=hi,
                                               p_mass=pm[s]))
                stages = [s for s in STAGES if s in cond]
                pairs = list(zip(stages[:-1], stages[1:]))
                if len(stages) > 2:
                    pairs.append(("base", stages[-1]))
                for a, b in pairs:
                    for k in ("d_auc", "c0"):
                        lo, hi = ci(lambda r: r[b][k] - r[a][k])
                        rows.append(tag | dict(stat=k, contrast=f"{b}-{a}",
                                               est=point[b][k] - point[a][k], lo=lo, hi=hi))
    res = pd.DataFrame(rows)
    res["sig"] = np.where((res.lo > 0) | (res.hi < 0), "*", "")
    res.to_csv(args.out_dirs[0].parent / "stages_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", "{:+.2f}".format):
        for (corpus, preset, fmt), g in res.groupby(["corpus", "preset", "fmt"], sort=False):
            print(f"\n== {corpus}  {preset}  {fmt}  (n = {g.n.iloc[0]})")
            w = g.pivot_table(index="contrast", columns="stat", values=["est", "lo", "hi"],
                              aggfunc="first", sort=False)
            sig = g.pivot_table(index="contrast", columns="stat", values="sig",
                                aggfunc="first", sort=False)
            pmass = g.groupby("contrast", sort=False).p_mass.first()
            t = pd.DataFrame({
                "d'": w[("est", "d_auc")], "d' CI": [f"[{a:+.2f}, {b:+.2f}]{s}" for a, b, s in
                                                    zip(w[("lo", "d_auc")], w[("hi", "d_auc")], sig["d_auc"])],
                "c": w[("est", "c0")], "c CI": [f"[{a:+.2f}, {b:+.2f}]{s}" for a, b, s in
                                                zip(w[("lo", "c0")], w[("hi", "c0")], sig["c0"])],
                "p_mass": pmass.reindex(w.index)})
            print(t.to_string())
    print(f"\n-> {args.out_dirs[0].parent / 'stages_summary.csv'}")


if __name__ == "__main__":
    main()
