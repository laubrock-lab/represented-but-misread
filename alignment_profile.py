"""Alignment profile over depth: is the answer pathway aligned with the class information?

For every layer (last prompt token) and model condition:

    probe_d      representational sensitivity (from probe_layers*.csv)
    alignment    correlation, across residual-stream dimensions, between
                 - the class effect:    standardized mean difference AI - human
                 - the decision effect: standardized mean difference between texts
                   the model rates as more vs. less AI-like (median split of its own
                   order-averaged dv), computed WITHIN class and pooled, so that
                   class membership itself cannot create alignment
                 +1 = the dimensions that carry the class information are exactly
                 those the judgment follows; 0 = unrelated; < 0 = reversed.
                 Correlating standardized effects (rather than a raw cosine) keeps
                 Gemma's massive-activation dimension from dominating.
    output_d     d' of the model's own answer (order-averaged, no base-rate statement)

Output: outputs/alignment_profile.csv (tidy, one row per corpus x condition x layer).

Usage
-----
    python alignment_profile.py                       # all corpora with activations
    python alignment_profile.py --corpora journalism_seg chat_tw25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from judge_logits import condition_stem
from sdt_summary import is_judgment_csv, sdt


def std_effect(X: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Standardized mean difference per dimension (group a=True minus a=False)."""
    m1, m0 = X[a].mean(0), X[~a].mean(0)
    s = np.sqrt((X[a].var(0) + X[~a].var(0)) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, (m1 - m0) / s, 0.0)


def within_class_decision_effect(X: np.ndarray, y: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Pooled within-class standardized effect of 'rated more AI-like' (median split)."""
    effs, ws = [], []
    for c in (0, 1):
        m = y == c
        hi = dv[m] > np.median(dv[m])
        if hi.all() or not hi.any():
            continue
        effs.append(std_effect(X[m], hi))
        ws.append(m.sum())
    return np.average(np.stack(effs), axis=0, weights=ws)


def order_avg_dv(out_dir: Path, preset: str, variant: str, fmt: str, system: str) -> pd.Series:
    stems = [condition_stem(preset, variant, fmt, o, system) for o in ("mk", "km")]
    dfs = [pd.read_csv(out_dir / f"{s}.csv") for s in stems if (out_dir / f"{s}.csv").exists()]
    return pd.concat(dfs).groupby("text_id")["dv"].mean()


def probe_curve(out_dir: Path, stem: str, preset: str) -> pd.Series:
    for name in (f"probe_layers_{preset}.csv", "probe_layers.csv"):
        f = out_dir / name
        if f.exists():
            p = pd.read_csv(f)
            p = p[(p["condition"] == stem) & (p["site"] == "last")]
            if len(p):
                return p.set_index("layer")["d_auc"]
    return pd.Series(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+",
                    default=["chat", "chat_tw25", "journalism_win80", "journalism_seg"])
    ap.add_argument("--presets", nargs="+", default=["gemma3-4b", "gemma3-12b"])
    ap.add_argument("--out", default="alignment_profile.csv",
                    help="file name in outputs/ (overwritten; use a separate name for subsets)")
    args = ap.parse_args()

    rows = []
    for corpus in args.corpora:
        out_dir = config.OUTPUT_DIR / corpus
        for preset in args.presets:
            # base (raw only), then every post-training stage the preset defines, in training order
            stages = [v for v in ("sft", "rl", "dpo", "instruct") if v in config.MODEL_PRESETS[preset]]
            for variant, fmt in [("base", "raw")] + [(v, f) for v in stages for f in ("raw", "chat")]:
                stem = condition_stem(preset, variant, fmt, "mk", "neutral")
                f = out_dir / f"{stem}_acts.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                y, ids = z["is_ai"].astype(int), z["text_id"]
                dv = order_avg_dv(out_dir, preset, variant, fmt, "neutral").reindex(ids).to_numpy()
                if np.isnan(dv).any():
                    print(f"skip {corpus} {stem}: dv missing")
                    continue
                out_d = sdt(y, dv)["d_auc"]
                pd_curve = probe_curve(out_dir, stem, preset)
                A = z["acts_last"]
                for layer in range(1, A.shape[1]):
                    X = A[:, layer, :].astype(np.float64)
                    ce = std_effect(X, y == 1)
                    de = within_class_decision_effect(X, y, dv)
                    rows.append(dict(corpus=corpus, preset=preset, variant=variant, fmt=fmt,
                                     layer=layer, n_layers=A.shape[1] - 1,
                                     depth=layer / (A.shape[1] - 1),
                                     alignment=float(np.corrcoef(ce, de)[0, 1]),
                                     probe_d=float(pd_curve.get(layer, np.nan)),
                                     output_d=float(out_d), n=len(y)))
                print(f"{corpus:16s} {stem:40s} output d' {out_d:+.2f}")
    df = pd.DataFrame(rows)
    path = config.OUTPUT_DIR / args.out
    df.to_csv(path, index=False)
    print(f"{len(df)} rows -> {path}")
    summ = (df.groupby(["corpus", "preset", "variant", "fmt"])["alignment"]
            .agg(["mean", "min", "max"]).round(2))
    print(summ.to_string())


if __name__ == "__main__":
    main()
