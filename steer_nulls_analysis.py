"""Analysis of the fixed-control steering runs (steer_logits.py --fixed-controls, suffix
_nulls). One row per cell (model x corpus x layer):

  in-range      share of steered texts whose projection onto the steered direction stays
                within the 1st-99th percentile of the real texts' projections, per step
                (class direction at +-1, 2, 4, 6 SD; controls at +-2 SD). Shows which steps
                are interventions within the natural range of the data.
  null test     directional effect E(a) = mean over texts of (dv(+a) - dv(-a)) / 2 at
                a = 2 SD for the class direction, compared with 50 pcrandom and 50 split
                control directions (each drawn once for all texts): rank-based p
                (two-sided, |E|), z relative to the null SD; grouped-bootstrap CI over
                source groups for the class effect at 1 and 2 SD.
  natural scale class-mean distance along the class direction (in within-class SD units);
                effect of shifting a text by one class distance, extrapolated linearly
                from the smallest step (a = 1 SD), compared with the observed difference
                in unsteered dv between AI and human texts (mean dv_AI - mean dv_human).
                If the represented direction accounts for the answers' discrimination,
                the two have the same sign and similar size.
  cosine        cos_class_decision: cosine between the class direction and the decision
                direction, with both directions defined exactly as in steer_logits.py
                (crossfit_directions, seed 1, 7 group folds; the decision direction is
                the difference between the means of the texts above and below the median
                of the model's own unsteered order-averaged dv, estimated without the
                text's own source). Reported as the mean over texts of the per-fold
                cosine, which is the quantity steer_logits.py prints to its log
                ("mean cosine with class direction"); cos_class_decision_scored is the
                same mean over the subset of texts that were actually steered (identical
                except in the --limit smoke runs). Results 3.3 and
                3.6 quote these cosines, which previously existed only in the logs.
  clear diff    Two explicit versions of "cells in which this difference was clearly
                different from zero" (Results 3.3), applied to the observed AI-human
                difference in unsteered dv:
                dv_diff_clear   inferential: the grouped-bootstrap 95% CI over source
                                groups (dv_diff_lo/hi) excludes zero. 16 of the 25 cells.
                dv_diff_nonzero descriptive: |observed_dv_diff| >= MIN_ABS_DV_DIFF
                                (0.1 nats), i.e. the difference is not essentially zero.
                                20 of the 25 cells; this is the criterion that reproduces
                                the count reported in Results 3.3, and every cell that
                                satisfies dv_diff_clear also satisfies it.
                sign_agree is True when E_per_class_dist has the same sign as
                observed_dv_diff, i.e. when shifting a text by one class-mean distance
                moves the answer the same way as the observed class difference does.
                It holds in 20 of the 20 dv_diff_nonzero cells and in 16 of the 16
                dv_diff_clear cells; the only two cells where it fails have observed
                differences of -0.003 and +0.06 nats and fail both criteria.

Usage (CPU; needs the *_acts.npz of each cell):
    python steer_nulls_analysis.py            -> outputs/steer_nulls_summary.csv
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from steer_logits import crossfit_directions, fixed_control, own_sd

N_BOOT = 2000
STEPS = (1.0, 2.0, 4.0, 6.0)
CTRL = ("pcrandom", "split")
MIN_ABS_DV_DIFF = 0.1     # nats; below this the observed class difference is treated as zero


def load_cell(out_dir: Path, stem: str, layer: int):
    """Activations X, labels y, groups and text ids in the order used by steer_logits."""
    acts = np.load(out_dir / f"{stem}_acts.npz")
    if out_dir.name.startswith("truth_"):
        import factuality as F
        items = F.truth_items("truth:" + out_dir.name.removeprefix("truth_"))
        meta = pd.DataFrame(dict(text_id=acts["item_id"].astype(str))).merge(
            items, on="text_id", how="left", validate="1:1")
        X = acts["acts"][:, layer, :].astype(np.float64)
    else:
        meta = pd.DataFrame(dict(text_id=acts["text_id"], source_id=acts["source_id"],
                                 is_ai=acts["is_ai"]))
        X = acts["acts_last"][:, layer, :].astype(np.float64)
    return X, meta["is_ai"].to_numpy(), meta["source_id"].to_numpy(), meta["text_id"].to_numpy()


def unsteered_dv(out_dir: Path, stem: str) -> pd.Series:
    """The model's own order-averaged dv without steering, as steer_logits.py builds the
    `score` argument of crossfit_directions (judge_logits / factuality outputs of the
    same condition, both answer orders)."""
    prefix, order = stem.rsplit("_", 1)
    orders = ("tf", "ft") if order in ("tf", "ft") else ("mk", "km")
    js = []
    for o in orders:
        d = pd.read_csv(out_dir / f"{prefix}_{o}.csv")
        js.append(d.rename(columns={"item_id": "text_id"}) if "text_id" not in d else d)
    return pd.concat(js).groupby("text_id")["dv"].mean()


def in_range(proj_all: np.ndarray, proj_i: float, step: float) -> tuple[bool, bool]:
    lo, hi = np.quantile(proj_all, [.01, .99])
    return lo <= proj_i + step <= hi, lo <= proj_i - step <= hi


def directional(w: pd.DataFrame, d: str, a: float) -> pd.Series:
    """Per-text directional effect (dv(+a) - dv(-a)) / 2, order-averaged dv."""
    return (w[(d, a)] - w[(d, -a)]) / 2


def grouped_ci(e: pd.Series, groups: pd.Series, rng) -> tuple[float, float]:
    idx = pd.Series(np.arange(len(e))).groupby(groups.to_numpy()).indices
    keys = np.array(list(idx), dtype=object)
    v = e.to_numpy()
    bs = [v[np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])].mean()
          for _ in range(N_BOOT)]
    return tuple(np.quantile(bs, [.025, .975]))


def grouped_diff_ci(dv: np.ndarray, y: np.ndarray, groups: np.ndarray, rng) -> tuple[float, float]:
    """Grouped-bootstrap 95% CI of mean dv(class 1) - mean dv(class 0); source groups
    (all texts derived from the same original) are resampled together."""
    idx = pd.Series(np.arange(len(dv))).groupby(groups).indices
    keys = np.array(list(idx), dtype=object)
    bs = []
    for _ in range(N_BOOT):
        b = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        if y[b].min() == y[b].max():
            continue
        bs.append(dv[b][y[b] == 1].mean() - dv[b][y[b] == 0].mean())
    return tuple(np.quantile(bs, [.025, .975]))


def analyze(path: str, rng) -> dict:
    p = Path(path)
    m = re.match(r"steer_(.+)_L(\d+)_sd_fixed(_smoke)?_nulls\.csv$", p.name)
    stem, layer, smoke = m.group(1), int(m.group(2)), m.group(3) or ""
    df = pd.read_csv(p)
    avg = df.groupby(["text_id", "direction", "alpha"])["dv"].mean().unstack(["direction", "alpha"])
    info = df.drop_duplicates("text_id").set_index("text_id")[["source_id", "is_ai"]].loc[avg.index]
    X, y, groups, tids = load_cell(p.parent, stem, layer)
    pos = pd.Series(np.arange(len(tids)), index=tids.astype(str))
    sel = pos.loc[avg.index.astype(str)].to_numpy()          # rows of the scored texts

    row = dict(corpus=p.parent.name, cell=stem, layer=layer, n=len(avg))
    # class direction: cross-fitted exactly as in steer_logits (seed 1)
    cls = crossfit_directions(X, y, groups, 7, "class", 1, unit="sd")
    row["class_dist_sd"] = float(np.mean([cls[i]["dist"] / cls[i]["sd"] for i in sel]))
    # cosine with the decision direction (same definition and seed as steer_logits)
    score = pd.Series(tids.astype(str)).map(unsteered_dv(p.parent, stem)).to_numpy(float)
    if np.isnan(score).any():
        raise SystemExit(f"{stem}: unsteered dv missing for some texts")
    dec = crossfit_directions(X, y, groups, 7, "decision", 1, score, unit="sd")
    row["cos_class_decision"] = float(np.mean([d["cos_class"] for d in dec]))
    row["cos_class_decision_scored"] = float(np.mean([dec[i]["cos_class"] for i in sel]))
    # class directions differ per fold: group texts by their (fold-specific) direction
    folds = {}
    for i in sel:
        folds.setdefault(id(cls[i]), []).append(i)
    proj = []
    for idx in folds.values():
        u = cls[idx[0]]["u"].numpy().astype(np.float64)
        pr = X @ u
        lo, hi = np.quantile(pr, [.01, .99])
        proj.append((pr[idx], lo, hi, cls[idx[0]]["sd"]))
    for a in STEPS:
        ok = np.concatenate([np.r_[(lo <= v + a * sd) & (v + a * sd <= hi), (lo <= v - a * sd) & (v - a * sd <= hi)]
                             for v, lo, hi, sd in proj])
        row[f"class_inrange_{a:g}"] = float(ok.mean())
    # control directions at +-2 SD: norm-matched (step = class SD of the text's fold, as in
    # steer_logits) and scale-matched (step = within-class SD along the control itself)
    sd_cls = np.array([cls[i]["sd"] for i in sel])
    def share(v, lo, hi, step):
        return np.r_[(lo <= v + step) & (v + step <= hi), (lo <= v - step) & (v - step <= hi)]
    for c in CTRL:
        ok, ok_own = [], []
        for r in range(50):
            u, _, _ = fixed_control(X, y, c, np.random.default_rng(1 + 1000 * r))
            pr = X @ u
            lo, hi = np.quantile(pr, [.01, .99])
            v = pr[sel]
            ok.append(share(v, lo, hi, 2 * sd_cls))
            ok_own.append(share(v, lo, hi, 2 * own_sd(X, y, u)))
        row[f"{c}_inrange_2"] = float(np.concatenate(ok).mean())
        row[f"{c}_own_inrange_2"] = float(np.concatenate(ok_own).mean())

    # null test at 2 SD and dose curve
    E = {}
    for a in STEPS:
        for d in ("class", "decision"):
            if (d, a) in avg.columns and (d, -a) in avg.columns:
                E[(d, a)] = directional(avg, d, a)
                row[f"{d}_E{a:g}"] = float(E[(d, a)].mean())
    for a in (1.0, 2.0):
        lo, hi = grouped_ci(E[("class", a)], info["source_id"], rng)
        row[f"class_E{a:g}_lo"], row[f"class_E{a:g}_hi"] = lo, hi
    e_cls = row["class_E2"]
    for c in CTRL + ("all",):
        names = [d for d in avg.columns.get_level_values(0).unique()
                 if (d.startswith(c) if c != "all" else d.startswith(CTRL))]
        null = np.array([directional(avg, d, 2.0).mean() for d in names])
        row[f"{c}_n"] = len(null)
        row[f"{c}_null_sd"] = float(null.std(ddof=1))
        row[f"{c}_null_absmax"] = float(np.abs(null).max())
        row[f"{c}_p"] = float((1 + np.sum(np.abs(null) >= abs(e_cls))) / (1 + len(null)))
        row[f"{c}_z"] = float((e_cls - null.mean()) / null.std(ddof=1))
        if ("decision", 2.0) in avg.columns:     # positive control: is the decision direction special?
            e_dec = row["decision_E2"]
            row[f"{c}_p_decision"] = float((1 + np.sum(np.abs(null) >= abs(e_dec))) / (1 + len(null)))

    # scale-matched null (separate run, suffix _ownsd), same class effect
    fo = p.parent / f"steer_{stem}_L{layer}_sd_ctrl-pcrandom-split_fixed{smoke}_ownsd.csv"
    if fo.exists():
        do = pd.read_csv(fo)
        ao = do.groupby(["text_id", "direction", "alpha"])["dv"].mean().unstack(["direction", "alpha"])
        for c in CTRL + ("all",):
            names = [d for d in ao.columns.get_level_values(0).unique()
                     if (d.startswith(c) if c != "all" else d.startswith(CTRL))]
            null = np.array([directional(ao, d, 2.0).mean() for d in names])
            row[f"own_{c}_n"] = len(null)
            row[f"own_{c}_null_absmax"] = float(np.abs(null).max())
            row[f"own_{c}_p"] = float((1 + np.sum(np.abs(null) >= abs(e_cls))) / (1 + len(null)))
            row[f"own_{c}_z"] = float((e_cls - null.mean()) / null.std(ddof=1))
            if "decision_E2" in row:
                row[f"own_{c}_p_decision"] = float((1 + np.sum(np.abs(null) >= abs(row["decision_E2"]))) / (1 + len(null)))

    # natural scale
    dv0 = avg[("class", 0.0)]
    yy = info["is_ai"].to_numpy()
    row["observed_dv_diff"] = float(dv0[yy == 1].mean() - dv0[yy == 0].mean())
    row["E_per_class_dist"] = float(row["class_E1"] * row["class_dist_sd"])  # linear from 1 SD
    # explicit criterion for "the observed difference is clearly different from zero":
    # grouped-bootstrap 95% CI (resampling source groups) of mean dv_AI - mean dv_human.
    # Own generator, so that the existing columns keep the random stream they had.
    lo, hi = grouped_diff_ci(dv0.to_numpy(), yy, info["source_id"].to_numpy(),
                             np.random.default_rng(2))
    row["dv_diff_lo"], row["dv_diff_hi"] = lo, hi
    row["dv_diff_clear"] = bool(lo > 0 or hi < 0)
    row["dv_diff_nonzero"] = bool(abs(row["observed_dv_diff"]) >= MIN_ABS_DV_DIFF)
    row["sign_agree"] = bool(np.sign(row["E_per_class_dist"]) == np.sign(row["observed_dv_diff"]))
    return row


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rng = np.random.default_rng(1)
    files = sorted(f for f in glob.glob("outputs/*/steer_*_fixed*_nulls.csv")
                   if "summary" not in f and "decomp" not in f)
    rows = []
    for f in files:
        print(f)
        rows.append(analyze(f, rng))
    out = pd.DataFrame(rows)
    out.to_csv("outputs/steer_nulls_summary.csv", index=False)
    out[["corpus", "cell", "layer", "n", "cos_class_decision",
         "cos_class_decision_scored"]].to_csv("outputs/steer_cosines.csv", index=False)
    cols = ["corpus", "cell", "layer", "class_dist_sd", "cos_class_decision",
            "class_inrange_1", "class_inrange_2",
            "class_inrange_6", "pcrandom_inrange_2", "class_E1", "class_E2", "class_E2_lo",
            "class_E2_hi", "decision_E2", "all_null_absmax", "all_p", "all_z",
            "own_all_null_absmax", "own_all_p", "own_all_z",
            "E_per_class_dist", "observed_dv_diff", "dv_diff_lo", "dv_diff_hi",
            "dv_diff_clear", "dv_diff_nonzero", "sign_agree"]
    with pd.option_context("display.width", 250, "display.float_format", "{:+.2f}".format):
        print(out[[c for c in cols if c in out]].to_string(index=False))
    for name, sel in (("grouped-bootstrap 95% CI excludes 0", out.dv_diff_clear),
                      (f"|observed difference| >= {MIN_ABS_DV_DIFF} nats", out.dv_diff_nonzero)):
        s = out[sel]
        print(f"\ncells with a clear observed AI-human dv difference ({name}): "
              f"{len(s)} of {len(out)}; of these, E_per_class_dist has the same sign as "
              f"the observed difference in {int(s.sign_agree.sum())} of {len(s)}")
    print(f"(all 25 cells: sign agrees in {int(out.sign_agree.sum())})")
    print("\ncosines (-> outputs/steer_cosines.csv): range over all cells "
          f"{out.cos_class_decision.min():+.2f} to {out.cos_class_decision.max():+.2f}")


if __name__ == "__main__":
    main()
