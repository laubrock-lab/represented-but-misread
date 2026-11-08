"""Step 2: layer-wise linear probes — does the model represent what it doesn't say?

For every condition with saved activations (``*_acts.npz`` from
judge_logits.py), every site (``last`` = last prompt token, ``textmean`` =
mean over the text's own tokens) and every layer, a logistic-regression probe
predicts AI vs. human from the residual stream. Evaluation is out-of-fold with
grouped K-fold CV by source_id (all texts derived from one human original are
held out together), so the probe can't succeed by recognizing the topic.

Probe pipeline per training fold:
    StandardScaler -> PCA (all components) -> LogisticRegressionCV (L2)
With n < d, the L2 solution lies in the span of the training data, so
projecting onto all principal components is lossless; it only makes the fits fast.
C is selected by an inner grouped CV on the training fold.

Reported per (condition, site, layer):
    auc, d_auc [95% CI, grouped bootstrap]    representational sensitivity
    auc_naive, auc_optimized                  human vs. each AI prompt type
    auc_shuffled                              control: labels permuted *within*
                                              source groups (expected ~0.5)
and, for comparison, the output-level d' of the same condition (order-averaged
if both orders exist), a length-only baseline (log text length; AI and human
texts differ strongly in length, so a probe can succeed by reading off length),
and, optionally, a stylometric baseline (lab course detection features, same CV).

With --residualize-length, every activation dimension (and every stylometric
feature) is regressed on a cubic-spline basis of log text length within each
training fold, and the probe sees only the residuals (the test fold is
corrected with the training-fold coefficients). This removes smooth, possibly
non-linear length information. Diagnostics: rho_len_human / rho_len_ai are
the within-class Spearman correlations of the probe score with length (should
be ~0), and the length-only baseline should drop to ~0.5 AUC. Output files get
the suffix _lenres.

Outputs (in the condition directory):
    probe_layers.csv               one row per condition x site x layer
    probe_oof.csv                  out-of-fold probe logits per text and layer
                                   (item level, e.g. for comparison with human
                                   detection rates per text)
    probe_directions_<cond>.npz    per site and layer: difference-of-means and
                                   probe weight vectors in activation space,
                                   for steering (step 3)

Usage
-----
    python probe_layers.py outputs/chat
    python probe_layers.py outputs/chat --residualize-length
    python probe_layers.py outputs/chat --conditions gemma3-4b_base_raw_mk gemma3-4b_instruct_raw_mk
    python probe_layers.py outputs/chat --baseline-csv \\
        /path/to/labcourse/experiment_v1/python/output/chat/detection_scores.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from scipy.stats import spearmanr

from sdt_summary import grouped_bootstrap, sdt

CS = np.logspace(-4, 2, 10)


# ---------------------------------------------------------------------------
# Probe fitting
# ---------------------------------------------------------------------------

def _make_probe(groups_tr: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray, pca: bool):
    n_inner = min(5, len(np.unique(groups_tr)))
    inner = list(GroupKFold(n_splits=n_inner).split(X_tr, y_tr, groups_tr))
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if pca:
        steps.append(PCA(n_components=None, svd_solver="full"))
    # Note: scaler/PCA are fit on the whole training fold before the inner CV
    # selects C. This touches only hyperparameter selection; the outer
    # out-of-fold estimate stays clean.
    steps.append(LogisticRegressionCV(Cs=CS, cv=inner, scoring="roc_auc",
                                      class_weight="balanced", max_iter=5000))
    return make_pipeline(*steps)


def residualize(X_tr: np.ndarray, X_te: np.ndarray, c_tr: np.ndarray, c_te: np.ndarray,
                cells_tr: np.ndarray | None = None):
    """Remove a smooth function of the covariate (cubic spline of log length),
    estimated on the training fold only.

    cells_tr=None ("total"): regress on the spline alone and remove the fit. With
        strongly length-imbalanced classes this overcorrects and leaves an inverted
        length trend within each class.
    cells_tr given ("ancova"): regress on spline + cell dummies (prompt type) and
        remove only the spline term, i.e. the pooled *within-cell* length function.
        The classes are thereby compared at equal length, as in an ANCOVA. Test
        items need no labels: only the length term is subtracted.
    """
    X_tr = np.where(np.isnan(X_tr), np.nanmedian(X_tr, axis=0), X_tr)
    X_te = np.where(np.isnan(X_te), np.nanmedian(X_tr, axis=0), X_te)
    if cells_tr is None:
        basis = SplineTransformer(n_knots=4, degree=3, extrapolation="linear").fit(c_tr)
        Z_tr, Z_te = basis.transform(c_tr), basis.transform(c_te)   # partition of unity: includes intercept
        coef, *_ = np.linalg.lstsq(Z_tr, X_tr, rcond=None)
        R_tr, R_te = X_tr - Z_tr @ coef, X_te - Z_te @ coef
    else:
        basis = SplineTransformer(n_knots=4, degree=3, extrapolation="linear",
                                  include_bias=False).fit(c_tr)
        B_tr, B_te = basis.transform(c_tr), basis.transform(c_te)
        levels = np.unique(cells_tr)
        if len(c_tr) < 4 * (len(levels) + B_tr.shape[1]):
            # too few training items for a spline next to the cell intercepts:
            # fall back to a linear length term (keeps the fit identified)
            B_tr, B_te = c_tr - c_tr.mean(), c_te - c_tr.mean()
        D_tr = (cells_tr[:, None] == levels[None, :]).astype(float)  # one dummy per cell = intercepts
        coef, *_ = np.linalg.lstsq(np.hstack([D_tr, B_tr]), X_tr, rcond=None)
        g = coef[len(levels):]                                       # length term only
        R_tr, R_te = X_tr - B_tr @ g, X_te - B_te @ g
    # Dimensions that are constant or fully explained by the covariate (and cells) leave only
    # rounding noise, which StandardScaler would blow up to unit variance (and
    # which is itself a function of the covariate). Zero them out.
    dead = R_tr.std(0) <= np.maximum(1e-6 * X_tr.std(0), 1e-8)
    R_tr[:, dead], R_te[:, dead] = 0.0, 0.0
    return R_tr, R_te


def _fit_probe(groups_tr, X_tr, y_tr, pca: bool):
    """Fit; if LAPACK's SVD fails to converge in the PCA step (ill-conditioned
    residualized data), refit without PCA — the same L2 model, only slower."""
    try:
        return _make_probe(groups_tr, X_tr, y_tr, pca).fit(X_tr, y_tr)
    except np.linalg.LinAlgError:
        if not pca:
            raise
        return _make_probe(groups_tr, X_tr, y_tr, pca=False).fit(X_tr, y_tr)


def cv_logits(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int,
              pca: bool = True, covar: np.ndarray | None = None,
              cells: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Out-of-fold decision values (logits) with grouped K-fold CV.

    covar: optional [n, 1] covariate (log length) removed within each fold;
    cells: optional cell labels (prompt type) -> ANCOVA-style removal.
    """
    oof = np.full(len(y), np.nan)
    chosen = []
    outer = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        for tr, te in outer.split(X, y, groups):
            X_tr, X_te = X[tr], X[te]
            if covar is not None:
                X_tr, X_te = residualize(X_tr, X_te, covar[tr], covar[te],
                                         None if cells is None else cells[tr])
            probe = _fit_probe(groups[tr], X_tr, y[tr], pca)
            oof[te] = probe.decision_function(X_te)
            chosen.append(float(probe[-1].C_[0]))
    return oof, float(np.median(chosen))


def shuffle_within_groups(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    y2 = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        y2[idx] = rng.permutation(y[idx])
    return y2


def directions(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               covar: np.ndarray | None = None,
               cells: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Difference of class means, and the probe weight mapped back to activation space."""
    if covar is not None:
        X, _ = residualize(X, X[:1], covar, covar[:1], cells)
    diff = X[y == 1].mean(0) - X[y == 0].mean(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        probe = _fit_probe(groups, X, y, pca=True)
    steps = dict(probe.steps)
    scaler, lr = steps["standardscaler"], steps["logisticregressioncv"]
    w_std = (steps["pca"].components_.T @ lr.coef_[0] if "pca" in steps
             else lr.coef_[0])                     # in standardized space
    w = w_std / scaler.scale_                      # in raw activation space
    return diff.astype(np.float32), (w / np.linalg.norm(w)).astype(np.float32)


def run_layer(X: np.ndarray, y: np.ndarray, groups: np.ndarray, ptype: np.ndarray,
              loglen: np.ndarray, lc: str, n_splits: int, n_perm: int, seed: int,
              want_dirs: bool) -> dict:
    covar = loglen if lc != "none" else None
    cells = ptype if lc == "ancova" else None
    oof, C = cv_logits(X, y, groups, n_splits, covar=covar, cells=cells)
    rng = np.random.default_rng(seed)
    shuffled = [sdt(ys, cv_logits(X, ys, groups, n_splits, covar=covar, cells=cells)[0])["auc"]
                for ys in (shuffle_within_groups(y, groups, rng) for _ in range(n_perm))]
    out = dict(oof=oof, C=C, auc_shuffled=float(np.mean(shuffled)) if shuffled else np.nan)
    for cls, name in ((0, "rho_len_human"), (1, "rho_len_ai")):
        m = y == cls
        out[name] = float(spearmanr(oof[m], loglen[m, 0])[0])
    for pt in ("naive", "optimized"):
        m = np.isin(ptype, ["human", pt])
        out[f"auc_{pt}"] = sdt(y[m], oof[m])["auc"] if m.any() else np.nan
    if want_dirs:
        out["diff"], out["w"] = directions(X, y, groups, covar, cells)
    return out


# ---------------------------------------------------------------------------
# Reference values: output level and stylometric baseline
# ---------------------------------------------------------------------------

def output_level(out_dir: Path, stem: str) -> dict:
    """d' of the model's own answer for this condition (order-averaged if possible)."""
    parts = stem.split("_")
    other = "km" if parts[3] == "mk" else "mk"
    stems = [stem, "_".join(parts[:3] + [other] + parts[4:])]
    dfs = [pd.read_csv(out_dir / f"{s}.csv") for s in stems if (out_dir / f"{s}.csv").exists()]
    if not dfs:
        return {}
    df = pd.concat(dfs).groupby(["text_id", "source_id", "is_ai"], as_index=False)["dv"].mean()
    r = sdt(df["is_ai"].to_numpy(), df["dv"].to_numpy())
    ci = grouped_bootstrap(df, 1000, np.random.default_rng(0))
    return dict(out_d=r["d_auc"], out_d_lo=ci["d_auc_lo"], out_d_hi=ci["d_auc_hi"],
                out_c=r["c0"], out_orders=len(dfs))


def length_baseline(meta: pd.DataFrame, n_splits: int, lc: str,
                    groups: np.ndarray | None = None) -> dict:
    """Control: how well does log text length alone separate the classes?
    (With length control this is a sanity check and should be ~0.)"""
    X = np.log(meta[["n_text_tokens"]].to_numpy(float))
    if groups is None:
        groups = meta["source_id"].to_numpy()
    oof, _ = cv_logits(X, meta["is_ai"].to_numpy(), groups,
                       n_splits, pca=False, covar=X if lc != "none" else None,
                       cells=meta["prompt_type"].to_numpy() if lc == "ancova" else None)
    tmp = meta[["source_id", "is_ai"]].assign(dv=oof)
    ci = grouped_bootstrap(tmp, 1000, np.random.default_rng(0))
    return dict(len_d=sdt(tmp["is_ai"].to_numpy(), oof)["d_auc"],
                len_d_lo=ci["d_auc_lo"], len_d_hi=ci["d_auc_hi"])


def stylometric_baseline(path: Path, meta: pd.DataFrame, n_splits: int, lc: str) -> dict:
    feats = pd.read_csv(path)
    feats["model_name"] = feats["model_name"].fillna("").astype(str)
    drop = {"source_id", "generator", "model_name", "is_ai"}
    cols = [c for c in feats.columns if c not in drop and not c.startswith("ai_prob")
            and pd.api.types.is_numeric_dtype(feats[c])]
    ai = meta[meta["is_ai"] == 1].merge(
        feats, left_on=["source_id", "generator", "gen_model"],
        right_on=["source_id", "generator", "model_name"], how="left", suffixes=("", "_f"))
    hu = meta[meta["is_ai"] == 0].merge(
        feats.drop_duplicates(["source_id", "generator"]),
        on=["source_id", "generator"], how="left", suffixes=("", "_f"))
    df = pd.concat([ai, hu], ignore_index=True)
    missing = df[cols].isna().all(axis=1).sum()
    df = df[~df[cols].isna().all(axis=1)]
    covar = np.log(df[["n_text_tokens"]].to_numpy(float)) if lc != "none" else None
    cells = df["prompt_type"].to_numpy() if lc == "ancova" else None
    oof, _ = cv_logits(df[cols].to_numpy(float), df["is_ai"].to_numpy(),
                       df["source_id"].to_numpy(), n_splits, pca=False, covar=covar,
                       cells=cells)
    tmp = df[["source_id", "is_ai"]].assign(dv=oof)
    ci = grouped_bootstrap(tmp, 1000, np.random.default_rng(0))
    return dict(base_d=sdt(tmp["is_ai"].to_numpy(), oof)["d_auc"],
                base_d_lo=ci["d_auc_lo"], base_d_hi=ci["d_auc_hi"],
                base_n=len(df), base_unmatched=int(missing), base_n_features=len(cols))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="directory with judge_logits.py outputs")
    ap.add_argument("--conditions", nargs="*",
                    help="condition stems (default: all with *_acts.npz)")
    ap.add_argument("--sites", nargs="+", default=["last", "textmean"],
                    choices=["last", "textmean"])
    ap.add_argument("--layer-step", type=int, default=1, help="probe every k-th layer")
    ap.add_argument("--n-splits", type=int, default=10, help="outer grouped CV folds")
    ap.add_argument("--group-by", choices=["source", "author"], default="source",
                    help="grouping for the outer CV and the shuffled-label control. "
                         "'author' holds out whole human authors (chat only; author = "
                         "the source_id prefix, ~10 people). Files get suffix _authgrp.")
    ap.add_argument("--n-perm", type=int, default=2, help="shuffled-label runs per layer")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--baseline-csv", type=Path,
                    help="lab course detection_scores.csv for the stylometric baseline")
    ap.add_argument("--length-control", choices=["none", "total", "ancova"], default="none",
                    help="remove a spline of log text length within each fold: 'total' = "
                         "regression on length alone; 'ancova' = pooled within-cell length "
                         "function (prompt-type intercepts), i.e. comparison at equal length")
    ap.add_argument("--residualize-length", action="store_true",
                    help="alias for --length-control total")
    ap.add_argument("--suffix", default="",
                    help="appended to output file names, e.g. _gemma3-12b (avoids overwriting)")
    ap.add_argument("--no-directions", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    stems = args.conditions or sorted(p.name.removesuffix("_acts.npz")
                                      for p in args.out_dir.glob("*_acts.npz"))
    if not stems:
        raise SystemExit(f"No *_acts.npz in {args.out_dir} — run judge_logits.py first.")

    rows, oof_rows, baseline = [], [], None
    _lc = "total" if args.residualize_length else args.length_control
    sfx = {"none": "", "total": "_lenres", "ancova": "_lenanc"}[_lc] + args.suffix
    if args.group_by == "author":
        sfx = "_authgrp" + sfx
    for stem in stems:
        z = np.load(args.out_dir / f"{stem}_acts.npz")
        meta = pd.DataFrame(dict(text_id=z["text_id"], source_id=z["source_id"],
                                 is_ai=z["is_ai"]))
        judg = pd.read_csv(args.out_dir / f"{stem}.csv", keep_default_na=False)
        meta = meta.merge(judg[["text_id", "generator", "gen_model", "prompt_type", "n_text_tokens"]],
                          on="text_id", how="left", validate="1:1")
        y, groups = meta["is_ai"].to_numpy(), meta["source_id"].to_numpy()
        if args.group_by == "author":
            groups = np.array([g.split("__")[0] for g in groups])
            if len(np.unique(groups)) < 3:
                raise SystemExit(f"author grouping needs >= 3 authors, got "
                                 f"{len(np.unique(groups))} in {stem}")
        loglen = np.log(meta[["n_text_tokens"]].to_numpy(float))
        lc = "total" if args.residualize_length else args.length_control
        ptype = meta["prompt_type"].to_numpy()

        ref = output_level(args.out_dir, stem) | length_baseline(meta, args.n_splits, lc, groups)
        print(f"length-only baseline: d' = {ref['len_d']:.2f} [{ref['len_d_lo']:.2f}, {ref['len_d_hi']:.2f}]")
        if args.baseline_csv and baseline is None:
            baseline = stylometric_baseline(args.baseline_csv, meta, args.n_splits, lc)
            print(f"stylometric baseline: d' = {baseline['base_d']:.2f} "
                  f"[{baseline['base_d_lo']:.2f}, {baseline['base_d_hi']:.2f}], "
                  f"{baseline['base_n_features']} features, {baseline['base_unmatched']} texts unmatched")

        dirs = {}
        for site in args.sites:
            acts = z[f"acts_{site}"]                    # [n, L+1, d]
            layers = list(range(0, acts.shape[1], args.layer_step))
            print(f"\n=== {stem} | {site} | {len(layers)} layers x {args.n_splits}-fold "
                  f"grouped CV by {args.group_by} (+{args.n_perm} shuffles) | "
                  f"n = {len(y)}, d = {acts.shape[2]}")
            res = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(run_layer)(acts[:, l, :], y, groups, ptype, loglen, lc, args.n_splits,
                                   args.n_perm, args.seed + l, not args.no_directions)
                for l in layers)
            rng = np.random.default_rng(args.seed)
            for l, r in zip(layers, res):
                s = sdt(y, r["oof"])
                ci = grouped_bootstrap(meta.assign(dv=r["oof"]), args.n_boot, rng)
                rows.append(dict(condition=stem, site=site, layer=l, n_layers=acts.shape[1] - 1,
                                 auc=s["auc"], d_auc=s["d_auc"], d_auc_lo=ci["d_auc_lo"],
                                 d_auc_hi=ci["d_auc_hi"], auc_naive=r["auc_naive"],
                                 auc_optimized=r["auc_optimized"], auc_shuffled=r["auc_shuffled"],
                                 rho_len_human=r["rho_len_human"], rho_len_ai=r["rho_len_ai"],
                                 length_control=lc,
                                 C=r["C"], **ref, **(baseline or {})))
                oof_rows.append(meta[["text_id", "source_id", "is_ai", "prompt_type"]]
                                .assign(condition=stem, site=site, layer=l, probe_logit=r["oof"]))
                if "diff" in r:
                    dirs[f"{site}_diff"] = dirs.get(f"{site}_diff", []) + [r["diff"]]
                    dirs[f"{site}_w"] = dirs.get(f"{site}_w", []) + [r["w"]]
            if dirs:
                dirs[f"{site}_layers"] = [np.array(layers)]

            best = max((x for x in rows if x["condition"] == stem and x["site"] == site),
                       key=lambda x: x["d_auc"])
            msg = (f"    best layer {best['layer']}/{best['n_layers']}: probe d' = {best['d_auc']:.2f} "
                   f"[{best['d_auc_lo']:.2f}, {best['d_auc_hi']:.2f}], shuffled AUC "
                   f"{best['auc_shuffled']:.2f}; layer 0: d' = {rows[-len(layers)]['d_auc']:.2f}")
            if ref:
                msg += (f"\n    output d' = {ref['out_d']:.2f} [{ref['out_d_lo']:.2f}, "
                        f"{ref['out_d_hi']:.2f}], c = {ref['out_c']:.2f} ({ref['out_orders']} order(s))")
            print(msg)

        if dirs:
            np.savez(args.out_dir / f"probe_directions_{stem}{sfx}.npz",
                     **{k: np.stack(v) if not k.endswith("_layers") else v[0]
                        for k, v in dirs.items()})

    pd.DataFrame(rows).to_csv(args.out_dir / f"probe_layers{sfx}.csv", index=False)
    pd.concat(oof_rows).to_csv(args.out_dir / f"probe_oof{sfx}.csv", index=False)
    print(f"\n-> {args.out_dir / f'probe_layers{sfx}.csv'}, probe_oof{sfx}.csv, probe_directions_*{sfx}.npz")


if __name__ == "__main__":
    main()
