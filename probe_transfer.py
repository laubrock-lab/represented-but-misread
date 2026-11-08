"""Cross-corpus probe transfer: is the human/AI class axis corpus-specific?

The in-domain probes of ``probe_layers.py`` are supervised on the labels and
selected (layer, site, C) on the corpus they are evaluated on. With n = 103-305
and d ~ 2 560-3 840, such a probe can separate *any* two corpora that differ in
source, topic, register or length. Grouped CV and shuffled-label controls rule
out memorising individual texts, but not a corpus-level confound.

This script fits the readout on one corpus and evaluates it, unchanged, on a
corpus with different sources. See ``PLAN_probe_transfer.md``.

Readouts per directed pair (source -> target), condition and site
----------------------------------------------------------------
probe            supervised transfer: full probe_layers pipeline fit on all
                 source items, scored on the target
probe_recenter   same standardized weights, but the target is standardized with
                 its own per-dimension mean/SD (transductive diagnostic: does a
                 distribution shift, rather than the direction, break transfer?)
diff             unsupervised: source difference of class means, projected
diff_std         source standardized mean difference (Cohen's d vector)
oracle_diff      target's own mean difference (upper bound, uses target labels)

The layer is chosen on the *source* only (max grouped-CV d' in
``probe_layers*.csv``), then transfer is evaluated at that layer, so the
reported number is not selected on the target.

Overlapping source_ids (same items or same articles) are dropped by default;
such pairs are not independent and are skipped unless --include-overlap.
With four corpora only the chat <-> journalism pairs are independent.

Usage
-----
    python probe_transfer.py
    python probe_transfer.py --mode curve --layer-step 2
    python probe_transfer.py --pairs chat:journalism_seg --n-perm 100
    python probe_transfer.py --include-overlap --n-perm 0

Outputs (outputs/transfer/):
    transfer_selected.csv   one row per pair x condition x site x method
    transfer_curve.csv      with --mode curve
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from probe_layers import _fit_probe
from sdt_summary import grouped_bootstrap, sdt

CORPORA = ["chat", "chat_tw25", "journalism_win80", "journalism_seg"]
METHODS = ["probe", "probe_recenter", "diff", "diff_std", "oracle_diff"]


# ---------------------------------------------------------------------------
# Loading and metadata
# ---------------------------------------------------------------------------

def preset_of(condition: str) -> str:
    return condition.split("_")[0]


def condition_info(condition: str) -> dict:
    parts = condition.split("_")
    return dict(preset=parts[0], variant=parts[1], fmt=parts[2])


def load_corpus(root: Path, corpus: str, condition: str, site: str) -> dict | None:
    path = root / corpus / f"{condition}_acts.npz"
    if not path.exists():
        return None
    z = np.load(path)
    X = z[f"acts_{site}"]
    if np.isnan(X).any():
        raise SystemExit(f"NaN in activations: {path} [{site}]")
    return dict(X=X, y=z["is_ai"].astype(int),
                sid=z["source_id"].astype(str), tid=z["text_id"].astype(str))


def probe_table(root: Path, corpus: str, preset: str) -> pd.DataFrame:
    for name in (f"probe_layers_{preset}.csv", "probe_layers.csv"):
        f = root / corpus / name
        if f.exists():
            return pd.read_csv(f)
    raise SystemExit(f"no probe_layers table in {root / corpus}")


def select_layer(tbl: pd.DataFrame, condition: str, site: str) -> dict | None:
    sub = tbl[(tbl["condition"] == condition) & (tbl["site"] == site)]
    if sub.empty:
        return None
    r = sub.loc[sub["d_auc"].idxmax()]
    return dict(layer=int(r["layer"]), source_cv_d=float(r["d_auc"]),
                source_cv_lo=float(r["d_auc_lo"]), source_cv_hi=float(r["d_auc_hi"]))


def target_ceiling(tbl: pd.DataFrame, condition: str, site: str) -> float:
    sub = tbl[(tbl["condition"] == condition) & (tbl["site"] == site)]
    return float(sub["d_auc"].max()) if len(sub) else np.nan


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

def _rng_fit(pipeline, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return pipeline.fit(X, y)
        except np.linalg.LinAlgError:
            return pipeline.fit(X, y)


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[object, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model = _fit_probe(groups, X.astype(float), y, True)
    return model, float(model[-1].C_[0])


def fit_fixed(X: np.ndarray, y: np.ndarray, C: float):
    steps = [SimpleImputer(strategy="median"), StandardScaler(),
             PCA(n_components=None, svd_solver="full"),
             LogisticRegression(C=C, class_weight="balanced", max_iter=5000)]
    try:
        return _rng_fit(make_pipeline(*steps), X.astype(float), y)
    except np.linalg.LinAlgError:
        return _rng_fit(make_pipeline(steps[0], steps[1], steps[3]), X.astype(float), y)


def standardized_params(model) -> tuple[np.ndarray, float]:
    """Weights in the standardized space the probe sees, for the re-centering test."""
    steps = dict(model.steps)
    lr = steps["logisticregressioncv"] if "logisticregressioncv" in steps else steps["logisticregression"]
    w = steps["pca"].components_.T @ lr.coef_[0] if "pca" in steps else lr.coef_[0]
    return w, float(lr.intercept_[0])


def mean_diff(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def std_diff(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = mean_diff(X, y)
    s = np.sqrt((X[y == 1].var(0) + X[y == 0].var(0)) / 2)
    out = np.zeros_like(d)
    np.divide(d, s, out=out, where=s > 0)
    return out


def shuffle_within_groups(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    y2 = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        y2[idx] = rng.permutation(y[idx])
    return y2


def permute_labels(y: np.ndarray, groups: np.ndarray, rng, mode: str) -> np.ndarray:
    """Null for transfer. 'global' shuffles all labels (standard permutation test,
    destroys class balance and the source grouping). 'group' shuffles within
    source groups, which keeps the balance intact and is the weaker leakage
    control used in probe_layers.py."""
    return rng.permutation(y) if mode == "global" else shuffle_within_groups(y, groups, rng)


def metrics(y: np.ndarray, dv: np.ndarray, sid: np.ndarray, n_boot: int, seed: int) -> dict:
    s = sdt(y, dv)
    df = pd.DataFrame(dict(source_id=sid, is_ai=y, dv=dv))
    ci = grouped_bootstrap(df, n_boot, np.random.default_rng(seed))
    return dict(auc=float(s["auc"]), d_auc=float(s["d_auc"]),
                d_auc_lo=float(ci["d_auc_lo"]), d_auc_hi=float(ci["d_auc_hi"]))


# ---------------------------------------------------------------------------
# One directed pair x condition x site
# ---------------------------------------------------------------------------

def analyze_pair(args, src: str, tgt: str, condition: str, site: str) -> list[dict]:
    root = Path(args.out_dir)
    info = condition_info(condition)
    s = load_corpus(root, src, condition, site)
    t = load_corpus(root, tgt, condition, site)
    if s is None or t is None:
        return []

    # independence: drop target items whose source also occurs in the source corpus
    overlap = np.isin(t["sid"], np.unique(s["sid"]))
    n_dropped = int(overlap.sum())
    if n_dropped and not args.include_overlap:
        keep = ~overlap
        t = dict(X=t["X"][keep], y=t["y"][keep], sid=t["sid"][keep], tid=t["tid"][keep])
    independent = not bool(overlap.any())
    if len(np.unique(t["y"])) < 2 or len(np.unique(t["sid"])) < 3:
        print(f"skip {src}->{tgt} {condition} {site}: no independent target items")
        return []

    stbl = probe_table(root, src, info["preset"])
    ttbl = probe_table(root, tgt, info["preset"])
    sel = select_layer(stbl, condition, site)
    if sel is None:
        return []
    layer = sel["layer"]
    ceiling = target_ceiling(ttbl, condition, site)
    trow = ttbl[(ttbl["condition"] == condition) & (ttbl["site"] == site)]
    target_out = float(trow["out_d"].iloc[0]) if len(trow) and "out_d" in trow else np.nan

    Xs, ys, gs = s["X"][:, layer, :], s["y"], s["sid"]
    Xt, yt, gt = t["X"][:, layer, :], t["y"], t["sid"]

    d_src, d_std = mean_diff(Xs, ys), std_diff(Xs, ys)
    d_tgt = mean_diff(Xt, yt)
    cos_dir = float(d_src @ d_tgt / (np.linalg.norm(d_src) * np.linalg.norm(d_tgt) + 1e-12))

    model, C = fit_probe(Xs, ys, gs)
    w_std, b = standardized_params(model)

    base = dict(source=src, target=tgt, condition=condition, **info, site=site,
                layer=layer, independent=independent, n=int(len(yt)),
                n_dropped=n_dropped, source_cv_d=sel["source_cv_d"],
                source_cv_lo=sel["source_cv_lo"], source_cv_hi=sel["source_cv_hi"],
                target_cv_d=ceiling, target_out_d=target_out,
                cos_dir=cos_dir, C=C, n_perm=args.n_perm,
                p_perm=np.nan, null_auc_mean=np.nan, null_auc_hi=np.nan)

    rows = []

    def add(method: str, dv: np.ndarray, **extra):
        rows.append({**base, "method": method,
                     **metrics(yt, dv, gt, args.n_boot, args.seed), **extra})

    add("probe", model.decision_function(Xt.astype(float)))
    add("probe_recenter", ((Xt - Xt.mean(0)) / np.where(Xt.std(0) > 0, Xt.std(0), 1.0)) @ w_std + b)
    add("diff", Xt @ d_src)
    add("diff_std", Xt @ d_std)
    add("oracle_diff", Xt @ d_tgt)

    # permutation null: shuffle source labels (global by default), refit, transfer
    if args.n_perm:
        rng = np.random.default_rng(args.seed + layer)
        null = []
        for _ in range(args.n_perm):
            null.append(sdt(yt, fit_fixed(Xs, permute_labels(ys, gs, rng, args.perm_mode),
                                          C).decision_function(Xt.astype(float)))["auc"])
        obs = [r for r in rows if r["method"] == "probe"][0]["auc"]
        p = (1 + int(np.sum(np.asarray(null) >= obs))) / (1 + len(null))
        for r in rows:
            if r["method"] == "probe":
                r.update(p_perm=p, null_auc_mean=float(np.mean(null)),
                         null_auc_hi=float(np.quantile(null, 0.95)))
    return rows


def analyze_pair_curve(args, src: str, tgt: str, condition: str, site: str) -> list[dict]:
    """Direction readouts at every layer, probe every --layer-step layers."""
    root = Path(args.out_dir)
    info = condition_info(condition)
    s = load_corpus(root, src, condition, site)
    t = load_corpus(root, tgt, condition, site)
    if s is None or t is None:
        return []
    overlap = np.isin(t["sid"], np.unique(s["sid"]))
    n_dropped = int(overlap.sum())
    if n_dropped and not args.include_overlap:
        keep = ~overlap
        t = dict(X=t["X"][keep], y=t["y"][keep], sid=t["sid"][keep], tid=t["tid"][keep])
    independent = not bool(overlap.any())
    if len(np.unique(t["y"])) < 2 or len(np.unique(t["sid"])) < 3:
        return []

    n_layers = s["X"].shape[1]
    layers = range(1, n_layers, args.layer_step)
    rows = []
    for layer in layers:
        Xs, ys, gs = s["X"][:, layer, :], s["y"], s["sid"]
        Xt, yt, gt = t["X"][:, layer, :], t["y"], t["sid"]
        d_src, d_std = mean_diff(Xs, ys), std_diff(Xs, ys)
        base = dict(source=src, target=tgt, condition=condition, **info, site=site,
                    layer=layer, independent=independent, n=int(len(yt)),
                    n_dropped=n_dropped, layer_frac=layer / (n_layers - 1))
        for method, dv in (("diff", Xt @ d_src), ("diff_std", Xt @ d_std),
                           ("oracle_diff", Xt @ mean_diff(Xt, yt))):
            rows.append({**base, "method": method,
                         **metrics(yt, dv, gt, args.n_boot, args.seed)})
        try:
            model, _ = fit_probe(Xs, ys, gs)
            rows.append({**base, "method": "probe",
                         **metrics(yt, model.decision_function(Xt.astype(float)),
                                   gt, args.n_boot, args.seed)})
        except Exception as e:  # ill-conditioned fold; keep the direction readouts
            print(f"probe failed at {src}->{tgt} {condition} {site} L{layer}: {e}")
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def bh_fdr(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index)
    pv = p[ok].to_numpy()
    n = len(pv)
    if not n:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[ok] = np.clip(ranked, 0, 1)[np.argsort(order)]
    return q


def summarize(df: pd.DataFrame, condition: str, site: str) -> None:
    sub = df[(df["condition"] == condition) & (df["site"] == site)]
    for method in ("probe", "diff"):
        p = sub[sub["method"] == method].pivot_table(
            index="source", columns="target", values="d_auc", aggfunc="first")
        print(f"\n{method} d' | {condition} | {site}")
        print(p.reindex(index=CORPORA, columns=CORPORA).round(2).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--pairs", nargs="*", help="restrict to 'src:tgt' pairs")
    ap.add_argument("--mode", choices=["selected", "curve"], default="selected")
    ap.add_argument("--sites", nargs="+", default=["last", "textmean"],
                    choices=["last", "textmean"])
    ap.add_argument("--layer-step", type=int, default=2, help="--mode curve only")
    ap.add_argument("--n-perm", type=int, default=20, help="permutation null (0 = off)")
    ap.add_argument("--perm-mode", choices=["global", "group"], default="global",
                    help="global = shuffle all labels (standard); group = within "
                         "source groups (weaker; keeps class balance)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--include-overlap", action="store_true",
                    help="also run non-independent pairs (same items/articles)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    root = args.out_dir
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    else:
        pairs = [(a, b) for a in args.corpora for b in args.corpora if a != b]

    conditions = sorted(
        p.name.removesuffix("_acts.npz")
        for p in (root / args.corpora[0]).glob("*_acts.npz")
    )
    if not conditions:
        raise SystemExit(f"no *_acts.npz in {root / args.corpora[0]}")

    jobs = []
    for src, tgt in pairs:
        for cond in conditions:
            for site in args.sites:
                jobs.append((src, tgt, cond, site))
    print(f"{len(jobs)} jobs | mode {args.mode} | {args.n_jobs} workers")

    fn = analyze_pair if args.mode == "selected" else analyze_pair_curve
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fn)(args, *job) for job in jobs)

    rows = [r for chunk in results if chunk for r in chunk]
    if not rows:
        raise SystemExit("no transfer rows (all pairs skipped?)")
    df = pd.DataFrame(rows)
    for col in ("p_perm", "q_bh", "q_bh_pair", "null_auc_mean", "null_auc_hi"):
        if col not in df:
            df[col] = np.nan
    if args.mode == "selected" and df["p_perm"].notna().any():
        mask = (df["method"] == "probe") & df["p_perm"].notna()
        df.loc[mask, "q_bh"] = bh_fdr(df.loc[mask, "p_perm"])
        for _, idx in df[mask].groupby(["source", "target"]).groups.items():
            df.loc[idx, "q_bh_pair"] = bh_fdr(df.loc[idx, "p_perm"])

    out = root / "transfer"
    out.mkdir(exist_ok=True)
    name = "transfer_selected.csv" if args.mode == "selected" else "transfer_curve.csv"
    df.to_csv(out / name, index=False)
    print(f"\n-> {out / name}  ({len(df)} rows)")

    if args.mode == "selected":
        cols = ["source", "target", "preset", "variant", "fmt", "site", "method",
                "layer", "d_auc", "d_auc_lo", "d_auc_hi", "p_perm", "q_bh",
                "q_bh_pair", "source_cv_d", "target_cv_d", "target_out_d",
                "cos_dir", "n", "independent"]
        top = df[df["method"].isin(["probe", "diff"])].sort_values(
            ["method", "d_auc"], ascending=[True, False])
        with pd.option_context("display.width", 220, "display.max_rows", 60,
                               "display.float_format", "{:.2f}".format):
            print(top[cols].to_string(index=False))
        for cond in ("gemma3-4b_instruct_raw_mk", "gemma3-4b_instruct_chat_mk_neutral",
                     "gemma3-12b_instruct_raw_mk"):
            if cond in set(df["condition"]):
                summarize(df, cond, "last")


if __name__ == "__main__":
    main()
