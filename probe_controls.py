"""Within-corpus controls for the cross-corpus transfer result.

The transfer matrix (``probe_transfer.py``) shows that a probe fitted on chat
still ranks human vs. AI in journalism. Three objections remain, and each is
tested here:

1. **Is the chat probe an author detector?**  ``author_cv`` repeats the
   in-domain CV with leave-one-author-out groups (author = the ``source_id``
   prefix, ~10 people). The original CV already holds out all texts derived
   from one human original (grouped by ``source_id``), but the same person
   contributes several chats, so an author-level confound is still possible.

2. **Is the axis one generator's fingerprint?**  ``generator_xfer`` fits on
   human + one generator's rewrites and tests on human + the other
   generator's rewrites *of held-out authors*, against the same-generator
   baseline. Chat AI texts come from Claude Opus 4.6 and GPT-5.4, fully
   crossed with prompt type.

3. **Do the two AI versions (naive vs. style-matched) behave differently?**
   ``stratum_cv`` and ``stratum_xfer`` repeat the analysis per prompt type
   and cross the types (naive -> optimized and back), with authors held out,
   plus ``output``: the model's own answer d' per stratum.

The journalism corpus has a **single human author**, so author grouping is
impossible there; the script groups by source (article) instead and says so.
Generator transfer is skipped when only one generator exists.

Layer selection: the source-selected layer from ``probe_layers*.csv`` (max
grouped-CV d' for that condition and site), exactly as in
``probe_transfer.py``. It is not re-selected per stratum, so no stratum gets
its own best layer. Every cross-condition comparison uses the same folds.

Usage
-----
    python probe_controls.py                                   # chat, both sites
    python probe_controls.py --corpus journalism_seg           # stratum + output
    python probe_controls.py --conditions gemma3-4b_base_raw_mk --sites last
    python probe_controls.py --n-jobs 8 --n-boot 1000

Output (``outputs/controls/``): one row per corpus x condition x site x
analysis cell, with pooled AUC/d' [grouped-bootstrap CI] and the mean and sign
count of the per-fold AUCs (the pooling check: folds differ in scale).
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold

import config
from probe_layers import _fit_probe, cv_logits
from probe_transfer import probe_table, select_layer
from sdt_summary import grouped_bootstrap, sdt

CORPORA = ["chat", "chat_tw25", "journalism_seg", "journalism_win80"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def authors_of(corpus: str, sid: np.ndarray) -> np.ndarray:
    """Chat: the source_id prefix (one human author, ~2-5 chats each).
    Journalism: the article itself — the corpus has a single human author."""
    if corpus.startswith("chat"):
        return np.array([s.split("__")[0] for s in sid])
    return sid


def load_items(out_dir: Path, corpus: str, condition: str, site: str,
               preset: str) -> tuple[np.ndarray, pd.DataFrame, dict] | None:
    z = np.load(out_dir / corpus / f"{condition}_acts.npz")
    tbl = probe_table(out_dir, corpus, preset)
    sel = select_layer(tbl, condition, site)
    if sel is None:
        return None
    judg = pd.read_csv(out_dir / corpus / f"{condition}.csv", keep_default_na=False)
    judg["text_id"] = judg["text_id"].astype(str)
    meta = pd.DataFrame(dict(text_id=z["text_id"].astype(str),
                             source_id=z["source_id"].astype(str),
                             is_ai=z["is_ai"].astype(int)))
    meta = meta.merge(judg[["text_id", "prompt_type", "gen_model"]],
                      on="text_id", how="left", validate="1:1")
    if meta["prompt_type"].isna().any():
        raise SystemExit(f"missing prompt_type for {corpus}/{condition}")
    meta["group"] = authors_of(corpus, meta["source_id"].to_numpy())
    meta["gen_model"] = meta["gen_model"].fillna("").astype(str)
    X = z[f"acts_{site}"][:, sel["layer"], :]
    if np.isnan(X).any():
        raise SystemExit(f"NaN in activations: {corpus}/{condition} [{site}]")
    return X, meta, sel


def load_dv(out_dir: Path, stem: str) -> pd.DataFrame:
    """Order-averaged output-level judgment per text (as in probe_layers.py)."""
    parts = stem.split("_")
    other = "km" if parts[3] == "mk" else "mk"
    stems = [stem, "_".join(parts[:3] + [other] + parts[4:])]
    dfs = [pd.read_csv(out_dir / f"{s}.csv") for s in stems
           if (out_dir / f"{s}.csv").exists()]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs)
    df["text_id"] = df["text_id"].astype(str)
    return (df.groupby(["text_id", "source_id", "is_ai", "prompt_type"],
                       as_index=False)["dv"].mean())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def n_folds_for(groups: np.ndarray) -> int:
    """Leave-one-group-out for small group counts (chat: 14 authors,
    journalism: 14 articles), 10-fold beyond that."""
    ng = len(np.unique(groups))
    return ng if ng <= 15 else 10


def fold_aucs(y: np.ndarray, dv: np.ndarray, groups: np.ndarray,
              n_splits: int) -> list[float]:
    out = []
    kf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    idx = np.arange(len(y))
    for _, te in kf.split(idx, y, groups):
        if len(np.unique(y[te])) == 2:
            out.append(float(sdt(y[te], dv[te])["auc"]))
    return out


def eval_scores(oof: np.ndarray, meta: pd.DataFrame, n_splits: int,
                n_boot: int, seed: int) -> dict:
    ok = ~np.isnan(oof)
    y = meta["is_ai"].to_numpy()[ok]
    dv = oof[ok]
    g = meta["group"].to_numpy()[ok]
    s = sdt(y, dv)
    ci = grouped_bootstrap(meta.loc[ok, ["source_id", "is_ai"]].assign(dv=dv),
                           n_boot, np.random.default_rng(seed))
    fa = fold_aucs(y, dv, g, n_splits)
    return dict(auc=float(s["auc"]), d_auc=float(s["d_auc"]),
                d_auc_lo=float(ci["d_auc_lo"]), d_auc_hi=float(ci["d_auc_hi"]),
                auc_fold_mean=float(np.mean(fa)) if fa else np.nan,
                n_folds=len(fa), n_folds_pos=int(np.sum(np.asarray(fa) > 0.5)))


def cv_cell(X: np.ndarray, meta: pd.DataFrame, analysis: str, train: str,
            test: str, n_boot: int, seed: int) -> dict:
    y = meta["is_ai"].to_numpy()
    g = meta["group"].to_numpy()
    n_splits = n_folds_for(g)
    oof, C = cv_logits(X, y, g, n_splits)
    return dict(analysis=analysis, train=train, test=test,
                n_train=int(len(y)), n_test=int(len(y)), n_groups=len(np.unique(g)),
                C=C, **eval_scores(oof, meta, n_splits, n_boot, seed))


def xfer_cell(X: np.ndarray, meta: pd.DataFrame, analysis: str,
              train_mask: np.ndarray, test_mask: np.ndarray,
              train_label: str, test_label: str,
              n_boot: int, seed: int) -> dict:
    """Author-held-out cross-condition transfer: every fold trains on the other
    authors and predicts the held-out author's test-condition rows. Pooled
    out-of-fold scores give one AUC per cell."""
    y = meta["is_ai"].to_numpy()
    g = meta["group"].to_numpy()
    n_splits = n_folds_for(g)
    oof = np.full(len(y), np.nan)
    Cs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(X, y, g):
            tr = np.zeros(len(y), bool)
            te = np.zeros(len(y), bool)
            tr[tr_idx] = True
            te[te_idx] = True
            tr &= train_mask
            te &= test_mask
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            probe = _fit_probe(g[tr], X[tr], y[tr], True)
            oof[te] = probe.decision_function(X[te])
            Cs.append(float(probe[-1].C_[0]))
    return dict(analysis=analysis, train=train_label, test=test_label,
                n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
                n_groups=len(np.unique(g)),
                C=float(np.median(Cs)) if Cs else np.nan,
                **eval_scores(oof, meta, n_splits, n_boot, seed))


# ---------------------------------------------------------------------------
# One condition x site
# ---------------------------------------------------------------------------

def run_cell(out_dir: Path, corpus: str, condition: str, site: str,
             n_boot: int, seed: int) -> list[dict]:
    preset = condition.split("_")[0]
    loaded = load_items(out_dir, corpus, condition, site, preset)
    if loaded is None:
        return []
    X, meta, sel = loaded
    y = meta["is_ai"].to_numpy()
    pt = meta["prompt_type"].to_numpy()
    gm = meta["gen_model"].to_numpy()
    human = y == 0
    base = dict(corpus=corpus, condition=condition, preset=preset,
                variant=condition.split("_")[1], fmt=condition.split("_")[2],
                site=site, layer=sel["layer"], source_cv_d=sel["source_cv_d"])
    rows: list[dict] = []

    def add(**kw):
        rows.append({**base, **kw})

    # 1. author- vs. source-grouped CV on all rows
    add(**cv_cell(X, meta, "cv", "all", "all", n_boot, seed))
    meta_src = meta.assign(group=meta["source_id"].to_numpy())
    add(**cv_cell(X, meta_src, "cv_source_group", "all", "all", n_boot, seed))

    # 2. in-domain CV per prompt-type stratum
    for st in ("naive", "optimized"):
        m = human | (pt == st)
        if len(np.unique(y[m])) == 2:
            add(**cv_cell(X[m], meta[m].reset_index(drop=True),
                          "stratum_cv", f"human+{st}", f"human+{st}", n_boot, seed))

    # 3. are the two AI versions distinguishable from each other?
    ai = ~human
    if len(np.unique(pt[ai])) >= 2:
        meta_ai = meta[ai].reset_index(drop=True)
        y_ai = (meta_ai["prompt_type"] == "optimized").astype(int).to_numpy()
        g_ai = meta_ai["group"].to_numpy()
        oof, C = cv_logits(X[ai], y_ai, g_ai, n_folds_for(g_ai))
        s = sdt(y_ai, oof)
        ci = grouped_bootstrap(meta_ai[["source_id"]].assign(
            is_ai=y_ai, dv=oof), n_boot, np.random.default_rng(seed))
        add(analysis="naive_vs_optimized", train="AI naive", test="AI optimized",
            n_train=int(ai.sum()), n_test=int(ai.sum()),
            n_groups=len(np.unique(g_ai)), C=C,
            auc=float(s["auc"]), d_auc=float(s["d_auc"]),
            d_auc_lo=float(ci["d_auc_lo"]), d_auc_hi=float(ci["d_auc_hi"]),
            auc_fold_mean=np.nan, n_folds=np.nan, n_folds_pos=np.nan)

    # 4. generator transfer (chat only: Claude vs. GPT, fully crossed)
    gens = sorted(v for v in np.unique(gm[ai]) if v)
    if len(gens) >= 2:
        for g_tr, g_te in product(gens, repeat=2):
            tr = human | ((y == 1) & (gm == g_tr))
            te = human | ((y == 1) & (gm == g_te))
            add(**xfer_cell(X, meta, "generator_xfer", tr, te,
                            f"human+{g_tr}", f"human+{g_te}", n_boot, seed))
        for g_te in gens:
            te = human | ((y == 1) & (gm == g_te))
            add(**xfer_cell(X, meta, "generator_xfer", np.ones(len(y), bool), te,
                            "all", f"human+{g_te}", n_boot, seed))

    # 5. naive vs. optimized transfer (generators pooled)
    for s_tr, s_te in product(("naive", "optimized"), repeat=2):
        tr = human | (pt == s_tr)
        te = human | (pt == s_te)
        add(**xfer_cell(X, meta, "stratum_xfer", tr, te,
                        f"human+{s_tr}", f"human+{s_te}", n_boot, seed))
    for s_te in ("naive", "optimized"):
        te = human | (pt == s_te)
        add(**xfer_cell(X, meta, "stratum_xfer", np.ones(len(y), bool), te,
                        "all", f"human+{s_te}", n_boot, seed))

    # 6. the model's own answer, overall and per stratum
    dv = load_dv(out_dir / corpus, condition)
    if len(dv):
        m = dv.merge(meta[["text_id"]], on="text_id", how="inner")
        for label, mask in (("all", np.ones(len(m), bool)),
                            ("human+naive", np.isin(m["prompt_type"], ["human", "naive"])),
                            ("human+optimized",
                             np.isin(m["prompt_type"], ["human", "optimized"]))):
            d = m[mask]
            if len(np.unique(d["is_ai"])) < 2:
                continue
            s = sdt(d["is_ai"].to_numpy(), d["dv"].to_numpy())
            ci = grouped_bootstrap(d[["source_id", "is_ai"]].assign(
                dv=d["dv"].to_numpy()), n_boot, np.random.default_rng(seed))
            add(analysis="output", train="model answer", test=label,
                n_train=int(len(d)), n_test=int(len(d)), n_groups=np.nan, C=np.nan,
                auc=float(s["auc"]), d_auc=float(s["d_auc"]),
                d_auc_lo=float(ci["d_auc_lo"]), d_auc_hi=float(ci["d_auc_hi"]),
                auc_fold_mean=np.nan, n_folds=np.nan, n_folds_pos=np.nan)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    ap.add_argument("--corpus", default="chat", choices=CORPORA)
    ap.add_argument("--conditions", nargs="*", help="default: all with *_acts.npz")
    ap.add_argument("--sites", nargs="+", default=["last", "textmean"],
                    choices=["last", "textmean"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    root = args.out_dir
    stems = args.conditions or sorted(
        p.name.removesuffix("_acts.npz")
        for p in (root / args.corpus).glob("*_acts.npz"))
    if not stems:
        raise SystemExit(f"no *_acts.npz in {root / args.corpus}")

    jobs = [(args.corpus, stem, site) for stem in stems for site in args.sites]
    print(f"{len(jobs)} jobs | corpus {args.corpus} | {args.n_jobs} workers")
    done = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_cell)(root, *job, args.n_boot, args.seed) for job in jobs)

    rows = [r for chunk in done if chunk for r in chunk]
    if not rows:
        raise SystemExit("no control rows")
    df = pd.DataFrame(rows)
    out = root / "controls"
    out.mkdir(exist_ok=True)
    name = f"controls_{args.corpus}.csv"
    df.to_csv(out / name, index=False)
    print(f"\n-> {out / name}  ({len(df)} rows)")

    show = ["analysis", "train", "test", "n_test", "auc", "d_auc", "d_auc_lo",
            "d_auc_hi", "auc_fold_mean", "n_folds_pos", "n_folds"]
    with pd.option_context("display.width", 220, "display.max_rows", 200,
                           "display.float_format", "{:.2f}".format):
        print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
