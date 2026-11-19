"""Training study: the misleading-news-cue account as a prediction, not a post hoc story.

After C1 (2 000 correctly labelled authorship judgments of NEWS texts) the answers on
the short chats are BELOW chance. The manuscript explains this by saying that the news
cues the model has been trained on are misleading for chats. If that is the mechanism,
then a *news* readout that is defined without ever looking at the fine-tuned model's
answers must already point the wrong way on the chats, and the fine-tuned model's
answers must follow it.

Tests (all on corpus chat_tw25, 300 texts, 61 source groups):

P0   d' of the news-probe score on the chats (AI vs human).           predicted < 0
P1   Spearman correlation across the 300 chats between the news-probe score and the
     order-averaged dv of each fine-tuned model, pooled / within human / within AI,
     per condition and seed, the mean over seeds, and the differences C1-C3, C1-C4,
     C1-C0 of the seed means (paired bootstrap).                      predicted C1 > 0,
                                                                      C1 > C0, C3, C4
P1b  same, with the probe fitted on the TEXT-MEAN activations of each fine-tuned model
     itself instead of the base model (secondary).
P1c  same at the LAST prompt position (secondary; partly circular, because the last
     position is where the trained readout lives -- flagged in the output).
P2   cosine, at the last position, between the news class direction in the fine-tuned
     model (difference of class means; and separately the probe weight vector mapped
     back to activation space) and that model's decision direction on the chats.
     Reference: cosines of the decision direction with 1 000 random directions from the
     same PCA subspace as the steer_logits.py "pcrandom" control.   predicted C1 > 0,
                                                                     > reference, > C3

The PRIMARY predictor (P0, P1) is defined independently of the fine-tuned models: a
probe fitted on the TEXT-MEAN activations of the BASE model on a news corpus, applied
unchanged to the base model's text-mean activations of the chats. The same score vector
is therefore correlated with all 15 fine-tuned models' answers.

All probes use the probe_transfer.py / probe_layers.py pipeline (median imputation,
standardisation, full PCA, L2 logistic regression with the C selected by grouped inner
CV). CIs are grouped bootstrap over the chat source_id (the same 1 000 resamples for
every statistic, so seed means and differences are paired). Fixed layers 20 and 24 and
their mean carry CIs; the full layer curve is descriptive (point estimates only).

Usage (cluster, CPU):
    python training_study/prediction_test.py --n-jobs 8
    python training_study/prediction_test.py --tests P0 P1 --no-curve

Output: outputs/training_study/prediction_test.csv (long format; numbers and IDs only).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from probe_transfer import fit_probe  # noqa: E402
from sdt_summary import sdt  # noqa: E402
from analyze_answers import answers as order_avg_answers  # noqa: E402

SUBSETS = ("pooled", "human", "ai")
MEAN_KEY = "20_24mean"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_acts(path: Path, sites: tuple[str, ...], layers: list[int] | None = None) -> dict:
    """{site: X[n, len(layers), d]} plus labels, groups, ids. Reads the npz once."""
    z = np.load(path)
    out = dict(y=z["is_ai"].astype(int), sid=z["source_id"].astype(str),
               tid=z["text_id"].astype(str))
    for site in sites:
        X = z[f"acts_{site}"]
        out[site] = X if layers is None else X[:, layers, :].copy()
        del X
    return out


def fit_score(Xn: np.ndarray, yn: np.ndarray, gn: np.ndarray,
              Xc: np.ndarray) -> np.ndarray:
    """Fit the probe_transfer pipeline on the news activations, score the chats."""
    model, _ = fit_probe(Xn.astype(float), yn, gn)
    return model.decision_function(Xc.astype(float))


def reorder(d: dict, tid: np.ndarray, sites: tuple[str, ...]) -> dict:
    """Align rows to the canonical text_id order."""
    pos = pd.Series(np.arange(len(d["tid"])), index=d["tid"])
    ix = pos.reindex(tid).to_numpy()
    if np.isnan(ix).any():
        raise SystemExit("text_id mismatch between activation files")
    ix = ix.astype(int)
    out = dict(y=d["y"][ix], sid=d["sid"][ix], tid=d["tid"][ix])
    for site in sites:
        out[site] = d[site][ix]
    return out


def run_tags(ts_dir: Path, preset: str) -> list[tuple[str, int]]:
    """(condition, seed) of every fine-tuned run with per-text answers."""
    tags = []
    for f in sorted(ts_dir.glob(f"{preset}_ts*_raw_mk.csv")):
        m = re.search(r"_ts(C\d)s(\d+)_raw_mk", f.name)
        if m:
            tags.append((m.group(1), int(m.group(2))))
    return tags


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def boot_indices(sid: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Grouped bootstrap: resample source groups with replacement, once for all tests."""
    by = {g: np.flatnonzero(sid == g) for g in np.unique(sid)}
    groups = np.array(list(by), dtype=object)
    rng = np.random.default_rng(seed)
    return [np.concatenate([by[g] for g in rng.choice(groups, size=len(groups), replace=True)])
            for _ in range(n_boot)]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float(ra @ rb / den) if den > 0 else np.nan


def ci(boot: np.ndarray | None) -> tuple[float, float]:
    if boot is None or not len(boot) or np.all(np.isnan(boot)):
        return np.nan, np.nan
    return float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))


def corr_stats(score: np.ndarray, dv: np.ndarray, y: np.ndarray,
               idxs: list[np.ndarray] | None) -> dict:
    """{subset: (estimate, boot array or None)} for pooled / human / AI."""
    masks = dict(pooled=np.ones(len(y), bool), human=y == 0, ai=y == 1)
    out = {}
    for name, m in masks.items():
        est = spearman(score[m], dv[m])
        if idxs is None:
            out[name] = (est, None, int(m.sum()))
            continue
        b = np.empty(len(idxs))
        for k, ix in enumerate(idxs):
            j = ix[m[ix]]
            b[k] = spearman(score[j], dv[j])
        out[name] = (est, b, int(m.sum()))
    return out


def dprime_stats(score: np.ndarray, y: np.ndarray,
                 idxs: list[np.ndarray] | None) -> tuple[float, np.ndarray | None]:
    est = sdt(y, score)["d_auc"]
    if idxs is None:
        return float(est), None
    b = np.array([sdt(y[ix], score[ix])["d_auc"] for ix in idxs])
    return float(est), b


# ---------------------------------------------------------------------------
# Directions (P2)
# ---------------------------------------------------------------------------

def weight_vector(model) -> np.ndarray:
    """Probe weights mapped back to activation space (as in probe_layers.directions)."""
    steps = dict(model.steps)
    lr = steps.get("logisticregressioncv") or steps.get("logisticregression")
    w = (steps["pca"].components_.T @ lr.coef_[0]) if "pca" in steps else lr.coef_[0]
    w = w / steps["standardscaler"].scale_
    return w / np.linalg.norm(w)


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def decision_dir_split(X: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """steer_logits.py "decision": mean(dv above median) - mean(dv below), all chats."""
    hi = dv > np.median(dv)
    return unit(X[hi].mean(0) - X[~hi].mean(0))


def decision_dir_ridge(X: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Literal reading of "regression of dv on activations" (ridge, standardised)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    r = RidgeCV(alphas=np.logspace(0, 7, 15)).fit((X - mu) / sd, dv - dv.mean())
    return unit(r.coef_ / sd)


def crossfit_decision_dirs(X: np.ndarray, dv: np.ndarray, groups: np.ndarray,
                           n_splits: int = 5) -> np.ndarray:
    """Mean of the fold-wise median-split directions (cross-fitted over source groups,
    as in steer_logits.py). Reported next to the all-chats direction."""
    us = []
    k = min(n_splits, len(np.unique(groups)))
    for tr, _ in GroupKFold(n_splits=k).split(X, dv, groups):
        us.append(decision_dir_split(X[tr], dv[tr]))
    return unit(np.mean(us, axis=0))


def pcrandom_ref(X: np.ndarray, y: np.ndarray, u_dec: np.ndarray, n: int,
                 seed: int) -> np.ndarray:
    """Cosines of u_dec with n random directions in the top-20 PCs of the
    within-class-centred activations (steer_logits.py "pcrandom" control)."""
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    Xc = X - np.where((y == 1)[:, None], mu1, mu0)
    _, _, Vt = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
    k = min(20, Vt.shape[0])
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, k)) @ Vt[:k]
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return R @ u_dec


# ---------------------------------------------------------------------------
# Row emission
# ---------------------------------------------------------------------------

def row(**kw) -> dict:
    base = dict(test="", news_corpus="", probe_model="", site="", layer="",
                condition="", seed="", statistic="", estimate=np.nan,
                ci_lo=np.nan, ci_hi=np.nan, n=np.nan, note="")
    base.update(kw)
    return base


def emit_corr(store: dict, rows: list, test: str, news: str, site: str,
              probe_model_of, note: str = "") -> None:
    """store: {(cond, seed, layer_key, subset): (est, boot, n)} -> rows incl. seed means
    and the C1-C0 / C1-C3 / C1-C4 differences (paired over the same resamples)."""
    conds = sorted({k[0] for k in store})
    layer_keys = sorted({k[2] for k in store})
    for (cond, seed, lk, sub), (est, boot, n) in sorted(store.items()):
        lo, hi = ci(boot)
        rows.append(row(test=test, news_corpus=news, probe_model=probe_model_of(cond, seed),
                        site=site, layer=lk, condition=cond, seed=str(seed),
                        statistic=f"spearman_{sub}", estimate=est, ci_lo=lo, ci_hi=hi,
                        n=n, note=note))
    # seed means
    means: dict[tuple[str, str, str], tuple[float, np.ndarray | None, int]] = {}
    for cond in conds:
        for lk in layer_keys:
            for sub in SUBSETS:
                vals = [(e, b, n) for (c, s, l, u), (e, b, n) in store.items()
                        if c == cond and l == lk and u == sub]
                if not vals:
                    continue
                est = float(np.mean([v[0] for v in vals]))
                boot = (np.mean([v[1] for v in vals], axis=0)
                        if all(v[1] is not None for v in vals) else None)
                n = vals[0][2]
                means[(cond, lk, sub)] = (est, boot, n)
                lo, hi = ci(boot)
                rows.append(row(test=test, news_corpus=news,
                                probe_model=probe_model_of(cond, "mean"), site=site,
                                layer=lk, condition=cond, seed="mean",
                                statistic=f"spearman_{sub}", estimate=est, ci_lo=lo,
                                ci_hi=hi, n=n, note=note))
    for a, b in (("C1", "C3"), ("C1", "C4"), ("C1", "C0")):
        for lk in layer_keys:
            for sub in SUBSETS:
                if (a, lk, sub) not in means or (b, lk, sub) not in means:
                    continue
                ea, ba, n = means[(a, lk, sub)]
                eb, bb, _ = means[(b, lk, sub)]
                boot = None if ba is None or bb is None else ba - bb
                lo, hi = ci(boot)
                rows.append(row(test=test, news_corpus=news, probe_model="", site=site,
                                layer=lk, condition=f"{a}-{b}", seed="mean",
                                statistic=f"spearman_diff_{sub}", estimate=ea - eb,
                                ci_lo=lo, ci_hi=hi, n=n, note=note))


def add_mean_layer(store: dict, layers: list[int]) -> None:
    """Add the layer-20/24 mean as its own layer key (per resample -> paired CI)."""
    keys = {(c, s, u) for (c, s, l, u) in store if l in {str(x) for x in layers}}
    for (c, s, u) in keys:
        vals = [store.get((c, s, str(l), u)) for l in layers]
        if any(v is None for v in vals):
            continue
        est = float(np.mean([v[0] for v in vals]))
        boot = (np.mean([v[1] for v in vals], axis=0)
                if all(v[1] is not None for v in vals) else None)
        store[(c, s, MEAN_KEY, u)] = (est, boot, vals[0][2])


# ---------------------------------------------------------------------------
# Pass 2 worker: one fine-tuned model x one news corpus
# ---------------------------------------------------------------------------

def ts_task(args, cond: str, seed: int, news: str, tid: np.ndarray, dv: np.ndarray,
            y: np.ndarray, sid: np.ndarray, idxs: list[np.ndarray]) -> dict:
    """P1b (textmean), P1c (last) and P2 for one run, at the fixed layers."""
    ts_root = Path(args.out_dir) / "training_study"
    stem = f"{args.preset}_ts{cond}s{seed}_raw_mk_acts.npz"
    sites = ("textmean", "last")
    L = list(args.layers)
    chat = reorder(load_acts(ts_root / args.chat_corpus / stem, sites, L), tid, sites)
    newsd = load_acts(ts_root / news / stem, sites, L)

    out = {"corr": {}, "p2": [], "p2_ref_ci": []}
    for si, site in enumerate(sites):
        for li, layer in enumerate(L):
            model, _ = fit_probe(newsd[site][:, li, :].astype(float), newsd["y"], newsd["sid"])
            Xc = chat[site][:, li, :].astype(float)
            score = model.decision_function(Xc)
            for sub, (est, boot, n) in corr_stats(score, dv, y, idxs).items():
                out["corr"][(site, cond, seed, str(layer), sub)] = (est, boot, n)
            if site != "last":
                continue
            # --- P2: news class direction vs chat decision direction, last position
            Xn = newsd[site][:, li, :].astype(float)
            d_class = unit(Xn[newsd["y"] == 1].mean(0) - Xn[newsd["y"] == 0].mean(0))
            w_probe = weight_vector(model)
            dirs = {"split": decision_dir_split(Xc, dv),
                    "ridge": decision_dir_ridge(Xc, dv),
                    "splitcv": crossfit_decision_dirs(Xc, dv, sid)}
            for dname, u_dec in dirs.items():
                out["p2"].append((cond, seed, layer, f"cos_classdiff_decision_{dname}",
                                  float(d_class @ u_dec), len(dv)))
                out["p2"].append((cond, seed, layer, f"cos_probew_decision_{dname}",
                                  float(w_probe @ u_dec), len(dv)))
            ref = pcrandom_ref(Xc, y, dirs["split"], args.n_rand, args.seed + layer)
            out["p2"].append((cond, seed, layer, "cos_random_ref_split",
                              float(np.mean(ref)), args.n_rand))
            out["p2"].append((cond, seed, layer, "cos_random_ref_split_abs95",
                              float(np.quantile(np.abs(ref), 0.95)), args.n_rand))
            out["p2_ref_ci"].append((cond, seed, layer,
                                     float(np.quantile(ref, 0.025)),
                                     float(np.quantile(ref, 0.975))))
            out["p2"].append((cond, seed, layer, "cos_classdiff_probew",
                              float(d_class @ w_probe), len(newsd["y"])))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR,
                    help="root of the outputs tree (default: config.OUTPUT_DIR)")
    ap.add_argument("--out", type=Path, default=None,
                    help="result CSV (default: <out-dir>/training_study/prediction_test.csv)")
    ap.add_argument("--preset", default="olmo2-7b")
    ap.add_argument("--base-variant", default="base")
    ap.add_argument("--chat-corpus", default="chat_tw25")
    ap.add_argument("--news-corpora", nargs="+", default=["gnad_heldout", "journalism_seg"])
    ap.add_argument("--layers", nargs="+", type=int, default=[20, 24])
    ap.add_argument("--curve-step", type=int, default=1)
    ap.add_argument("--no-curve", action="store_true")
    ap.add_argument("--tests", nargs="+",
                    default=["context", "P0", "P1", "P1b", "P1c", "P2"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-rand", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--answers-summary", type=Path, default=None,
                    help="answers_summary.csv to check the context d' against")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    root = Path(args.out_dir)
    ts_root = root / "training_study"
    out_csv = args.out or (ts_root / "prediction_test.csv")
    rows: list[dict] = []

    # ---- answers (order-averaged dv per run) -------------------------------
    tags = run_tags(ts_root / args.chat_corpus, args.preset)
    if not tags:
        raise SystemExit(f"no runs in {ts_root / args.chat_corpus}")
    print(f"runs: {[f'{c}s{s}' for c, s in tags]}", flush=True)
    ans = {}
    for cond, seed in tags:
        f = ts_root / args.chat_corpus / f"{args.preset}_ts{cond}s{seed}_raw_mk.csv"
        a = order_avg_answers([f, f.with_name(f.name.replace("_raw_mk", "_raw_km"))])
        if a is None:
            raise SystemExit(f"missing answers for {cond}s{seed}")
        a["text_id"] = a["text_id"].astype(str)
        ans[(cond, seed)] = a.set_index("text_id")

    # ---- base activations --------------------------------------------------
    chat_f = root / args.chat_corpus / f"{args.preset}_{args.base_variant}_raw_mk_acts.npz"
    base_chat = load_acts(chat_f, ("textmean",))
    tid, y, sid = base_chat["tid"], base_chat["y"], base_chat["sid"]
    n_layers = base_chat["textmean"].shape[1]
    dv = {k: v["dv"].reindex(tid).to_numpy() for k, v in ans.items()}
    for k, v in dv.items():
        if np.isnan(v).any():
            raise SystemExit(f"answers do not cover all chat texts for {k}")
        lab = ans[k]["is_ai"].reindex(tid).to_numpy()
        if not np.array_equal(lab.astype(int), y):
            raise SystemExit(f"is_ai mismatch between answers and activations for {k}")
    idxs = boot_indices(sid, args.n_boot, args.seed)
    print(f"{len(tid)} chat texts | {len(np.unique(sid))} source groups | "
          f"{n_layers} layers | {args.n_boot} resamples", flush=True)

    # ---- context: each model's own d' on the chats -------------------------
    if "context" in args.tests:
        ref = None
        if args.answers_summary and Path(args.answers_summary).exists():
            r = pd.read_csv(args.answers_summary)
            ref = r[r["corpus"] == args.chat_corpus].set_index(["condition", "seed"])["d"]
        bad = []
        for (cond, seed) in tags:
            est, boot = dprime_stats(dv[(cond, seed)], y, idxs)
            lo, hi = ci(boot)
            rows.append(row(test="context", site="answers", condition=cond, seed=str(seed),
                            statistic="d_auc_answers", estimate=est, ci_lo=lo, ci_hi=hi,
                            n=len(y)))
            if ref is not None and (cond, seed) in ref.index:
                if abs(float(ref.loc[(cond, seed)]) - est) > args.tol:
                    bad.append((cond, seed, float(ref.loc[(cond, seed)]), est))
        if bad:
            raise SystemExit(f"context d' does not match answers_summary.csv: {bad}")
        print("context d' matches answers_summary.csv" if ref is not None
              else "context d' computed (no answers_summary.csv given)", flush=True)

    # ---- P0 / P1: base-model news probe on base-model chat activations -----
    curve_layers = ([] if args.no_curve
                    else list(range(1, n_layers, max(1, args.curve_step))))
    fixed = list(args.layers)
    need = sorted(set(fixed) | set(curve_layers))
    if "P0" in args.tests or "P1" in args.tests:
        for news in args.news_corpora:
            nf = root / news / f"{args.preset}_{args.base_variant}_raw_mk_acts.npz"
            if not nf.exists():
                print(f"skip {news}: no base activations", flush=True)
                continue
            nd = load_acts(nf, ("textmean",))
            print(f"[P0/P1] {news}: {len(nd['y'])} news texts, fitting {len(need)} probes",
                  flush=True)

            got = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(fit_score)(nd["textmean"][:, l, :], nd["y"], nd["sid"],
                                   base_chat["textmean"][:, l, :]) for l in need)
            scores = dict(zip(need, got))
            del nd
            pm = f"{args.base_variant}"
            p0_store = {}
            for layer in need:
                boot_ok = layer in fixed
                est, boot = dprime_stats(scores[layer], y, idxs if boot_ok else None)
                lo, hi = ci(boot)
                rows.append(row(test="P0", news_corpus=news, probe_model=pm,
                                site="textmean", layer=str(layer), statistic="d_auc",
                                estimate=est, ci_lo=lo, ci_hi=hi, n=len(y),
                                note="" if boot_ok else "curve (descriptive)"))
                if boot_ok:
                    p0_store[layer] = (est, boot)
            if len(p0_store) == len(fixed) and fixed:
                est = float(np.mean([p0_store[l][0] for l in fixed]))
                boot = np.mean([p0_store[l][1] for l in fixed], axis=0)
                lo, hi = ci(boot)
                rows.append(row(test="P0", news_corpus=news, probe_model=pm,
                                site="textmean", layer=MEAN_KEY, statistic="d_auc",
                                estimate=est, ci_lo=lo, ci_hi=hi, n=len(y)))

            if "P1" in args.tests:
                store = {}
                for layer in need:
                    boot_ok = layer in fixed
                    for (cond, seed) in tags:
                        st = corr_stats(scores[layer], dv[(cond, seed)], y,
                                        idxs if boot_ok else None)
                        for sub, v in st.items():
                            if boot_ok:
                                store[(cond, seed, str(layer), sub)] = v
                            else:
                                rows.append(row(test="P1", news_corpus=news,
                                                probe_model=pm, site="textmean",
                                                layer=str(layer), condition=cond,
                                                seed=str(seed),
                                                statistic=f"spearman_{sub}",
                                                estimate=v[0], n=v[2],
                                                note="curve (descriptive)"))
                    print(f"  layer {layer} done", flush=True)
                add_mean_layer(store, fixed)
                emit_corr(store, rows, "P1", news, "textmean", lambda c, s: pm)
                print(f"[P1] {news}: done", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    # ---- P1b / P1c / P2: probes in the fine-tuned models themselves --------
    want2 = [t for t in ("P1b", "P1c", "P2") if t in args.tests]
    if want2:
        jobs = [(c, s, news) for news in args.news_corpora for (c, s) in tags
                if (ts_root / news / f"{args.preset}_ts{c}s{s}_raw_mk_acts.npz").exists()]
        print(f"[P1b/P1c/P2] {len(jobs)} model x corpus tasks", flush=True)
        res = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(ts_task)(args, c, s, news, tid, dv[(c, s)], y, sid, idxs)
            for (c, s, news) in jobs)
        by_news: dict[str, dict] = {}
        for (c, s, news), r in zip(jobs, res):
            d = by_news.setdefault(news, {"corr": {}, "p2": [], "p2_ref_ci": []})
            d["corr"].update(r["corr"])
            d["p2"].extend(r["p2"])
            d["p2_ref_ci"].extend(r.get("p2_ref_ci", []))
        for news, d in by_news.items():
            for site, test in (("textmean", "P1b"), ("last", "P1c")):
                if test not in args.tests:
                    continue
                store = {k[1:]: v for k, v in d["corr"].items() if k[0] == site}
                add_mean_layer(store, fixed)
                note = ("partly circular: the last position carries the trained readout"
                        if test == "P1c" else "")
                emit_corr(store, rows, test, news, site,
                          lambda c, s: f"ts{c}" if s == "mean" else f"ts{c}s{s}", note)
            if "P2" not in args.tests:
                continue
            ref_ci = {(c, s, l): (lo, hi) for (c, s, l, lo, hi) in d["p2_ref_ci"]}
            per_layer: dict[tuple[str, str, str], list[float]] = {}
            for (c, s, layer, stat, est, n) in d["p2"]:
                lo, hi = ref_ci.get((c, s, layer), (np.nan, np.nan)) \
                    if stat == "cos_random_ref_split" else (np.nan, np.nan)
                rows.append(row(test="P2", news_corpus=news, probe_model=f"ts{c}s{s}",
                                site="last", layer=str(layer), condition=c, seed=str(s),
                                statistic=stat, estimate=est, ci_lo=lo, ci_hi=hi, n=n))
                per_layer.setdefault((c, stat, str(layer)), {}).setdefault(s, []).append(est)
                per_layer.setdefault((c, stat, MEAN_KEY), {}).setdefault(s, []).append(est)
            for (c, stat, lk), by_seed in sorted(per_layer.items()):
                # one value per seed (for MEAN_KEY: that seed's mean over the fixed layers), so
                # that min/max is a range across seeds and not across seeds and layers
                vals = [float(np.mean(v)) for v in by_seed.values()]
                rows.append(row(test="P2", news_corpus=news, probe_model="", site="last",
                                layer=lk, condition=c, seed="mean", statistic=stat,
                                estimate=float(np.mean(vals)),
                                ci_lo=float(np.min(vals)), ci_hi=float(np.max(vals)),
                                n=len(vals), note="seed mean (ci = min/max over seeds)"))

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n-> {out_csv}  ({len(df)} rows)", flush=True)

    key = df[(df["layer"] == MEAN_KEY) & (df["seed"].isin(["mean", ""]))]
    with pd.option_context("display.width", 220, "display.max_rows", 400,
                           "display.float_format", "{:.3f}".format):
        print(key[["test", "news_corpus", "site", "condition", "statistic",
                   "estimate", "ci_lo", "ci_hi", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
