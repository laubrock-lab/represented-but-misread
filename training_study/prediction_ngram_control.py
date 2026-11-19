"""Reviewer control for P1: does a *non-probe* classifier trained on the same news
predict the fine-tuned models' chat answers as well as the news probe does?

The reviewer's point: any competent AI-vs-human classifier fitted on the news texts
should agree with the C1 model on the chats, whatever the internal mechanism. This
script therefore replaces the news *probe* (a linear readout of the base model's
text-mean activations) by two surface-level text classifiers fitted on exactly the
same news texts -- the n-gram pipelines of baselines.py:

    word_ngram   TF-IDF word 1-2-grams  + logistic regression (C=10)
    char_ngram   TF-IDF char_wb 2-4-grams + logistic regression (C=10)

Both are fitted on the gnad_heldout texts (label is_ai) and applied unchanged to the
300 chat_tw25 texts; one score per chat per classifier.

Tests (all on the same 300 chats, the same 61 source groups and the SAME 1 000
grouped-bootstrap resamples as prediction_test.py, so every seed mean and difference
is paired):

N0  d' (AUC-based, sdt_summary.sdt) of each n-gram score on the chats.
N1  the P1 statistics with the n-gram score in place of the probe score: Spearman
    correlation with the order-averaged dv, pooled / within human / within AI, per
    condition and seed, the seed means, and the differences C1-C3, C1-C4, C1-C0.
N2  agreement between predictors: Spearman across chats between each n-gram score and
    the news-probe score used in P1.
N3  unique contributions for the C1 models (C3 as a reference): partial Spearman
    (rank-transform, then residualise the ranks linearly) of the probe score with dv
    controlling for the n-gram score, and of the n-gram score with dv controlling for
    the probe score; seed means with grouped-bootstrap CIs, pooled and within class.

The news probe is the P1 predictor: probe_transfer.fit_probe on the BASE model's
text-mean activations of gnad_heldout, applied to the base model's text-mean chat
activations, fitted separately at layers 20 and 24. prediction_test.py correlates per
layer and averages the per-layer statistics afterwards (add_mean_layer), so this script
does the same: every probe-based statistic is computed per layer and the "20_24mean"
row is the mean of the two per-layer values (noted in the `note` column). As a
consistency check the P1 seed-mean pooled correlation for C1 is recomputed here and
compared with prediction_test.csv (0.675).

Texts are read only inside this script and are never printed or written out; the
output contains numbers and anonymous text ids only.

Usage (cluster, CPU):
    python training_study/prediction_ngram_control.py --n-jobs 4

Output: outputs/training_study/prediction_ngram_control.csv (long format).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from prediction_test import (MEAN_KEY, add_mean_layer, boot_indices, ci,  # noqa: E402
                             corr_stats, dprime_stats, emit_corr, fit_score,
                             load_acts, run_tags, spearman)
from analyze_answers import answers as order_avg_answers  # noqa: E402

NGRAMS = ("word_ngram", "char_ngram")
PROBE = "news_probe"
SUBSETS = ("pooled", "human", "ai")


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def ngram_pipeline(kind: str):
    """The n-gram pipelines of baselines.py (oof_scores), parameters verbatim.

    baselines.py builds them inline inside its cross-validation loop and exposes no
    factory, so the two pipelines are re-stated here with identical parameters.
    """
    if kind == "char_ngram":
        return make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                            sublinear_tf=True),
            LogisticRegression(C=10, max_iter=5000))
    if kind == "word_ngram":
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            lowercase=True),
            LogisticRegression(C=10, max_iter=5000))
    raise ValueError(kind)


def texts_path(data_dir: Path, corpus: str) -> Path:
    for p in (data_dir / f"texts_{corpus}.parquet",
              data_dir / "training_study" / f"texts_{corpus}.parquet"):
        if p.exists():
            return p
    raise SystemExit(f"no text file for corpus {corpus} under {data_dir}")


def load_texts(data_dir: Path, corpus: str) -> pd.DataFrame:
    d = pd.read_parquet(texts_path(data_dir, corpus),
                        columns=["text_id", "source_id", "is_ai", "text"])
    d["text_id"] = d["text_id"].astype(str)
    return d


# ---------------------------------------------------------------------------
# Partial Spearman (N3)
# ---------------------------------------------------------------------------

def _resid(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Residual of the centred vector r after a linear fit on the centred vector c."""
    den = float(c @ c)
    return r if den <= 0 else r - (float(r @ c) / den) * c


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Spearman partial correlation of a and b given c: rank-transform all three,
    residualise the ranks of a and of b linearly on the ranks of c, correlate."""
    if len(a) < 4:
        return np.nan
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    ra, rb, rc = ra - ra.mean(), rb - rb.mean(), rc - rc.mean()
    ea, eb = _resid(ra, rc), _resid(rb, rc)
    den = np.sqrt(float(ea @ ea) * float(eb @ eb))
    return float(ea @ eb / den) if den > 0 else np.nan


def subset_stats(fn, y: np.ndarray, idxs: list[np.ndarray] | None, *vecs) -> dict:
    """{subset: (estimate, boot array or None, n)} -- corr_stats generalised to fn."""
    masks = dict(pooled=np.ones(len(y), bool), human=y == 0, ai=y == 1)
    out = {}
    for name, m in masks.items():
        est = fn(*[v[m] for v in vecs])
        if idxs is None:
            out[name] = (est, None, int(m.sum()))
            continue
        b = np.empty(len(idxs))
        for k, ix in enumerate(idxs):
            j = ix[m[ix]]
            b[k] = fn(*[v[j] for v in vecs])
        out[name] = (est, b, int(m.sum()))
    return out


# ---------------------------------------------------------------------------
# Row emission (long format: test, classifier, condition, seed, statistic, ...)
# ---------------------------------------------------------------------------

def row(**kw) -> dict:
    base = dict(test="", classifier="", condition="", seed="", statistic="",
                estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=np.nan, note="")
    base.update(kw)
    return base


def with_layer(note: str, layer: str) -> str:
    if not layer or layer == "ngram":   # the n-gram scores have no layer
        return note
    tag = ("mean of the per-layer statistics at layers 20 and 24 (as P1)"
           if layer == MEAN_KEY else f"layer={layer}")
    return f"{note}; {tag}" if note else tag


def convert(pt_rows: list[dict], classifier: str) -> list[dict]:
    """prediction_test.row dicts (as emitted by emit_corr) -> output schema."""
    out = []
    for r in pt_rows:
        out.append(row(test=r["test"], classifier=r["probe_model"] or classifier,
                       condition=r["condition"], seed=r["seed"],
                       statistic=r["statistic"], estimate=r["estimate"],
                       ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n=r["n"],
                       note=with_layer(r["note"], str(r["layer"]))))
    return out


def emit_seed_means(rows: list[dict], test: str, classifier: str, store: dict,
                    stat: str, note: str = "") -> None:
    """store {(cond, seed, layer, subset): (est, boot, n)} -> per-seed and seed-mean
    rows (the seed mean of the paired bootstrap arrays, as in emit_corr)."""
    for (cond, seed, lk, sub), (est, boot, n) in sorted(store.items()):
        lo, hi = ci(boot)
        rows.append(row(test=test, classifier=classifier, condition=cond,
                        seed=str(seed), statistic=f"{stat}_{sub}", estimate=est,
                        ci_lo=lo, ci_hi=hi, n=n, note=with_layer(note, lk)))
    for cond in sorted({k[0] for k in store}):
        for lk in sorted({k[2] for k in store}):
            for sub in SUBSETS:
                vals = [(e, b, n) for (c, s, l, u), (e, b, n) in store.items()
                        if c == cond and l == lk and u == sub]
                if not vals:
                    continue
                est = float(np.mean([v[0] for v in vals]))
                boot = (np.mean([v[1] for v in vals], axis=0)
                        if all(v[1] is not None for v in vals) else None)
                lo, hi = ci(boot)
                rows.append(row(test=test, classifier=classifier, condition=cond,
                                seed="mean", statistic=f"{stat}_{sub}", estimate=est,
                                ci_lo=lo, ci_hi=hi, n=vals[0][2],
                                note=with_layer(note, lk)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <out-dir>/training_study/prediction_ngram_control.csv")
    ap.add_argument("--preset", default="olmo2-7b")
    ap.add_argument("--base-variant", default="base")
    ap.add_argument("--chat-corpus", default="chat_tw25")
    ap.add_argument("--news-corpus", default="gnad_heldout")
    ap.add_argument("--layers", nargs="+", type=int, default=[20, 24])
    ap.add_argument("--n3-conditions", nargs="+", default=["C1", "C3"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reference-csv", type=Path, default=None,
                    help="default: <out-dir>/training_study/prediction_test.csv")
    ap.add_argument("--reference-value", type=float, default=0.675,
                    help="fallback if the reference CSV is missing")
    args = ap.parse_args()

    root = Path(args.out_dir)
    ts_root = root / "training_study"
    out_csv = args.out or (ts_root / "prediction_ngram_control.csv")
    ref_csv = args.reference_csv or (ts_root / "prediction_test.csv")
    news = args.news_corpus
    L = list(args.layers)
    rows: list[dict] = []

    # ---- answers (order-averaged dv per run), exactly as prediction_test.py -----
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

    # ---- base activations of the chats: canonical ids, labels, groups ----------
    chat_f = root / args.chat_corpus / f"{args.preset}_{args.base_variant}_raw_mk_acts.npz"
    base_chat = load_acts(chat_f, ("textmean",), L)
    tid, y, sid = base_chat["tid"], base_chat["y"], base_chat["sid"]
    dv = {k: v["dv"].reindex(tid).to_numpy() for k, v in ans.items()}
    for k, v in dv.items():
        if np.isnan(v).any():
            raise SystemExit(f"answers do not cover all chat texts for {k}")
        lab = ans[k]["is_ai"].reindex(tid).to_numpy()
        if not np.array_equal(lab.astype(int), y):
            raise SystemExit(f"is_ai mismatch between answers and activations for {k}")
    idxs = boot_indices(sid, args.n_boot, args.seed)
    print(f"{len(tid)} chat texts | {len(np.unique(sid))} source groups | "
          f"{args.n_boot} resamples", flush=True)

    # ---- the P1 news-probe scores (base model, text-mean, per layer) -----------
    nf = root / news / f"{args.preset}_{args.base_variant}_raw_mk_acts.npz"
    nd = load_acts(nf, ("textmean",), L)
    print(f"[probe] {news}: {len(nd['y'])} news texts, fitting {len(L)} probes",
          flush=True)
    got = Parallel(n_jobs=min(args.n_jobs, len(L)), verbose=5)(
        delayed(fit_score)(nd["textmean"][:, li, :], nd["y"], nd["sid"],
                           base_chat["textmean"][:, li, :]) for li in range(len(L)))
    probe = {str(l): s for l, s in zip(L, got)}
    del nd, base_chat

    # ---- the n-gram classifiers: fit on the news texts, apply to the chats -----
    news_txt = load_texts(args.data_dir, news)
    chat_txt = load_texts(args.data_dir, args.chat_corpus).set_index("text_id").reindex(tid)
    if chat_txt["text"].isna().any():
        raise SystemExit("chat text file does not cover all 300 activation text_ids")
    if not np.array_equal(chat_txt["is_ai"].to_numpy().astype(int), y):
        raise SystemExit("is_ai mismatch between chat texts and activations")
    if not np.array_equal(chat_txt["source_id"].astype(str).to_numpy(), sid):
        raise SystemExit("source_id mismatch between chat texts and activations")
    print(f"[ngram] fit on {len(news_txt)} {news} texts -> {len(chat_txt)} chats",
          flush=True)
    ng = {}
    for kind in NGRAMS:
        m = ngram_pipeline(kind)
        m.fit(news_txt["text"], news_txt["is_ai"].to_numpy().astype(int))
        ng[kind] = m.decision_function(chat_txt["text"])
        print(f"  {kind}: fitted", flush=True)
    del news_txt, chat_txt

    # ---- consistency check: P1 seed-mean pooled correlation for C1 -------------
    c1 = [(c, s) for (c, s) in tags if c == "C1"]
    vals = [spearman(probe[str(l)], dv[k]) for l in L for k in c1]
    got_c1 = float(np.mean(vals))
    ref = args.reference_value
    if Path(ref_csv).exists():
        r = pd.read_csv(ref_csv)
        sel = r[(r["test"] == "P1") & (r["news_corpus"] == news)
                & (r["layer"] == MEAN_KEY) & (r["condition"] == "C1")
                & (r["seed"] == "mean") & (r["statistic"] == "spearman_pooled")
                & (r["site"] == "textmean")]
        if len(sel) != 1:
            raise SystemExit(f"cannot identify the P1 reference row in {ref_csv} "
                             f"({len(sel)} matches)")
        ref = float(sel["estimate"].iloc[0])
    if round(got_c1, 3) != round(ref, 3):
        raise SystemExit(f"CONSISTENCY CHECK FAILED: recomputed P1 C1 seed-mean pooled "
                         f"correlation {got_c1:.6f} != reference {ref:.6f}")
    print(f"consistency check ok: P1 C1 seed-mean pooled = {got_c1:.6f} "
          f"(reference {ref:.6f})", flush=True)
    rows.append(row(test="check", classifier=PROBE, condition="C1", seed="mean",
                    statistic="spearman_pooled", estimate=got_c1, n=len(y),
                    note=f"recomputed P1 value; prediction_test.csv = {ref:.6f}; "
                         + with_layer("", MEAN_KEY)))

    # ---- N0: d' of each score on the chats -------------------------------------
    for kind in NGRAMS:
        est, boot = dprime_stats(ng[kind], y, idxs)
        lo, hi = ci(boot)
        rows.append(row(test="N0", classifier=kind, statistic="d_auc", estimate=est,
                        ci_lo=lo, ci_hi=hi, n=len(y)))
    p0 = {}
    for l in L:
        est, boot = dprime_stats(probe[str(l)], y, idxs)
        lo, hi = ci(boot)
        p0[l] = (est, boot)
        rows.append(row(test="N0", classifier=PROBE, statistic="d_auc", estimate=est,
                        ci_lo=lo, ci_hi=hi, n=len(y),
                        note=with_layer("reference (= P0)", str(l))))
    est = float(np.mean([p0[l][0] for l in L]))
    boot = np.mean([p0[l][1] for l in L], axis=0)
    lo, hi = ci(boot)
    rows.append(row(test="N0", classifier=PROBE, statistic="d_auc", estimate=est,
                    ci_lo=lo, ci_hi=hi, n=len(y),
                    note=with_layer("reference (= P0)", MEAN_KEY)))
    print("[N0] done", flush=True)

    # ---- N1: the P1 statistics with the n-gram score ---------------------------
    for kind in NGRAMS:
        store = {}
        for (cond, seed) in tags:
            for sub, v in corr_stats(ng[kind], dv[(cond, seed)], y, idxs).items():
                store[(cond, seed, "ngram", sub)] = v
        pt: list[dict] = []
        emit_corr(store, pt, "N1", news, "text", lambda c, s: kind)
        rows += convert(pt, kind)
        print(f"[N1] {kind} done", flush=True)
    store = {}
    for l in L:
        for (cond, seed) in tags:
            for sub, v in corr_stats(probe[str(l)], dv[(cond, seed)], y, idxs).items():
                store[(cond, seed, str(l), sub)] = v
    add_mean_layer(store, L)
    pt = []
    emit_corr(store, pt, "N1", news, "textmean", lambda c, s: PROBE,
              "reference (= P1)")
    rows += convert(pt, PROBE)
    print("[N1] probe reference done", flush=True)

    # ---- N2: agreement between the predictors ----------------------------------
    for kind in NGRAMS:
        per_layer = {}
        for l in L:
            est = spearman(ng[kind], probe[str(l)])
            boot = np.array([spearman(ng[kind][ix], probe[str(l)][ix]) for ix in idxs])
            per_layer[l] = (est, boot)
            lo, hi = ci(boot)
            rows.append(row(test="N2", classifier=kind,
                            statistic=f"spearman_vs_{PROBE}", estimate=est, ci_lo=lo,
                            ci_hi=hi, n=len(y), note=with_layer("", str(l))))
        est = float(np.mean([per_layer[l][0] for l in L]))
        boot = np.mean([per_layer[l][1] for l in L], axis=0)
        lo, hi = ci(boot)
        rows.append(row(test="N2", classifier=kind, statistic=f"spearman_vs_{PROBE}",
                        estimate=est, ci_lo=lo, ci_hi=hi, n=len(y),
                        note=with_layer("", MEAN_KEY)))
    est = spearman(ng["word_ngram"], ng["char_ngram"])
    boot = np.array([spearman(ng["word_ngram"][ix], ng["char_ngram"][ix]) for ix in idxs])
    lo, hi = ci(boot)
    rows.append(row(test="N2", classifier="word_ngram",
                    statistic="spearman_vs_char_ngram", estimate=est, ci_lo=lo,
                    ci_hi=hi, n=len(y)))
    print("[N2] done", flush=True)

    # ---- N3: unique contributions (C1, reference C3) ---------------------------
    n3_tags = [(c, s) for (c, s) in tags if c in args.n3_conditions]
    for kind in NGRAMS:
        s_probe_given_ng, s_ng_given_probe = {}, {}
        for (cond, seed) in n3_tags:
            d = dv[(cond, seed)]
            for l in L:
                p = probe[str(l)]
                for sub, v in subset_stats(partial_spearman, y, idxs, p, d,
                                           ng[kind]).items():
                    s_probe_given_ng[(cond, seed, str(l), sub)] = v
                for sub, v in subset_stats(partial_spearman, y, idxs, ng[kind], d,
                                           p).items():
                    s_ng_given_probe[(cond, seed, str(l), sub)] = v
        for st, stat in ((s_probe_given_ng, f"partial_probe_dv_given_{kind}"),
                         (s_ng_given_probe, f"partial_{kind}_dv_given_probe")):
            add_mean_layer(st, L)
            emit_seed_means(rows, "N3", PROBE if "probe_dv" in stat else kind, st, stat)
        print(f"[N3] {kind} done", flush=True)
    # zero-order references at the same layer keys
    store = {}
    for (cond, seed) in n3_tags:
        for l in L:
            for sub, v in corr_stats(probe[str(l)], dv[(cond, seed)], y, idxs).items():
                store[(cond, seed, str(l), sub)] = v
    add_mean_layer(store, L)
    emit_seed_means(rows, "N3", PROBE, store, "spearman_probe_dv", "zero-order")
    for kind in NGRAMS:
        store = {}
        for (cond, seed) in n3_tags:
            for sub, v in corr_stats(ng[kind], dv[(cond, seed)], y, idxs).items():
                store[(cond, seed, "ngram", sub)] = v
        emit_seed_means(rows, "N3", kind, store, f"spearman_{kind}_dv", "zero-order")

    df = pd.DataFrame(rows, columns=["test", "classifier", "condition", "seed",
                                     "statistic", "estimate", "ci_lo", "ci_hi", "n",
                                     "note"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n-> {out_csv}  ({len(df)} rows)", flush=True)

    with pd.option_context("display.width", 220, "display.max_rows", 600,
                           "display.float_format", "{:.3f}".format):
        print("\n== N0 ==")
        print(df[df.test == "N0"].to_string(index=False))
        print("\n== N1 (seed means and differences, pooled/human/ai) ==")
        print(df[(df.test == "N1") & (df.seed == "mean")].to_string(index=False))
        print("\n== N2 ==")
        print(df[df.test == "N2"].to_string(index=False))
        print("\n== N3 (seed means) ==")
        print(df[(df.test == "N3") & (df.seed == "mean")].to_string(index=False))


if __name__ == "__main__":
    main()
