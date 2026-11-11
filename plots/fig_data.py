"""Tidy plot data for the paper figures (no text content; IDs and numbers only).

Writes outputs/figdata/*.csv, read by plots/figures.R:
    fig1_info.csv        sensitivity of baselines, probes, EditLens, and model answers per corpus
    fig2_models.csv      base vs. final post-trained d' (raw and chat) per family x corpus, with CIs
    fig2_contrasts.csv   clean contrasts (final raw - base raw) with Holm/BH p-values
    fig2_fewshot.csv     few-shot curves (k = 0, 2, 8, 16), seed range
    fig3_stitch.csv      stitching curves with CIs (+ parent d')
    fig3_patch.csv       patching at layers 8 and 16 with CIs

Usage:  python plots/fig_data.py
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from sdt_summary import sdt  # noqa: E402

OUT = config.OUTPUT_DIR / "figdata"
CORPORA = {"chat_tw25": "Chats (25 words)", "journalism_seg": "News segments", "textbook": "Textbook summaries"}
FAMILIES = {"gemma3-4b": "Gemma 3 4B", "gemma3-12b": "Gemma 3 12B", "qwen3-8b": "Qwen3 8B",
            "olmo2-7b": "OLMo 2 7B", "apertus15-8b": "Apertus 1.5 8B", "olmo3-7b-think": "OLMo 3 7B Think"}
RNG = np.random.default_rng(1)
NB = 1000


def boot(y, s, groups, fn) -> tuple[float, float]:
    idx = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).indices
    keys = np.array(list(idx), dtype=object)
    vals = []
    for _ in range(NB):
        i = np.concatenate([idx[k] for k in RNG.choice(keys, len(keys))])
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], s[i]))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def dprime(y, s):
    return sdt(y, s)["d_auc"]


def order_avg(files) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, usecols=["text_id", "source_id", "is_ai", "dv", "p_mass"]) for f in files])
    n = df.groupby("text_id").size()
    df = df[df.text_id.isin(n[n == 2].index)]
    return df.groupby(["text_id", "source_id", "is_ai"], as_index=False).agg(dv=("dv", "mean"), p_mass=("p_mass", "mean"))


def cond_files(corpus, preset, variant, fmt):
    sfx = "_neutral" if fmt == "chat" else ""
    fs = [config.OUTPUT_DIR / corpus / f"{preset}_{variant}_{fmt}_{o}{sfx}.csv" for o in ("mk", "km")]
    return fs if all(f.exists() for f in fs) else None


def model_rows() -> pd.DataFrame:
    rows = []
    for corpus, clab in CORPORA.items():
        for preset, flab in FAMILIES.items():
            for variant, fmt in [("base", "raw"), ("instruct", "raw"), ("instruct", "chat")]:
                fs = cond_files(corpus, preset, variant, fmt)
                if fs is None:
                    continue
                a = order_avg(fs)
                y, s = a.is_ai.to_numpy(), a.dv.to_numpy()
                lo, hi = boot(y, s, a.source_id, dprime)
                rows.append(dict(corpus=clab, family=flab, stage="Base" if variant == "base" else "Post-trained",
                                 fmt=fmt, d=dprime(y, s), lo=lo, hi=hi, p_mass=a.p_mass.mean(),
                                 in_format=a.p_mass.mean() >= .3))
    return pd.DataFrame(rows)


def fig1(models: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_csv(config.OUTPUT_DIR / "baselines.csv")
    lab = {"word_ngram": "Word n-grams", "char_ngram": "Character n-grams", "surface": "Surface statistics",
           "emb_L0": "Input embedding (probe)", "probe_67": "Model activations (probe)"}
    rows = []
    for _, r in b[b.corpus.isin(CORPORA)].iterrows():
        rows.append(dict(corpus=CORPORA[r.corpus], measure=lab[r.baseline], group="Text and representation",
                         d=r.d_auc, lo=r.d_auc_lo, hi=r.d_auc_hi,
                         interval="range over models" if r.baseline in ("emb_L0", "probe_67") else "95% CI"))
    for corpus, clab in CORPORA.items():
        f = config.OUTPUT_DIR / corpus / "editlens_llama32-3b.csv"
        e = pd.read_csv(f)
        y, s = e.is_ai.to_numpy(), e.score.to_numpy()
        lo, hi = boot(y, s, e.source_id, dprime)
        rows.append(dict(corpus=clab, measure="EditLens (trained detector)", group="Text and representation",
                         d=dprime(y, s), lo=lo, hi=hi, interval="95% CI"))
    m = models[(models.fmt == "raw") & models.in_format]
    for _, r in m.iterrows():
        rows.append(dict(corpus=r.corpus, measure=f"Answers: {r.stage.lower()} models", group="Model answers",
                         d=r.d, lo=r.lo, hi=r.hi, interval="per family", family=r.family))
    # ceiling of the AUC-based d' (AUC = 1 - 1/(2 n_min)) per corpus
    out = pd.DataFrame(rows)
    ceil = {}
    for corpus, clab in CORPORA.items():
        t = pd.read_parquet(config.REPO / "data" / f"texts_{corpus}.parquet", columns=["is_ai"])
        nmin = min(t.is_ai.sum(), (1 - t.is_ai).sum())
        ceil[clab] = float(np.sqrt(2) * norm.ppf(1 - 1 / (2 * nmin)))
    out["ceiling"] = out.corpus.map(ceil)
    return out


def fewshot() -> pd.DataFrame:
    d = pd.read_csv(config.OUTPUT_DIR / "fewshot" / "fewshot_summary.csv")
    d = d[(d.fmt == "raw") & d.preset.isin(["gemma3-4b", "qwen3-8b", "olmo2-7b"]) & d.variant.isin(["base", "instruct"])]
    rows = []
    for (corpus, preset, variant), g in d.groupby(["corpus", "preset", "variant"]):
        first = g.iloc[0]
        rows.append(dict(corpus=CORPORA[corpus], family=FAMILIES[preset],
                         stage="Base" if variant == "base" else "Post-trained", k=0, d=first.d_zero,
                         lo=np.nan, hi=np.nan))
        for _, r in g.iterrows():
            rows.append(dict(corpus=CORPORA[corpus], family=FAMILIES[preset],
                             stage="Base" if variant == "base" else "Post-trained", k=int(r.k), d=r.d_few,
                             lo=r.d_seed_min, hi=r.d_seed_max))
    return pd.DataFrame(rows).sort_values(["corpus", "family", "stage", "k"])


def stitch() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(config.OUTPUT_DIR / "stitch" / "textbook" / "*_raw_mk.csv")):
        m = re.match(r".*/(.+?)_(base|sft)-(base|sft)_L(\d+)(?:_(attn|mlp))?_raw_mk.csv", f)
        if not m:
            continue
        p, lo_, up, L, comp = m.groups()
        a = order_avg([f, f.replace("_mk.csv", "_km.csv")])
        y, s = a.is_ai.to_numpy(), a.dv.to_numpy()
        l, h = boot(y, s, a.source_id, dprime)
        rows.append(dict(family=FAMILIES[p], lower=lo_, upper=up, layers=int(L), component=comp or "all",
                         d=dprime(y, s), lo=l, hi=h, p_mass=a.p_mass.mean()))
    out = pd.DataFrame(rows)
    for p in ["apertus15-8b", "olmo2-7b"]:
        for v in ["base", "sft"]:
            a = order_avg(cond_files("textbook", p, v, "raw"))
            out.loc[out.family == FAMILIES[p], f"parent_{v}"] = dprime(a.is_ai.to_numpy(), a.dv.to_numpy())
    return out.sort_values(["family", "component", "lower", "layers"])


def patch() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(config.OUTPUT_DIR / "patch" / "textbook" / "*.csv")):
        m = re.match(r".*/(.+?)_(\w+)-from-(\w+)_L(\d+)\.csv$", f)
        if not m:
            continue
        p, rec, don, L = m.groups()
        df = pd.read_csv(f)
        avg = df.groupby(["set", "text_id", "is_ai", "source_id"], as_index=False).dv.mean()
        for s_, g in avg.groupby("set"):
            y, s = g.is_ai.to_numpy(), g.dv.to_numpy()
            l, h = boot(y, s, g.source_id, dprime)
            rows.append(dict(family=FAMILIES[p], recipient=rec, donor=don, layer=int(L), set=s_,
                             d=dprime(y, s), lo=l, hi=h))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = model_rows()
    models.to_csv(OUT / "fig2_models.csv", index=False)
    fig1(models).to_csv(OUT / "fig1_info.csv", index=False)
    # primary family = the final-vs-base contrasts (meta_contrasts.py); Holm/BH within it
    c = pd.read_csv(config.OUTPUT_DIR / "contrasts_primary.csv")
    c = c[c.preset.isin(FAMILIES)]
    c.assign(corpus=c.corpus.map(CORPORA), family=c.preset.map(FAMILIES))[
        ["corpus", "family", "d_a", "d_b", "diff", "lo", "hi", "p", "p_holm_primary", "p_bh_primary"]].rename(
        columns={"p_holm_primary": "p_holm", "p_bh_primary": "p_bh"}).to_csv(OUT / "fig2_contrasts.csv", index=False)
    fewshot().to_csv(OUT / "fig2_fewshot.csv", index=False)
    stitch().to_csv(OUT / "fig3_stitch.csv", index=False)
    patch().to_csv(OUT / "fig3_patch.csv", index=False)
    print("written:", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
