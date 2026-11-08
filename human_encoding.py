"""Do the models' activations predict human judgments, and does the instruct
model's readout follow the *errors* humans make?

Human judgments per item (data/human_items_<corpus>.csv, from
export_human_items.R): share of "KI" answers p_ki over n participants, as an
empirical logit. The human judgment splits into a valid part (the class mean)
and a heuristic part h_res = judgment minus its class mean, i.e. what makes
humans call one human text more "KI" than another.

1. Behaviour, per model condition (no activations): Spearman correlation of
   the model's dv with the human judgment, overall and within class. A
   within-class correlation > 0 means the model errs on the same items.
2. Encoding, per condition x site x layer: ridge regression from activations to
   the human judgment and to h_res, cross-validated by source text, as a share of
   the noise ceiling (sqrt of the reliability implied by binomial sampling,
   1 - mean sampling variance / observed variance). Baselines: class + length,
   and length alone for h_res.
3. Mimicry, per layer: correlation across dimensions between the model's
   within-class decision effect (alignment_profile.py) and the human heuristic
   effect (per-dimension correlation with h_res, within class). > 0 = the
   dimensions along which the model's answer varies are those along which
   human judgments vary beyond the truth. Permutation null at selected depths:
   h_res shuffled within class.
   Prediction of the mimicry hypothesis: higher for instruct than for base.

Usage
-----
    python human_encoding.py                          # chat and journalism_seg
    python human_encoding.py --root ../latent-detection --corpora journalism_seg
Output: outputs/human_encoding/{behaviour.csv, <corpus>_layers.csv, mimicry_perm.csv}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import config
from alignment_profile import std_effect, within_class_decision_effect
from judge_logits import condition_stem

CONDS = [("base", "raw"), ("instruct", "raw"), ("instruct", "chat")]
ALPHAS = np.logspace(0, 7, 15)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_items(corpus: str, data: Path, out: Path) -> pd.DataFrame:
    cols = ["text_id", "source_id", "generator", "gen_model", "prompt_type", "is_ai"]
    texts = pd.read_parquet(data / f"texts_{corpus}.parquet", columns=cols)   # no text column
    hum = pd.read_csv(data / f"human_items_{corpus}.csv")
    if "text_id" in hum:
        d = texts.merge(hum[["text_id", "n", "p_ki"]], on="text_id")
    else:
        for t in (texts, hum):
            t["gen_model"] = t["gen_model"].fillna("")
        d = texts.merge(hum[["source_id", "generator", "gen_model", "n", "p_ki"]],
                        on=["source_id", "generator", "gen_model"])
    k = np.round(d["p_ki"] * d["n"])
    d["elogit"] = np.log((k + 0.5) / (d["n"] - k + 0.5))
    d["samp_var"] = 1 / (k + 0.5) + 1 / (d["n"] - k + 0.5)
    d["h_res"] = d["elogit"] - d.groupby("is_ai")["elogit"].transform("mean")
    any_csv = sorted(out.joinpath(corpus).glob("*_base_raw_mk.csv"))[0]
    length = pd.read_csv(any_csv, usecols=["text_id", "n_text_tokens"]).set_index("text_id")
    d["length"] = np.log(d["text_id"].map(length["n_text_tokens"]))
    return d.reset_index(drop=True)


def ceiling(y: np.ndarray, samp_var: np.ndarray) -> float:
    return float(np.sqrt(max(1 - samp_var.mean() / y.var(ddof=1), 0)))


def model_dv(out: Path, corpus: str, preset: str, variant: str, fmt: str) -> pd.Series | None:
    fs = [out / corpus / f"{condition_stem(preset, variant, fmt, o, 'neutral')}.csv" for o in ("mk", "km")]
    if not all(f.exists() for f in fs):
        return None
    return pd.concat(pd.read_csv(f, usecols=["text_id", "dv"]) for f in fs).groupby("text_id")["dv"].mean()


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def within_spearman(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    rs, ws = [], []
    for c in (0, 1):
        m = y == c
        rs.append(spearmanr(a[m], b[m])[0])
        ws.append(m.sum())
    return float(np.average(rs, weights=ws))


def cv_r(X: np.ndarray, target: np.ndarray, groups: np.ndarray, ridge: bool = True) -> float:
    pred = np.zeros_like(target)
    for tr, te in GroupKFold(n_splits=5).split(X, target, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = np.inf
        model = RidgeCV(alphas=ALPHAS) if ridge else LinearRegression()
        model.fit((X[tr] - mu) / sd, target[tr])
        pred[te] = model.predict((X[te] - mu) / sd)
    return float(np.corrcoef(pred, target)[0, 1])


def human_effect(X: np.ndarray, y: np.ndarray, h_res: np.ndarray) -> np.ndarray:
    """Per-dimension correlation with h_res, with X centered within class."""
    Xc = X - np.where(y[:, None] == 1, X[y == 1].mean(0), X[y == 0].mean(0))
    hc = h_res - h_res.mean()
    sd = Xc.std(0) * hc.std()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, (Xc * hc[:, None]).mean(0) / sd, 0.0)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=config.REPO,
                    help="repo folder whose data/ and outputs/ are used")
    ap.add_argument("--corpora", nargs="+", default=["chat", "journalism_seg"])
    ap.add_argument("--presets", nargs="+", default=["gemma3-4b", "gemma3-12b", "qwen3-8b"])
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    data, out = args.root / "data", args.root / "outputs"
    res_dir = out / "human_encoding"
    res_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    beh, perm_rows = [], []
    for corpus in args.corpora:
        d = load_items(corpus, data, out)
        y, g = d["is_ai"].to_numpy(), d["source_id"].to_numpy()
        e, h = d["elogit"].to_numpy(), d["h_res"].to_numpy()
        ceil_e, ceil_h = ceiling(e, d["samp_var"].to_numpy()), ceiling(h, d["samp_var"].to_numpy())
        L = d[["length"]].to_numpy()
        base_e = cv_r(np.column_stack([y, L]), e, g, ridge=False)
        base_h = cv_r(L, h, g, ridge=False)
        print(f"\n=== {corpus}: {len(d)} items | human AUC {roc_auc_score(y, e):.2f} | noise ceiling "
              f"judgment {ceil_e:.2f}, heuristic part {ceil_h:.2f} | baseline r (class+length) "
              f"{base_e:.2f}, heuristic ~ length {base_h:.2f}")
        rows = []
        for preset in args.presets:
            for variant, fmt in CONDS:
                dv = model_dv(out, corpus, preset, variant, fmt)
                f = out / corpus / f"{condition_stem(preset, variant, fmt, 'mk', 'neutral')}_acts.npz"
                if dv is None or not f.exists():
                    continue
                mdv = d["text_id"].map(dv).to_numpy()
                beh.append(dict(corpus=corpus, preset=preset, variant=variant, fmt=fmt,
                                model_auc=roc_auc_score(y, mdv),
                                rho_human=spearmanr(mdv, e)[0],
                                rho_human_within=within_spearman(mdv, e, y)))
                z = np.load(f)
                idx = pd.Series(np.arange(len(z["text_id"])), index=z["text_id"])[d["text_id"]].to_numpy()
                n_layers = z["acts_last"].shape[1] - 1
                perm_layers = {round(q * n_layers) for q in (0.5, 0.67, 0.83)}
                for site in ("last", "textmean"):
                    A = z[f"acts_{site}"][idx]
                    for layer in range(1, n_layers + 1):
                        if layer % args.layer_step and layer not in perm_layers:
                            continue
                        X = A[:, layer].astype(np.float64)
                        m_eff = within_class_decision_effect(X, y, mdv)
                        h_eff = human_effect(X, y, h)
                        row = dict(corpus=corpus, preset=preset, variant=variant, fmt=fmt,
                                   site=site, layer=layer, depth=layer / n_layers,
                                   r_judgment=cv_r(X, e, g) / ceil_e,
                                   r_heuristic=cv_r(X, h, g) / ceil_h,
                                   mimicry=corr(m_eff, h_eff),
                                   human_valid=corr(std_effect(X, y == 1), human_effect(X, y, e - e.mean())))
                        rows.append(row)
                        if layer in perm_layers and site == "last":
                            null = []
                            for _ in range(args.n_perm):
                                hp = h.copy()
                                for c in (0, 1):
                                    m = y == c
                                    hp[m] = rng.permutation(hp[m])
                                null.append(corr(m_eff, human_effect(X, y, hp)))
                            null = np.array(null)
                            perm_rows.append(dict(row, null_mean=null.mean(), null_sd=null.std(),
                                                  p=(1 + (np.abs(null) >= abs(row["mimicry"])).sum())
                                                    / (1 + len(null))))
                print(f"  {preset} {variant} {fmt}: done")
        df = pd.DataFrame(rows)
        df.to_csv(res_dir / f"{corpus}_layers.csv", index=False)
        df["band"] = pd.cut(df["depth"], [0, .2, .4, .6, .8, 1.0],
                            labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
        pd.set_option("display.width", 200)
        print(df[df.site == "last"].groupby(["preset", "variant", "fmt", "band"], observed=True)
                [["r_judgment", "r_heuristic", "mimicry"]].mean().round(2).unstack("band").to_string())

    b = pd.DataFrame(beh)
    b.to_csv(res_dir / "behaviour.csv", index=False)
    print("\n=== behaviour: does the model err on the same items as humans?")
    print(b.round(2).to_string(index=False))
    p = pd.DataFrame(perm_rows)
    p.to_csv(res_dir / "mimicry_perm.csv", index=False)
    print("\n=== mimicry at 50/67/83 % depth (last token), permutation null")
    print(p[["corpus", "preset", "variant", "fmt", "layer", "mimicry", "null_mean", "null_sd", "p"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
