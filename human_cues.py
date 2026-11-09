"""What drives human judgments beyond the truth? Classical cues and SAE latents.

Builds on human_encoding.py: the heuristic part of a human judgment is
h_res = empirical logit of p("KI") minus its class mean, i.e. what makes humans
call one human (or one AI) text more "KI" than another.

A  Cue table (46 classical cues from the lab course feature extraction): ecological
   validity (point-biserial r with is_ai), utilization overall (r with the
   human judgment), and heuristic utilization (r of the within-class centered cue
   with h_res), with Fisher p-values and BH-FDR over cues.
B  Prediction of h_res, cross-validated by source text, as a share of the noise
   ceiling: cues alone, activations alone, activations + cues. The gain of the
   combination over cues alone is what the model represents beyond the cues.
C  What the activation-based predictor tracks: its out-of-fold predictions
   correlated (within class) with every cue.
D  Sparse-autoencoder latents (Gemma Scope 2, Gemma 3 4B pt, residual stream
   after decoder layer 22 = hidden_states[23], JumpReLU, 16k): latent activations
   at the last prompt token and for the mean over text tokens, correlated
   within class with h_res; latents active on >= 10 % of items; BH-FDR. Latent
   ids can be looked up on Neuronpedia (gemma-3-4b, layer 22, 16k). Note: the
   text-mean code is the SAE applied to the mean residual, an approximation to
   the mean of per-token codes.

No text is read: cue tables are loaded without their text column.

Usage
-----
    python human_cues.py --root ../latent-detection
Output: outputs/human_encoding/{cues_<corpus>.csv, predict_<corpus>.csv,
        tracks_<corpus>.csv, sae_<corpus>.csv}
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

import config
from human_encoding import ALPHAS, ceiling, cv_r, load_items
from judge_logits import condition_stem

SAE_GLOB = str(Path.home() / ".cache/huggingface/hub/models--google--gemma-scope-2-4b-pt/"
               "snapshots/*/resid_post/layer_22_width_16k_l0_big")
SAE_LAYER = 23                      # hidden_states index of model.layers.22.output
META = {"text_id", "source_id", "generator", "gen_model", "model_name", "model_group",
        "prompt_type", "is_ai", "seg_idx", "text", "segment"}


def load_cues(corpus: str, root: Path) -> pd.DataFrame:
    if corpus == "chat":
        f = config.LABCOURSE_DIR / "python" / "output" / "chat" / "features_raw.csv"
        cols = [c for c in pd.read_csv(f, nrows=0).columns if c not in ("text", "segment")]
        c = pd.read_csv(f, usecols=cols)
        c["gen_model"] = np.where(c["generator"] == "human_text", "", c["model_name"])
        return c.drop_duplicates(["source_id", "generator", "gen_model"])
    f = root / "outputs" / corpus / f"features_{corpus}.csv"
    cols = [c for c in pd.read_csv(f, nrows=0).columns if c not in ("text", "segment")]
    return pd.read_csv(f, usecols=cols)


def merge_cues(d: pd.DataFrame, cues: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keys = ["text_id"] if "text_id" in cues else ["source_id", "generator", "gen_model"]
    names = [c for c in cues.columns if c not in META and pd.api.types.is_numeric_dtype(cues[c])]
    m = d.merge(cues[keys + names], on=keys, how="left", indicator=True)
    assert len(m) == len(d) and (m["_merge"] == "both").all(), "cue join failed"
    # coherence cues are undefined for one-sentence texts; drop cues missing for > 20 %
    names = [c for c in names if m[c].isna().mean() <= 0.20]
    m[names] = m[names].fillna(m[names].median())
    return m, names


def within(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Center columns of x within the classes y."""
    x = np.asarray(x, float)
    out = x.copy()
    for c in (0, 1):
        out[y == c] -= x[y == c].mean(0)
    return out


def r_cols(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    Xc, tc = X - X.mean(0), t - t.mean()
    den = np.sqrt((Xc ** 2).sum(0) * (tc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, Xc.T @ tc / den, 0.0)


def p_from_r(r: np.ndarray, n: int) -> np.ndarray:
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(n - 3)
    return 2 * norm.sf(np.abs(z))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def oof_pred(X: np.ndarray, t: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    pred = np.zeros_like(t)
    for tr, te in GroupKFold(n_splits=5).split(X, t, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = np.inf
        m = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, t[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    return pred


def cv_r_nested(X: np.ndarray, t: np.ndarray, groups: np.ndarray) -> float:
    """Grouped CV with the ridge penalty chosen by an inner grouped CV (RidgeCV's
    leave-one-out choice lets texts of the same source inform each other)."""
    from sklearn.linear_model import Ridge
    pred = np.zeros_like(t)
    for tr, te in GroupKFold(n_splits=5).split(X, t, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = np.inf
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        scores = []
        for a in ALPHAS:
            p_in = np.zeros(len(tr))
            for itr, ite in GroupKFold(n_splits=4).split(Xtr, t[tr], groups[tr]):
                p_in[ite] = Ridge(alpha=a).fit(Xtr[itr], t[tr][itr]).predict(Xtr[ite])
            scores.append(np.corrcoef(p_in, t[tr])[0, 1])
        best = ALPHAS[int(np.nanargmax(scores))]
        pred[te] = Ridge(alpha=best).fit(Xtr, t[tr]).predict(Xte)
    return float(np.corrcoef(pred, t)[0, 1])


def sae_encode(X: np.ndarray, sae: dict) -> np.ndarray:
    pre = X @ sae["w_enc"] + sae["b_enc"]
    return np.where(pre > sae["threshold"], pre, 0.0)


def acts_for(root: Path, corpus: str, preset: str, variant: str, d: pd.DataFrame) -> dict | None:
    f = root / "outputs" / corpus / f"{condition_stem(preset, variant, 'raw', 'mk', 'neutral')}_acts.npz"
    if not f.exists():
        return None
    z = np.load(f)
    idx = pd.Series(np.arange(len(z["text_id"])), index=z["text_id"])[d["text_id"]].to_numpy()
    return {s: z[f"acts_{s}"][idx] for s in ("last", "textmean")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=config.REPO)
    ap.add_argument("--corpora", nargs="+", default=["chat", "journalism_seg"])
    ap.add_argument("--presets", nargs="+", default=["gemma3-4b", "gemma3-12b", "qwen3-8b"])
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    out = args.root / "outputs" / "human_encoding"
    out.mkdir(parents=True, exist_ok=True)
    sae = {k: v.astype(np.float64) for k, v in load_file(glob.glob(SAE_GLOB + "/*.safetensors")[0]).items()}
    enc_layers = pd.concat([pd.read_csv(f) for f in out.glob("*_layers.csv")])
    pd.set_option("display.width", 220)

    for corpus in args.corpora:
        d = load_items(corpus, args.root / "data", args.root / "outputs")
        d, names = merge_cues(d, load_cues(corpus, args.root))
        y, g, n = d["is_ai"].to_numpy(), d["source_id"].to_numpy(), len(d)
        e, h = d["elogit"].to_numpy(), d["h_res"].to_numpy()
        ceil_h = ceiling(h, d["samp_var"].to_numpy())
        C = d[names].to_numpy(float)

        # A: cue table
        r_val, r_all, r_heu = r_cols(C, y.astype(float)), r_cols(C, e), r_cols(within(C, y), h)
        p_heu = p_from_r(r_heu, n)
        tab = pd.DataFrame(dict(cue=names, validity=r_val, utilization=r_all,
                                heuristic=r_heu, p_heuristic=p_heu, q_heuristic=bh(p_heu)))
        tab = tab.reindex(tab["heuristic"].abs().sort_values(ascending=False).index)
        tab.to_csv(out / f"cues_{corpus}.csv", index=False)
        print(f"\n=== {corpus}: {n} items; heuristic part noise ceiling {ceil_h:.2f}")
        print("A  cues driving the heuristic part of human judgments (r within class with h_res)")
        print(tab.head(args.top).round(3).to_string(index=False))

        # B + C: prediction of h_res, and what the activation predictor tracks
        Y = y[:, None].astype(float)          # class label as a predictor in every model
        r_cues = cv_r_nested(np.column_stack([C, Y]), h, g) / ceil_h
        r_len = cv_r(np.column_stack([C[:, names.index("word_count")], Y]), h, g, ridge=False) / ceil_h
        rows, tracks = [], []
        for preset in args.presets:
            best = (enc_layers.query("corpus == @corpus and preset == @preset and fmt == 'raw'")
                    .groupby(["variant", "site", "layer"])["r_heuristic"].mean())
            for variant in ("base", "instruct"):
                A = acts_for(args.root, corpus, preset, variant, d)
                if A is None:
                    continue
                n_layers = A["last"].shape[1] - 1
                for site in ("last", "textmean"):
                    best_layer = int(best.loc[(variant, site)].idxmax())
                    for which, layer in (("best (optimistic)", best_layer),
                                         ("67 % depth", round(0.67 * n_layers))):
                        X = A[site][:, layer].astype(np.float64)
                        rows.append(dict(preset=preset, variant=variant, site=site, layer=layer,
                                         choice=which, word_count=r_len, cues=r_cues,
                                         acts=cv_r_nested(np.column_stack([X, Y]), h, g) / ceil_h,
                                         acts_plus_cues=cv_r_nested(np.column_stack([X, C, Y]), h, g) / ceil_h))
                    pred = oof_pred(A[site][:, best_layer].astype(np.float64), h, g)
                    rt = r_cols(within(C, y), pred - pred.mean())
                    tracks.append(pd.Series(rt, index=names, name=f"{preset}_{variant}_{site}_L{best_layer}"))
        pr = pd.DataFrame(rows)
        pr.to_csv(out / f"predict_{corpus}.csv", index=False)
        print("\nB  prediction of the heuristic part (share of noise ceiling), best layer per condition")
        print(pr.round(2).to_string(index=False))
        tr = pd.concat(tracks, axis=1)
        tr.to_csv(out / f"tracks_{corpus}.csv")
        tr["mean_abs"] = tr.abs().mean(1)
        print("\nC  cues most tracked by the activation-based predictors (within-class r, top cues)")
        print(tr.sort_values("mean_abs", ascending=False).head(args.top).drop(columns="mean_abs")
                .round(2).to_string())

        # D: SAE latents, Gemma 3 4B base, layer 22
        A = acts_for(args.root, corpus, "gemma3-4b", "base", d)
        sae_rows = []
        for site in ("last", "textmean"):
            Z = sae_encode(A[site][:, SAE_LAYER].astype(np.float64), sae)
            freq = (Z > 0).mean(0)
            keep = np.where(freq >= 0.10)[0]
            Zk = Z[:, keep]
            r_h = r_cols(within(Zk, y), h)
            r_v = r_cols(Zk, y.astype(float))
            p = p_from_r(r_h, n)
            sae_rows.append(pd.DataFrame(dict(site=site, latent=keep, freq=freq[keep], heuristic=r_h,
                                              validity=r_v, p=p, q=bh(p))))
        sd = pd.concat(sae_rows, ignore_index=True)
        sd = sd.reindex(sd["heuristic"].abs().sort_values(ascending=False).index)
        sd.to_csv(out / f"sae_{corpus}.csv", index=False)
        print(f"\nD  SAE latents (Gemma 3 4B base, layer 22) driving the heuristic part; "
              f"tested: {len(sd)} (last {sum(sd.site == 'last')}, textmean {sum(sd.site == 'textmean')}); "
              f"q < .05: {(sd.q < .05).sum()}")
        print(sd.head(args.top).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
