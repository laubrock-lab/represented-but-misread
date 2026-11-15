"""Linear CKA between the base and a post-trained checkpoint, per layer, at the text positions
(mean over text tokens) and at the last prompt position (response to review, W3/Q7,
16 Nov 2026; the values in Section 3.4 were first computed ad hoc on 10 Nov).

Linear CKA (Kornblith et al., 2019) on the same texts, plain prompt, answer order mk:
CKA(X, Y) = ||Yc' Xc||_F^2 / (||Xc' Xc||_F ||Yc' Yc||_F), computed via n x n Gram matrices.
Usage:  python cka.py [--corpus textbook] [--post sft instruct]
Output: outputs/cka_<corpus>.csv (preset, post, site, layer, depth, cka) + summary print
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config

PRESETS = ["olmo2-7b", "apertus15-8b", "olmo3-7b-think", "gemma3-4b", "gemma3-12b", "qwen3-8b"]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="textbook")
    ap.add_argument("--post", nargs="+", default=["sft", "instruct"])
    a = ap.parse_args()
    d = config.OUTPUT_DIR / a.corpus
    rows = []
    for p in PRESETS:
        fb = d / f"{p}_base_raw_mk_acts.npz"
        if not fb.exists():
            continue
        B = np.load(fb)
        for post in a.post:
            fp = d / f"{p}_{post}_raw_mk_acts.npz"
            if not fp.exists():
                continue
            P = np.load(fp)
            assert (B["text_id"] == P["text_id"]).all()
            n_layers = B["acts_last"].shape[1] - 1
            for site in ("textmean", "last"):
                XB, XP = B[f"acts_{site}"], P[f"acts_{site}"]
                for l in range(1, n_layers + 1):
                    rows.append(dict(preset=p, post=post, site=site, layer=l, depth=l / n_layers,
                                     cka=linear_cka(XB[:, l].astype(np.float64), XP[:, l].astype(np.float64))))
    r = pd.DataFrame(rows)
    r.to_csv(config.OUTPUT_DIR / f"cka_{a.corpus}.csv", index=False)
    s = r.groupby(["preset", "post", "site"]).cka.agg(["min", "median", "max"]).round(3)
    print(s.to_string())


if __name__ == "__main__":
    main()
