"""Gain x angle decomposition of the readout.

Inputs per condition (outputs/<corpus>/): grad_<stem>.npz from gradient_readout.py
(d dv / d h_l at the last prompt token) and <stem>_acts.npz from judge_logits.py
(h_l at the last prompt token). Per layer l:

    G        mean over texts of g_i . dmu_(-i)   first-order class effect on the answer
             (nats for a shift by the AI-human mean difference; dmu cross-fitted by
             source pair, so a text's own activation never enters its class direction)
    gain     ||mean g||                          sensitivity of the answer
    dmu      ||dmu||                             strength of the class difference
    cos      cos(mean g, dmu)                    angle (sign = correct / reversed use)
    cos_std  same after scaling every dimension by its within-class SD (keeps
             massive-activation dimensions from dominating)
    z_pc, z_split
             angle relative to matched nulls: cos(mean g, u) for random unit vectors u
             in the top-20 PC subspace of the within-class-centred activations, and for
             mean differences of random within-class halves (n_null each)
    rot      (stage presets) cos between this condition's mean gradient and the base
             model's at the same layer (base is raw-format only; marked when formats differ)

Note: G is the linearised output separation; its agreement with output d' is
expected and is NOT evidence. The point is the decomposition (gain vs. angle),
the layer profile, and the change across post-training stages.

Usage
-----
    python readout_decomposition.py --corpora textbook
Output: outputs/readout_decomposition_<corpus>.csv
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config

STAGE_ORDER = {"base": 0, "sft": 1, "rl": 2, "dpo": 3, "instruct": 4}
PAT = re.compile(r"^grad_(?P<preset>.+?)_(?P<variant>base|sft|rl|dpo|instruct)_(?P<fmt>raw|chat)_mk(?:_neutral)?\.npz$")


def cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def analyse(gz, az, n_null: int, rng) -> tuple[list[dict], np.ndarray]:
    ids = pd.Index(az["text_id"].astype(str))
    gi = pd.Index(gz["text_id"].astype(str))
    common = gi.intersection(ids)
    G_all = gz["grad"][gi.get_indexer(common)].astype(np.float64)          # [n, L+1, d]
    A_all = az["acts_last"][ids.get_indexer(common)].astype(np.float64)    # [n, L+1, d]
    y = az["is_ai"][ids.get_indexer(common)].astype(int)
    src = az["source_id"][ids.get_indexer(common)].astype(str)
    folds = list(GroupKFold(n_splits=min(10, len(np.unique(src)))).split(A_all[:, 0], y, src))
    rows, gbar = [], G_all.mean(0)
    for layer in range(1, A_all.shape[1]):
        X, g = A_all[:, layer], G_all[:, layer]
        Gi = np.zeros(len(y))
        for tr, te in folds:
            dmu_tr = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
            Gi[te] = g[te] @ dmu_tr
        mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
        dmu = mu1 - mu0
        Xc = X - np.where((y == 1)[:, None], mu1, mu0)
        sd = Xc.std(0) + 1e-8
        gb = gbar[layer]
        # matched nulls
        _, _, Vt = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
        V = Vt[:min(20, Vt.shape[0])]
        c_pc = [cos(gb, rng.standard_normal(len(V)) @ V) for _ in range(n_null)]
        c_split = []
        for _ in range(n_null):
            half = np.zeros(len(y), bool)
            for c in (0, 1):
                idx = np.where(y == c)[0]
                half[rng.permutation(idx)[: len(idx) // 2]] = True
            c_split.append(cos(gb, X[half].mean(0) - X[~half].mean(0)))
        c = cos(gb, dmu)
        rows.append(dict(layer=layer, n_layers=A_all.shape[1] - 1, depth=layer / (A_all.shape[1] - 1),
                         G=Gi.mean(), G_se=Gi.std(ddof=1) / np.sqrt(len(Gi)),
                         gain=np.linalg.norm(gb), mean_grad_norm=np.linalg.norm(g, axis=1).mean(),
                         dmu=np.linalg.norm(dmu), cos=c, cos_std=cos(gb * sd, dmu / sd),
                         z_pc=(c - np.mean(c_pc)) / (np.std(c_pc) + 1e-12),
                         z_split=(c - np.mean(c_split)) / (np.std(c_split) + 1e-12)))
    return rows, gbar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", default=["textbook"])
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for corpus in args.corpora:
        d = config.OUTPUT_DIR / corpus
        conds = []
        for f in sorted(d.glob("grad_*.npz")):
            m = PAT.match(f.name)
            if m and "_smoke" not in f.name:
                conds.append(m.groupdict() | {"file": f})
        rows, gbars = [], {}
        for c in sorted(conds, key=lambda c: (c["preset"], STAGE_ORDER[c["variant"]], c["fmt"])):
            stem = c["file"].name.removeprefix("grad_").removesuffix(".npz")
            acts = d / f"{stem}_acts.npz"
            if not acts.exists():
                print(f"skip {stem}: no activations")
                continue
            r, gbar = analyse(np.load(c["file"]), np.load(acts), args.n_null, rng)
            gbars[(c["preset"], c["variant"], c["fmt"])] = gbar
            base = gbars.get((c["preset"], "base", "raw"))
            for row in r:
                row |= dict(corpus=corpus, preset=c["preset"], variant=c["variant"], fmt=c["fmt"],
                            rot=(cos(gbar[row["layer"]], base[row["layer"]]) if base is not None else np.nan),
                            rot_cross_format=c["fmt"] != "raw")
            rows += r
            up = pd.DataFrame(r)
            up = up[up.depth >= .5]
            print(f"{corpus:10s} {stem:42s} upper half: G {up.G.mean():+.3f}  gain {up.gain.mean():.3f}  "
                  f"cos {up.cos.mean():+.3f}  cos_std {up.cos_std.mean():+.3f}  z_split {up.z_split.mean():+.1f}")
        out = config.OUTPUT_DIR / f"readout_decomposition_{corpus}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
