"""How much does it matter that the probe pipeline fits standardization and the principal
component rotation on the whole outer training fold before the inner cross-validation selects
the regularization strength (probe_layers._make_probe)?

Compares, on the short chats (the corpus on which the probes are not at ceiling), the
out-of-fold d′ of the pipeline as used in the paper with a strictly nested version, in which
the whole pipeline (imputation, standardization, rotation, logistic regression) is refitted
within each inner training split (GridSearchCV over the same values of C, same inner folds,
same scoring). Both use the same outer folds. Last prompt position, layer at two thirds of
the depth, base and final checkpoint of every model family with local activations.

Output: outputs/probe_inner_cv_check.csv
Usage:  python probe_inner_cv_check.py [--corpus chat_tw25]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from probe_layers import CS, _fit_probe
from sdt_summary import sdt

PRESETS = ["gemma3-4b", "gemma3-12b", "qwen3-8b", "olmo2-7b", "apertus15-8b", "olmo3-7b-think"]


def nested_probe(groups_tr, X_tr, y_tr):
    n_inner = min(5, len(np.unique(groups_tr)))
    inner = list(GroupKFold(n_splits=n_inner).split(X_tr, y_tr, groups_tr))
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         PCA(n_components=None, svd_solver="full"),
                         LogisticRegression(class_weight="balanced", max_iter=5000))
    gs = GridSearchCV(pipe, {"logisticregression__C": list(CS)}, cv=inner, scoring="roc_auc", refit=True)
    return gs.fit(X_tr, y_tr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="chat_tw25")
    a = ap.parse_args()
    rows = []
    for p in PRESETS:
        for v in ("base", "instruct"):
            f = config.OUTPUT_DIR / a.corpus / f"{p}_{v}_raw_mk_acts.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            y, g = z["is_ai"].astype(int), z["source_id"].astype(str)
            L = z["acts_last"].shape[1] - 1
            layer = int(round(2 * L / 3))
            X = z["acts_last"][:, layer, :].astype(np.float32)
            oof_a, oof_b, c_a, c_b = np.full(len(y), np.nan), np.full(len(y), np.nan), [], []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", FutureWarning)
                for tr, te in GroupKFold(n_splits=min(10, len(np.unique(g)))).split(X, y, g):
                    pa = _fit_probe(g[tr], X[tr], y[tr], True)
                    oof_a[te] = pa.decision_function(X[te]); c_a.append(float(pa[-1].C_[0]))
                    pb = nested_probe(g[tr], X[tr], y[tr])
                    oof_b[te] = pb.decision_function(X[te]); c_b.append(float(pb.best_params_["logisticregression__C"]))
            rows.append(dict(corpus=a.corpus, preset=p, variant=v, layer=layer, n=len(y),
                             d_paper=sdt(y, oof_a)["d_auc"], d_nested=sdt(y, oof_b)["d_auc"],
                             r_scores=float(np.corrcoef(oof_a, oof_b)[0, 1]),
                             same_C_folds=int(np.sum(np.isclose(c_a, c_b))), folds=len(c_a),
                             median_C_paper=float(np.median(c_a)), median_C_nested=float(np.median(c_b))))
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d["diff"] = d.d_nested - d.d_paper
    d.to_csv(config.OUTPUT_DIR / "probe_inner_cv_check.csv", index=False)
    print(d.round(3).to_string(index=False))
    print("max |diff|:", d["diff"].abs().max().round(3), "mean diff:", d["diff"].mean().round(3))


if __name__ == "__main__":
    main()
