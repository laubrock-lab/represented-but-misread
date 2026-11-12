"""Truth-judgment replication across post-training stages (representation vs. readout).

For each statement set (outputs/truth_<dataset>/, written by slurm/factuality_stages.sh)
and each checkpoint (preset x variant, raw prompt, both answer orders averaged):

    answers   AUC-based d' (saturates at AUC = 1), equal-variance d_z of the log-odds
              "True" vs "False" (does not saturate), criterion c, p_mass
    probe     logistic probe on the last prompt token at ~67 % depth, out-of-fold
              (GroupKFold by pair), AUC-based d' and d_z
    transfer  the same probe trained on cities (same checkpoint and layer) and applied to
              the other sets: harder, and tests a shared truth direction (Marks & Tegmark)
    contrast  stage minus base for answer d_z and probe d_z, paired grouped bootstrap
              over pairs (same items in both conditions), z-test p

Usage:  python truth_stages.py [--datasets cities neg_cities counterfact] [--n-boot 1000]
Output: outputs/truth_<dataset>/stages.csv, outputs/truth_stages_contrasts.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from sdt_summary import sdt

STAGES = ["base", "sft", "rl", "dpo", "instruct"]


def dz(y: np.ndarray, s: np.ndarray) -> float:
    t, f = s[y == 1], s[y == 0]
    return float((t.mean() - f.mean()) / np.sqrt((t.var(ddof=1) + f.var(ddof=1)) / 2))


def answers(out, preset, variant) -> pd.DataFrame | None:
    fs = [out / f"{preset}_{variant}_raw_{o}.csv" for o in ("tf", "ft")]
    if not all(f.exists() for f in fs):
        return None
    df = pd.concat([pd.read_csv(f) for f in fs])
    n = df.groupby("item_id").size()
    df = df[df.item_id.isin(n[n == 2].index)]
    return df.groupby(["item_id", "pair_id", "is_true"], as_index=False).agg(
        dv=("dv", "mean"), p_mass=("p_mass", "mean"))


def probe_scores(out, preset, variant, items: pd.DataFrame) -> pd.Series | None:
    f = out / f"{preset}_{variant}_raw_tf_acts.npz"
    if not f.exists():
        return None
    z = np.load(f)
    A, ids = z["acts"], z["item_id"]
    layer = round(.67 * (A.shape[1] - 1))
    m = items.set_index("item_id").loc[ids]
    X, y, g = A[:, layer].astype(np.float64), m.is_true.to_numpy(), m.pair_id.to_numpy()
    s = np.zeros(len(y))
    for tr, te in GroupKFold(10).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)).fit(X[tr], y[tr])
        s[te] = clf.decision_function(X[te])
    return pd.Series(s, index=ids, name="probe")


def transfer_scores(preset, variant, items: pd.DataFrame, ds: str) -> pd.Series | None:
    """Probe fitted on all cities statements, applied to the statements of ds."""
    src = config.OUTPUT_DIR / "truth_cities" / f"{preset}_{variant}_raw_tf_acts.npz"
    tgt = config.OUTPUT_DIR / f"truth_{ds}" / f"{preset}_{variant}_raw_tf_acts.npz"
    if ds == "cities" or not (src.exists() and tgt.exists()):
        return None
    zs, zt = np.load(src), np.load(tgt)
    layer = round(.67 * (zs["acts"].shape[1] - 1))
    ys = np.array([1 if i.endswith("T") else 0 for i in zs["item_id"]])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))
    clf.fit(zs["acts"][:, layer].astype(np.float64), ys)
    return pd.Series(clf.decision_function(zt["acts"][:, layer].astype(np.float64)),
                     index=zt["item_id"], name="transfer")


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int, rng) -> tuple[float, float, float]:
    """Paired grouped bootstrap of dz(b) - dz(a); a, b indexed by item_id with pair_id, is_true."""
    m = a[["pair_id", "is_true", col]].join(b[[col]], rsuffix="_b", how="inner")
    idx = pd.Series(np.arange(len(m))).groupby(m.pair_id.to_numpy()).indices
    keys = np.array(list(idx), dtype=object)
    y, sa, sb = m.is_true.to_numpy(), m[col].to_numpy(), m[f"{col}_b"].to_numpy()
    vals = []
    for _ in range(n_boot):
        i = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        vals.append(dz(y[i], sb[i]) - dz(y[i], sa[i]))
    vals = np.array(vals)
    se = vals.std(ddof=1)
    diff = dz(y, sb) - dz(y, sa)
    return diff, se, float(2 * norm.sf(abs(diff) / se))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cities", "neg_cities", "counterfact"])
    ap.add_argument("--presets", nargs="+", default=["olmo2-7b", "apertus15-8b"])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    rows, cons = [], []
    for ds in args.datasets:
        out = config.OUTPUT_DIR / f"truth_{ds}"
        if not out.exists():
            continue
        for preset in args.presets:
            conds = {}
            for v in STAGES:
                a = answers(out, preset, v)
                if a is None:
                    continue
                p = probe_scores(out, preset, v, a)
                a = a.set_index("item_id")
                if p is not None:
                    a = a.join(p)
                t = transfer_scores(preset, v, a, ds)
                if t is not None:
                    a = a.join(t)
                conds[v] = a
                y, s = a.is_true.to_numpy(), a.dv.to_numpy()
                r = dict(dataset=ds, preset=preset, variant=v, n=len(a), ans_d=sdt(y, s)["d_auc"],
                         ans_dz=dz(y, s), ans_c=sdt(y, s)["c0"], share_true=(s > 0).mean(),
                         p_mass=a.p_mass.mean())
                if "probe" in a:
                    r |= dict(probe_d=sdt(y, a.probe.to_numpy())["d_auc"], probe_dz=dz(y, a.probe.to_numpy()))
                if "transfer" in a:
                    r |= dict(transfer_d=sdt(y, a.transfer.to_numpy())["d_auc"], transfer_dz=dz(y, a.transfer.to_numpy()))
                rows.append(r)
            if "base" not in conds:
                continue
            for v, b in conds.items():
                if v == "base":
                    continue
                for col, lab in (("dv", "answer"), ("probe", "probe"), ("transfer", "transfer")):
                    if col in b and col in conds["base"]:
                        d, se, p = boot_diff(conds["base"], b, col, args.n_boot, rng)
                        cons.append(dict(dataset=ds, preset=preset, contrast=f"{v} - base", measure=lab,
                                         diff_dz=d, se=se, lo=d - 1.96 * se, hi=d + 1.96 * se, p=p))
        res = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        if len(res):
            res.to_csv(out / "stages.csv", index=False)
    res, con = pd.DataFrame(rows), pd.DataFrame(cons)
    if len(con):
        con.to_csv(config.OUTPUT_DIR / "truth_stages_contrasts.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(res.to_string(index=False))
        print()
        print(con.to_string(index=False))


if __name__ == "__main__":
    main()
