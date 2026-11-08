"""Where in the network does a topic-specific criterion shift arise?

Same machinery as for the AI-text task, applied to the true/false statements of
factuality.py (TARGET topic vs. the MATCHED control topic of a statement set).
Two subcommands:

profile  (CPU; uses the activations and judgments from factuality.py)
    Per layer, at the last prompt token, for every model condition:
      truth_dz_<topic>  equal-variance d' of a truth direction fitted on the
                        other controls, applied to TARGET and to MATCHED (both
                        out of sample, same direction)
      shift             midpoint of the TARGET statements minus that of the
                        MATCHED statements along this truth direction, in
                        pooled within-class SD; < 0 = the topic is represented
                        as "more false" (a criterion shift inside the model)
      align_<topic>     alignment as in alignment_profile.py: correlation across
                        dimensions of the truth effect with the within-class
                        decision effect (median split of the model's own dv)
      topic_readout     correlation of the topic effect (TARGET vs. MATCHED,
                        within truth classes) with the readout,
                        i.e. the within-class decision effect on the controls;
                        < 0 = dimensions that mark the topic push the answer to "False"

steer  (GPU)
    Interventions at the last prompt token after decoder layer L, directions
    cross-fitted over statement pairs (no statement steers itself):
      topic   u = mean(TARGET) - mean(MATCHED); "scale 0" moves each statement
              onto the MATCHED level along u (ablation), "scale 2" doubles its
              distance from it; "add -a" moves away from TARGET
      truth   u = mean(true) - mean(false) on the controls; "scale 0" removes the
              truth component (positive control: d' should drop), "scale 3"
              amplifies it
      randomK random unit vectors with the same step sizes (null)
    Step unit: SD of the projections onto the topic direction (common to all
    directions, as in steer_logits.py). Reported per setting x topic: d' (AUC),
    d_dv, c, share of true statements answered "False", mean dv.

Usage:
    python factuality_internals.py profile --statements S.py --out-dir D
    python factuality_internals.py steer --statements S.py --out-dir D --preset qwen3-8b
    python factuality_internals.py steer ... --adapter A --tag <name>   # fine-tuned
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from pathlib import Path
from types import ModuleType

import factuality as fa
from alignment_profile import std_effect, within_class_decision_effect
from judge_logits import pick_device
from sdt_summary import sdt
from steer_logits import Steerer, find_decoder_layers, verify_alignment


def order_avg_dv(out: Path, stem: str) -> pd.Series:
    dfs = [pd.read_csv(out / f"{stem}_{o}.csv") for o in fa.ORDERS]
    return pd.concat(dfs).groupby("item_id")["dv"].mean()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

def profile(st: ModuleType, out: Path) -> pd.DataFrame:
    TARGET, MATCHED = st.TARGET, st.MATCHED
    OTHER = [t for t in st.TOPICS if t not in (TARGET, MATCHED)]
    meta = fa.items(st).set_index("item_id")
    rows = []
    for f in sorted(out.glob("*_tf_acts.npz")):
        cond = f.name[: -len("_tf_acts.npz")]
        preset, variant, fmt = cond.rsplit("_", 2)
        z = np.load(f)
        ids = z["item_id"]
        m = meta.loc[ids]
        topic, y = m["topic"].to_numpy(), m["is_true"].to_numpy()
        dv = order_avg_dv(out, cond).reindex(ids).to_numpy()
        tgt, mat, oth = topic == TARGET, topic == MATCHED, np.isin(topic, OTHER)
        ctrl = ~tgt
        n_layers = z["acts"].shape[1] - 1
        for layer in range(1, n_layers + 1):
            X = z["acts"][:, layer].astype(np.float64)
            mu, sd = X[oth].mean(0), X[oth].std(0)
            sd[sd < 1e-6 * max(sd.max(), 1e-12)] = np.inf
            Xs = (X - mu) / sd
            w = Xs[oth & (y == 1)].mean(0) - Xs[oth & (y == 0)].mean(0)
            s = Xs @ w
            mid = {k: (s[msk & (y == 1)].mean() + s[msk & (y == 0)].mean()) / 2
                   for k, msk in ((TARGET, tgt), (MATCHED, mat))}
            both = tgt | mat
            psd = np.sqrt(np.mean([s[both & (y == c)].var(ddof=1) for c in (0, 1)]))
            topic_eff = np.mean([std_effect(X[both & (y == c)], tgt[both & (y == c)])
                                 for c in (0, 1)], axis=0)
            readout = within_class_decision_effect(X[ctrl], y[ctrl], dv[ctrl])
            rows.append(dict(
                preset=preset, variant=variant, fmt=fmt, layer=layer, depth=layer / n_layers,
                truth_dz_target=fa.eq_var_d(s[tgt], y[tgt]),
                truth_dz_matched=fa.eq_var_d(s[mat], y[mat]),
                shift=(mid[TARGET] - mid[MATCHED]) / psd,
                align_target=corr(std_effect(X[tgt], y[tgt] == 1),
                                     within_class_decision_effect(X[tgt], y[tgt], dv[tgt])),
                align_matched=corr(std_effect(X[mat], y[mat] == 1),
                                 within_class_decision_effect(X[mat], y[mat], dv[mat])),
                topic_readout=corr(topic_eff, readout)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "internals_profile.csv", index=False)
    df["band"] = pd.cut(df["depth"], [0, .2, .4, .6, .8, 1.0],
                        labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    cols = ["truth_dz_target", "truth_dz_matched", "shift", "align_target", "align_matched",
            "topic_readout"]
    pd.set_option("display.width", 200)
    print(df.groupby(["preset", "variant", "fmt", "band"], observed=True)[cols]
            .mean().round(2).to_string())
    return df


# ---------------------------------------------------------------------------
# steer
# ---------------------------------------------------------------------------

def directions(X: np.ndarray, topic: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_random: int, seed: int, TARGET: str, MATCHED: str) -> dict[str, list[dict]]:
    """Cross-fitted (GroupKFold over pairs) topic, truth and random directions.
    'mid' is the reference point that scale mode pulls towards / pushes from."""
    rng = np.random.default_rng(seed)
    names = ["topic", "truth"] + [f"random{k + 1}" for k in range(n_random)]
    out = {n: [None] * len(y) for n in names}
    for tr, te in GroupKFold(n_splits=4).split(X, y, groups):
        Xt, tt, yt = X[tr], topic[tr], y[tr]
        a, b = Xt[tt == TARGET].mean(0), Xt[tt == MATCHED].mean(0)
        u_top = (a - b) / np.linalg.norm(a - b)
        p = Xt @ u_top
        unit = float(np.sqrt((p[tt == TARGET].var() + p[tt == MATCHED].var()) / 2))
        dirs = {"topic": (u_top, float(b @ u_top))}           # reference: MATCHED level
        c = tt != TARGET
        t1, t0 = Xt[c & (yt == 1)].mean(0), Xt[c & (yt == 0)].mean(0)
        u_tr = (t1 - t0) / np.linalg.norm(t1 - t0)
        dirs["truth"] = (u_tr, float(((t1 + t0) / 2) @ u_tr))  # reference: truth midpoint
        for k in range(n_random):
            r = rng.standard_normal(X.shape[1])
            r /= np.linalg.norm(r)
            dirs[f"random{k + 1}"] = (r, float(Xt.mean(0) @ r))
        for n, (u, ref) in dirs.items():
            d = dict(u=torch.tensor(u, dtype=torch.float32), delta=unit, mid=ref,
                     cos_topic=float(u @ u_top))
            for i in te:
                out[n][i] = d
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["layer", "direction", "mode", "alpha"]
    avg = df.groupby(keys + ["item_id", "topic", "is_true"], as_index=False)["dv"].mean()
    rows = []
    for key, g in avg.groupby(keys + ["topic"]):
        t, f = g.loc[g["is_true"] == 1, "dv"], g.loc[g["is_true"] == 0, "dv"]
        s = sdt(g["is_true"].to_numpy(), g["dv"].to_numpy())
        rows.append(dict(zip(keys + ["topic"], key), out_d=s["d_auc"], c=s["c0"],
                         d_dv=(t.mean() - f.mean()) / np.sqrt((t.var() + f.var()) / 2),
                         true_called_false=(t < 0).mean(), mean_dv=g["dv"].mean()))
    return pd.DataFrame(rows)


def steer(args, st: ModuleType, out: Path) -> None:
    TARGET, MATCHED = st.TARGET, st.MATCHED
    label = args.tag or args.variant
    stem = f"{args.preset}_{label}_{args.fmt}"
    meta = fa.items(st).set_index("item_id")
    z = np.load(out / f"{stem}_tf_acts.npz")
    ids = z["item_id"]
    m = meta.loc[ids].reset_index()
    topic, y, groups = m["topic"].to_numpy(), m["is_true"].to_numpy(), m["pair_id"].to_numpy()
    n_layers = z["acts"].shape[1] - 1
    layers_req = args.layers or sorted({round(f * n_layers) for f in (0.33, 0.5, 0.67, 0.83)})

    device = pick_device(args.device)
    variant = "instruct" if args.adapter else args.variant
    tok, model = fa.load_condition(args.preset, variant, device, args.adapter)
    dec = find_decoder_layers(model)
    cands, eow = fa.candidate_seqs(tok), {}
    prompts = {o: [fa.build_prompt(tok, s, args.fmt, o) for s in m["statement"]] for o in fa.ORDERS}

    rows = []
    for layer in layers_req:
        verify_alignment(model, tok, dec, layer, device)
        hook = Steerer(dec[layer - 1])          # hidden_states[L] = output of layer L-1
        dirs = directions(z["acts"][:, layer].astype(np.float64), topic, y, groups,
                          args.n_random, args.seed, TARGET, MATCHED)
        settings = ([("none", "add", 0.0),
                     ("topic", "scale", 0.0), ("topic", "scale", 2.0),
                     ("topic", "add", -3.0), ("topic", "add", -6.0),
                     ("truth", "scale", 0.0), ("truth", "scale", 3.0)] +
                    [(f"random{k + 1}", mode, a) for k in range(args.n_random)
                     for mode, a in (("scale", 0.0), ("add", -6.0))])
        t0 = time.time()
        for order in fa.ORDERS:
            for i, r in enumerate(m.itertuples(index=False)):
                prompt, add = prompts[order][i]
                n = len(tok(prompt, add_special_tokens=add)["input_ids"])
                hook.pos = slice(n - 1, n)
                for direction, mode, a in settings:
                    d = dirs["topic" if direction == "none" else direction][i]
                    hook.u, hook.delta, hook.mid = d["u"], d["delta"], d["mid"]
                    hook.mode, hook.alpha = mode, float(a)
                    res = fa.score(model, tok, prompt, add, cands, eow, device, False)
                    rows.append(dict(layer=layer, direction=direction, mode=mode, alpha=a,
                                     order=order, item_id=r.item_id, topic=r.topic,
                                     is_true=r.is_true, dv=res["dv"], p_mass=res["p_mass"]))
        hook.remove()
        print(f"{stem} L{layer}: {len(settings)} settings x {len(m)} statements x 2 orders "
              f"in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"steer_{stem}.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / f"steer_summary_{stem}.csv", index=False)
    pd.set_option("display.width", 200)
    show = summ[summ["topic"].isin([TARGET, MATCHED])]
    print(show.pivot_table(index=["layer", "direction", "mode", "alpha"], columns="topic",
                           values=["true_called_false", "d_dv", "mean_dv"]).round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("profile")
    st = sub.add_parser("steer")
    for p in (pr, st):
        p.add_argument("--statements", type=Path, required=True)
        p.add_argument("--out-dir", type=Path, required=True)
    st.add_argument("--adapter", type=Path, help="LoRA adapter applied to the instruct weights")
    st.add_argument("--tag", help="variant label of the adapter condition (as in factuality.py)")
    st.add_argument("--preset", required=True, choices=list(fa.config.MODEL_PRESETS))
    st.add_argument("--variant", default="instruct", choices=["base", "instruct"])
    st.add_argument("--fmt", default="raw", choices=["raw", "chat"])
    st.add_argument("--layers", type=int, nargs="+",
                    help="hidden_states indices (default: 33/50/67/83 %% depth)")
    st.add_argument("--n-random", type=int, default=3)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--device", default="auto")
    args = ap.parse_args()
    stmts = fa.load_statements(args.statements)
    if args.cmd == "profile":
        profile(stmts, args.out_dir)
    else:
        steer(args, stmts, args.out_dir)


if __name__ == "__main__":
    main()
