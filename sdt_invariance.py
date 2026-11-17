"""SDT invariance test (17 Nov 2026): do bias manipulations move the criterion c but leave the
sensitivity d′ unchanged, as signal detection theory assumes?

Bias manipulations (plain prompt):
  order  answer order in the question ("Mensch oder KI" = mk vs. "KI oder Mensch" = km)
  prior  a sentence stating the share of AI texts in the study (20, 50, or 80%;
         judge_logits.py --priors, wording "ai" = AI share mentioned first), vs. no sentence

Every single prompt condition is a legitimate SDT condition with its own criterion, so d′ and c
are computed per condition (dv of that prompt only, not averaged over orders): d′ = √2 z(AUC),
c at dv = 0 with the log-linear correction (sdt_summary.sdt). A condition is "saturated" if the
model gives the same binary answer to ≥ 98% of the texts (c then sits near its bound and is
insensitive to further shifts).

Tests per corpus × model × checkpoint (grouped bootstrap over source groups, same resamples
for all conditions):
  order effect   Δ = km − mk at prior none, for d′ and c
  sentence       Δ = p50 − none (mentioning a neutral 50% base rate), per order
  base rate      slope of d′ and of c on the stated AI share (20 → 80%, per 10 points), per order;
                 SDT predicts slope(c) < 0 (more "KI" answers) and slope(d′) = 0
Summary across cells: how often the CI excludes zero for c and for d′, and mean |Δ|.

Usage:  python sdt_invariance.py   -> outputs/sdt_invariance_conditions.csv, sdt_invariance_tests.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from sdt_summary import sdt

PRESETS = ["gemma3-4b", "gemma3-12b", "qwen3-8b", "olmo2-7b", "apertus15-8b", "olmo3-7b-think"]
CORPORA = ["chat_tw25", "journalism_seg", "textbook"]
PRIORS = {"none": "", "20": "_p20ai", "50": "_p50ai", "80": "_p80ai"}
N_BOOT = 1000
TOL = 1e-9  # bounds such as -1e-16 arise when c is identical in all conditions (constant answer)


def load(corpus, preset, variant):
    d = config.OUTPUT_DIR / corpus
    out = {}
    for order in ("mk", "km"):
        for pr, sfx in PRIORS.items():
            f = d / f"{preset}_{variant}_raw_{order}{sfx}.csv"
            if f.exists():
                x = pd.read_csv(f, usecols=["text_id", "source_id", "is_ai", "dv", "p_mass"])
                out[(order, pr)] = x.drop_duplicates("text_id").set_index("text_id")
    return out


def stats(y, s):
    r = sdt(y, s)
    return r["d_auc"], r["c0"]


def main() -> None:
    rng = np.random.default_rng(1)
    cond_rows, test_rows = [], []
    for corpus in CORPORA:
        for preset in PRESETS:
            for variant in ("base", "instruct"):
                C = load(corpus, preset, variant)
                if len(C) < 2:
                    continue
                ids = None
                for v in C.values():
                    ids = v.index if ids is None else ids.intersection(v.index)
                ref = next(iter(C.values())).loc[ids]
                y = ref.is_ai.to_numpy().astype(int)
                S = {k: v.loc[ids, "dv"].to_numpy() for k, v in C.items()}
                for k, v in C.items():
                    d, c = stats(y, S[k])
                    share = float((S[k] > 0).mean())
                    cond_rows.append(dict(corpus=corpus, preset=preset, variant=variant, order=k[0], prior=k[1],
                                          n=len(ids), d=d, c=c, share_ki=share,
                                          saturated=share <= .02 or share >= .98,
                                          p_mass=v.loc[ids, "p_mass"].mean()))
                idx = pd.Series(np.arange(len(ids))).groupby(ref.source_id.to_numpy()).indices
                keys = np.array(list(idx), dtype=object)
                boots = [np.concatenate([idx[k] for k in rng.choice(keys, len(keys))]) for _ in range(N_BOOT)]
                boots = [b for b in boots if y[b].min() != y[b].max()]
                # contrasts: (name, function of a stats-getter) -> value
                def contrasts(get):
                    out = {}
                    if ("mk", "none") in C and ("km", "none") in C:
                        for j, m in enumerate(("d", "c")):
                            out[("order km-mk", m)] = get(("km", "none"))[j] - get(("mk", "none"))[j]
                    for o in ("mk", "km"):
                        if (o, "none") in C and (o, "50") in C:
                            for j, m in enumerate(("d", "c")):
                                out[(f"sentence p50-none ({o})", m)] = get((o, "50"))[j] - get((o, "none"))[j]
                        if all((o, p) in C for p in ("20", "50", "80")):
                            for j, m in enumerate(("d", "c")):
                                vals = [get((o, p))[j] for p in ("20", "50", "80")]
                                out[(f"base rate slope per 10 pts ({o})", m)] = np.polyfit([2, 5, 8], vals, 1)[0]
                    return out
                point = contrasts(lambda k: stats(y, S[k]))
                bs = {key: [] for key in point}
                for b in boots:
                    cache = {}
                    def get(k, b=b):
                        if k not in cache:
                            cache[k] = stats(y[b], S[k][b])
                        return cache[k]
                    for key, val in contrasts(get).items():
                        bs[key].append(val)
                for (name, m), val in point.items():
                    arr = np.array(bs[(name, m)])
                    conds = [k for k in C if (name.startswith("order") and k[1] == "none") or
                             (not name.startswith("order") and name.endswith(f"({k[0]})"))]
                    test_rows.append(dict(corpus=corpus, preset=preset, variant=variant, contrast=name, measure=m,
                                          est=val, lo=np.quantile(arr, .025), hi=np.quantile(arr, .975),
                                          excl0=bool(np.quantile(arr, .025) > TOL or np.quantile(arr, .975) < -TOL),
                                          moved=bool(abs(val) > TOL),
                                          any_saturated=any(((S[k] > 0).mean() <= .02 or (S[k] > 0).mean() >= .98)
                                                            for k in conds)))
    cond = pd.DataFrame(cond_rows)
    tests = pd.DataFrame(test_rows)
    cond.to_csv(config.OUTPUT_DIR / "sdt_invariance_conditions.csv", index=False)
    tests.to_csv(config.OUTPUT_DIR / "sdt_invariance_tests.csv", index=False)
    kind = tests.contrast.str.replace(r" \((mk|km)\)", "", regex=True)
    summ = tests.assign(kind=kind).groupby(["kind", "measure"]).agg(
        cells=("est", "size"), mean_abs=("est", lambda v: np.abs(v).mean()), mean=("est", "mean"),
        n_excl0=("excl0", "sum"), n_moved=("moved", "sum"))
    summ_ns = tests[~tests.any_saturated].assign(kind=kind[~tests.any_saturated]).groupby(["kind", "measure"]).agg(
        cells=("est", "size"), mean_abs=("est", lambda v: np.abs(v).mean()), mean=("est", "mean"),
        n_excl0=("excl0", "sum"), n_moved=("moved", "sum"))
    # appendix table: mean absolute full-range effect per model x checkpoint (over corpora and
    # answer orders); base-rate slope scaled to the 20 -> 80% range (x 6)
    t = tests.assign(kind=kind)
    t["eff"] = np.where(t.kind.str.startswith("base rate"), t.est * 6, t.est)
    t["kind"] = t.kind.map({"base rate slope per 10 pts": "Base rate 20→80%", "order km-mk": "Answer order",
                            "sentence p50-none": "Base-rate sentence"})
    tab = t.pivot_table(index=["preset", "variant"], columns=["kind", "measure"], values="eff",
                        aggfunc=lambda v: np.abs(v).mean())
    names = {"gemma3-4b": "Gemma 3 4B", "gemma3-12b": "Gemma 3 12B", "qwen3-8b": "Qwen3 8B",
             "olmo2-7b": "OLMo 2 7B", "apertus15-8b": "Apertus 1.5 8B", "olmo3-7b-think": "OLMo 3 7B Think"}
    kinds = ["Base rate 20→80%", "Answer order", "Base-rate sentence"]
    lines = ["| Model | Checkpoint | " + " | ".join(f"{k}: c | {k}: d′" for k in kinds) + " |",
             "|:---|:---|" + "---:|" * (2 * len(kinds))]
    for (pr, v), r in tab.iterrows():
        vals = " | ".join(f"{r[(k, 'c')]:.2f} | {r[(k, 'd')]:.2f}" for k in kinds)
        lines.append(f"| {names.get(pr, pr)} | {'base' if v == 'base' else 'final'} | {vals} |")
    allm = t.groupby(["kind", "measure"]).eff.apply(lambda v: np.abs(v).mean())
    lines.append("| All | | " + " | ".join(f"{allm[(k, 'c')]:.2f} | {allm[(k, 'd')]:.2f}" for k in kinds) + " |")
    (config.OUTPUT_DIR / "sdt_invariance_table.md").write_text("\n".join(lines) + "\n")
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print("all cells:\n", summ.to_string())
        print("\nwithout saturated conditions:\n", summ_ns.to_string())


if __name__ == "__main__":
    main()
