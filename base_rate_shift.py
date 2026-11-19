"""Base-rate shift of the judgment logit.

judge_logits.py --priors states the share of AI texts in the study (20/50/80 %),
with the AI share or the human share named first (--prior-orders ai|hu).  If the
models used that statement as a Bayesian prior, dv = log P(KI) - log P(Mensch)
should move by the change in the prior log odds and by nothing else:

    80 % vs. 20 %   ln(.8/.2) - ln(.2/.8) = ln 16 = 2.7726
    50 % vs. 20 %   and   80 % vs. 50 %     ln 4  = 1.3863

This script measures the realised shift per text (paired by text_id) for every
corpus x preset x variant x answer order x prior order that has all three prior
files, and expresses it as a "gain" relative to that benchmark.  It also splits
the shift by true class (human vs. AI texts), because only a class-dependent
shift can move d'; a class-independent shift moves the criterion c only.

CIs are grouped bootstrap over source_id (all texts derived from the same human
original resampled together), as in sdt_summary.py / sdt_invariance.py.  The
resamples depend only on the group structure and the seed, so the same draws are
used for every condition of a cell, which makes the prior-order contrast
(hu - ai) and the AI - human contrast paired.

Usage
-----
    python base_rate_shift.py                       # -> outputs/base_rate_shift*.csv/.md
    python base_rate_shift.py --outputs-dir outputs --n-boot 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config

CORPORA = ["chat_tw25", "journalism_seg", "textbook"]
PRIORS = ["20", "50", "80"]
PRIOR_ORDERS = list(config.PRIOR_ORDERS)          # ai, hu
# (name, low prior, high prior, benchmark): the Bayes-optimal shift in nats
CONTRASTS = [("80-20", "20", "80", np.log(16.0)),
             ("50-20", "20", "50", np.log(4.0)),
             ("80-50", "50", "80", np.log(4.0))]
GROUP_COL = "source_id"
MAIN = "80-20"
NAMES = {"gemma3-4b": "Gemma 3 4B", "gemma3-12b": "Gemma 3 12B", "qwen3-8b": "Qwen3 8B",
         "olmo2-7b": "OLMo 2 7B", "apertus15-8b": "Apertus 1.5 8B",
         "olmo3-7b-think": "OLMo 3 7B Think", "olmo3-7b-instruct": "OLMo 3 7B Instruct",
         "olmo3-7b-rlzero": "OLMo 3 7B RL-Zero"}


def stem(preset: str, variant: str, fmt: str, order: str, system: str,
         prior: str, prior_order: str) -> str:
    """File stem of one condition (same rule as judge_logits.condition_stem)."""
    s = f"{preset}_{variant}_{fmt}_{order}" + (f"_{system}" if fmt == "chat" else "")
    return s + f"_p{prior}{prior_order}"


def load_cell(d: Path, preset, variant, fmt, order, system, prior_order, dv_col):
    """The three prior files of one cell, inner-joined on text_id; None if incomplete."""
    frames = {}
    for p in PRIORS:
        f = d / f"{stem(preset, variant, fmt, order, system, p, prior_order)}.csv"
        if not f.exists():
            return None
        x = pd.read_csv(f, usecols=["text_id", GROUP_COL, "is_ai", dv_col, "p_mass"])
        frames[p] = x.drop_duplicates("text_id").set_index("text_id")
    ids = frames["20"].index
    for x in frames.values():
        ids = ids.intersection(x.index)
    ids = ids.sort_values()
    return {p: x.loc[ids] for p, x in frames.items()}


def make_boots(groups: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Grouped bootstrap index sets; deterministic given the group vector and seed."""
    by_group = pd.Series(np.arange(len(groups))).groupby(groups).indices
    keys = np.array(list(by_group), dtype=object)
    rng = np.random.default_rng(seed)
    return [np.concatenate([by_group[k] for k in rng.choice(keys, size=len(keys))])
            for _ in range(n_boot)]


def ci(vals: list[float] | np.ndarray) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(np.nanquantile(a, .025)), float(np.nanquantile(a, .975))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", type=Path, default=config.OUTPUT_DIR,
                    help="directory holding the per-corpus subdirectories")
    ap.add_argument("--out-dir", type=Path, help="where the tables are written (default: --outputs-dir)")
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--presets", nargs="+", default=list(config.MODEL_PRESETS))
    ap.add_argument("--variants", nargs="+", default=["base", "instruct"])
    ap.add_argument("--formats", nargs="+", default=["raw"], choices=["raw", "chat"])
    ap.add_argument("--orders", nargs="+", default=["mk", "km"])
    ap.add_argument("--system", default="neutral", help="system prompt label of chat-format files")
    ap.add_argument("--dv", default="dv", choices=["dv", "dv_prefix"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out_dir = args.out_dir or args.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for corpus in args.corpora:
        d = args.outputs_dir / corpus
        if not d.is_dir():
            continue
        for preset in args.presets:
            for variant in args.variants:
                for fmt in args.formats:
                    for order in args.orders:
                        cells = {po: load_cell(d, preset, variant, fmt, order, args.system, po, args.dv)
                                 for po in PRIOR_ORDERS}
                        cells = {po: c for po, c in cells.items() if c is not None}
                        if not cells:
                            continue
                        key = dict(corpus=corpus, preset=preset, variant=variant, fmt=fmt, order=order)
                        shifts = {}       # (prior_order, contrast) -> per-text shift
                        for po, C in cells.items():
                            ref = C["20"]
                            groups = ref[GROUP_COL].to_numpy()
                            is_ai = ref["is_ai"].to_numpy().astype(bool)
                            boots = make_boots(groups, args.n_boot, args.seed)
                            for name, lo_p, hi_p, bench in CONTRASTS:
                                s = (C[hi_p][args.dv].to_numpy() - C[lo_p][args.dv].to_numpy())
                                shifts[(po, name)] = (s, is_ai, boots, C[lo_p].index)
                                bs = np.array([s[b].mean() for b in boots])
                                dif = np.array([s[b][is_ai[b]].mean() - s[b][~is_ai[b]].mean()
                                                for b in boots])
                                m_lo, m_hi = ci(bs)
                                d_lo, d_hi = ci(dif)
                                rows.append(key | dict(
                                    prior_order=po, contrast=name, n=len(s),
                                    n_groups=len(set(groups)),
                                    mean_shift=float(s.mean()), lo=m_lo, hi=m_hi,
                                    sd=float(s.std(ddof=1)),
                                    p05=float(np.percentile(s, 5)), p50=float(np.percentile(s, 50)),
                                    p95=float(np.percentile(s, 95)),
                                    mean_shift_human=float(s[~is_ai].mean()),
                                    mean_shift_ai=float(s[is_ai].mean()),
                                    diff_ai_human=float(s[is_ai].mean() - s[~is_ai].mean()),
                                    diff_lo=d_lo, diff_hi=d_hi,
                                    benchmark=float(bench), gain=float(s.mean() / bench),
                                    p_mass_lo=float(C[lo_p]["p_mass"].mean()),
                                    p_mass_hi=float(C[hi_p]["p_mass"].mean()),
                                    share_ki_lo=float((C[lo_p][args.dv].to_numpy() > 0).mean()),
                                    share_ki_hi=float((C[hi_p][args.dv].to_numpy() > 0).mean()),
                                    prior_lo=lo_p, prior_hi=hi_p))
                        # paired prior-order contrast (hu - ai) on the same texts
                        if ("ai", MAIN) in shifts and ("hu", MAIN) in shifts:
                            s_ai, is_ai_v, boots, idx_ai = shifts[("ai", MAIN)]
                            s_hu, _, _, idx_hu = shifts[("hu", MAIN)]
                            common = idx_ai.intersection(idx_hu)
                            a = pd.Series(s_ai, index=idx_ai).loc[common].to_numpy()
                            h = pd.Series(s_hu, index=idx_hu).loc[common].to_numpy()
                            sub = cells["ai"]["20"].loc[common]
                            groups = sub[GROUP_COL].to_numpy()
                            is_ai = sub["is_ai"].to_numpy().astype(bool)
                            boots = make_boots(groups, args.n_boot, args.seed)
                            s = h - a
                            bs = np.array([s[b].mean() for b in boots])
                            dif = np.array([s[b][is_ai[b]].mean() - s[b][~is_ai[b]].mean() for b in boots])
                            m_lo, m_hi = ci(bs)
                            d_lo, d_hi = ci(dif)
                            rows.append(key | dict(
                                prior_order="hu-ai", contrast=f"{MAIN} (hu-ai)", n=len(s),
                                n_groups=len(set(groups)),
                                mean_shift=float(s.mean()), lo=m_lo, hi=m_hi,
                                sd=float(s.std(ddof=1)),
                                p05=float(np.percentile(s, 5)), p50=float(np.percentile(s, 50)),
                                p95=float(np.percentile(s, 95)),
                                mean_shift_human=float(s[~is_ai].mean()),
                                mean_shift_ai=float(s[is_ai].mean()),
                                diff_ai_human=float(s[is_ai].mean() - s[~is_ai].mean()),
                                diff_lo=d_lo, diff_hi=d_hi,
                                benchmark=np.nan, gain=np.nan,
                                p_mass_lo=np.nan, p_mass_hi=np.nan,
                                share_ki_lo=np.nan, share_ki_hi=np.nan,
                                prior_lo="20", prior_hi="80"))
    if not rows:
        raise SystemExit(f"no prior files found under {args.outputs_dir}")
    cells_df = pd.DataFrame(rows)
    cells_df.to_csv(out_dir / "base_rate_shift.csv", index=False)

    # ---- per model: mean over corpora and answer orders of the 20->80 shift ----
    main = cells_df[cells_df.contrast.isin([MAIN, f"{MAIN} (hu-ai)"])]
    g = main.groupby(["preset", "variant", "prior_order"], as_index=False).agg(
        mean_shift=("mean_shift", "mean"), min_shift=("mean_shift", "min"),
        max_shift=("mean_shift", "max"), n_cells=("mean_shift", "size"),
        mean_gain=("gain", "mean"), mean_sd=("sd", "mean"),
        mean_diff_ai_human=("diff_ai_human", "mean"))
    # family-level difference instruct - base (same value on both rows of a preset)
    piv = g.pivot_table(index=["preset", "prior_order"], columns="variant", values="mean_shift")
    fam = (piv.get("instruct") - piv.get("base")) if {"base", "instruct"} <= set(piv.columns) else None
    g["instruct_minus_base"] = (
        [fam.get((p, po), np.nan) for p, po in zip(g.preset, g.prior_order)] if fam is not None
        else np.nan)
    g.to_csv(out_dir / "base_rate_shift_models.csv", index=False)

    # ---- markdown table ----
    lines = ["| Model | Checkpoint | Prior order | Mean 20→80 shift | Min–max over cells | "
             "Gain (/ln 16) | Mean between-text SD | Cells |",
             "|:---|:---|:---|---:|:---:|---:|---:|---:|"]
    for _, r in g.sort_values(["preset", "variant", "prior_order"]).iterrows():
        gain = "—" if not np.isfinite(r.mean_gain) else f"{r.mean_gain:.2f}"
        lines.append(f"| {NAMES.get(r.preset, r.preset)} | "
                     f"{'base' if r.variant == 'base' else 'final'} | {r.prior_order} | "
                     f"{r.mean_shift:+.2f} | {r.min_shift:+.2f} – {r.max_shift:+.2f} | {gain} | "
                     f"{r.mean_sd:.2f} | {int(r.n_cells)} |")
    (out_dir / "base_rate_shift_table.md").write_text("\n".join(lines) + "\n")

    with pd.option_context("display.width", 220, "display.max_columns", None,
                           "display.float_format", "{:+.3f}".format):
        print(g.to_string(index=False))
    print(f"\n-> {out_dir / 'base_rate_shift.csv'} ({len(cells_df)} rows)")
    print(f"-> {out_dir / 'base_rate_shift_models.csv'} ({len(g)} rows)")
    print(f"-> {out_dir / 'base_rate_shift_table.md'}")


if __name__ == "__main__":
    main()
