"""Brunswik lens model: which cues do humans and models use, and which are valid?

Criterion: is_ai (the true class). Cues: the 46 lab course text features.
Judges: humans (item-level share of "KI", or confidence-signed judgment) and
each model condition (dv = log P(KI) - log P(Mensch), averaged over answer
orders).

    achievement         r(judgment, criterion)
    ecological validity r_e,i = r(cue_i, criterion)
    cue utilization     r_s,i = r(cue_i, judgment)       per judge
    profile similarity  r over cues between utilization profiles
                        (model vs. human, model vs. ecological validity)
    item agreement      r(model dv, human judgment) — overall and within class

Two variants: "raw" (across all items, the classic lens model) and
"within" (cues and judgments centred within class x prompt type, i.e. cue use
*beyond* the class differences; this removes the trivial part of the overlap
between utilization and validity).

Item-level cue files with a text_id column (e.g. from segment_features.py)
are matched on text_id; the lab course chat features_raw.csv on source/generator.

All correlations are Spearman. CIs: grouped bootstrap over source_id. The key
test is the difference instruct - base in similarity to the human profile.

Usage
-----
    python lens_model.py outputs/chat --features \\
        $LABCOURSE_DIR/python/output/chat/features_raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from sdt_summary import is_judgment_csv

FEATURES = [
    "word_count", "sent_count", "mean_sent_len_tokens", "std_sent_len_tokens",
    "burstiness_cv", "ttr", "mtld", "punct_density", "mean_word_len", "hapax_ratio",
    "flesch_reading_ease_de", "wiener_sachtextformel",
    "pos_noun_ratio", "pos_verb_ratio", "pos_adj_ratio", "pos_adv_ratio", "pos_propn_ratio",
    "ne_density", "dep_depth_mean", "dep_depth_max", "dep_depth_std",
    "embedding_variance", "mean_adjacent_cosine_sim", "min_adjacent_cosine_sim",
    "global_coherence", "sent_centroid_mean_sim", "sent_centroid_std_sim",
    "coherence_slope", "noun_overlap_ratio",
    "mean_word_freq_zipf", "std_word_freq_zipf",
    "mean_logprob_a", "std_logprob_a", "perplexity_a", "mean_entropy_a",
    "std_entropy_a", "mean_log_rank_a", "cross_entropy_a",
    "mean_logprob_b", "std_logprob_b", "perplexity_b", "mean_entropy_b",
    "std_entropy_b", "mean_log_rank_b", "cross_entropy_b", "binoculars_score",
]
GROUP_MODEL = {"gpt": "gpt-5.4", "claude": "claude-opus-4-6"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_items(out_dir: Path, features_csv: Path, human_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    # model judgments, averaged over answer orders per condition
    files = [f for f in out_dir.glob("*.csv") if is_judgment_csv(f)]
    j = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    if "prior" in j:   # lens model uses the judgments without base-rate statement only
        j = j[j["prior"].fillna("none").astype(str).isin(["none", ""])]
    j["judge"] = j["preset"] + "_" + j["variant"] + "_" + j["fmt"]
    keys = ["text_id", "source_id", "generator", "gen_model", "prompt_type", "is_ai"]
    wide = (j.groupby(keys + ["judge"])["dv"].mean().unstack("judge").reset_index())
    judges = sorted(c for c in wide.columns if c not in keys)
    n_len = j.drop_duplicates("text_id").set_index("text_id")["n_text_tokens"]
    wide["n_text_tokens"] = wide["text_id"].map(n_len)

    # cues (feature columns only — lab course's features_raw.csv also contains the
    # texts, which are never read here). Item-level files (e.g. segments) are
    # matched on text_id, the lab course chat file on source_id/generator/model.
    header = pd.read_csv(features_csv, nrows=0).columns
    feats = [c for c in FEATURES if c in header]
    if "text_id" in header:
        f = pd.read_csv(features_csv, usecols=["text_id"] + feats)
        items = wide.merge(f, on="text_id", how="left")
    else:
        f = pd.read_csv(features_csv, usecols=["source_id", "generator", "model_group"] + feats)
        f["gen_model"] = np.where(f["generator"] == "human_text", "",
                                  f["model_group"].map(GROUP_MODEL))
        f = f.drop(columns="model_group").drop_duplicates(["source_id", "generator", "gen_model"])
        items = wide.merge(f, on=["source_id", "generator", "gen_model"], how="left")

    # human judgments
    h = pd.read_csv(human_csv, keep_default_na=False)
    h = h.rename(columns={"p_ki": "human_p_ki", "signed_conf": "human_signed", "n": "human_n"})
    on = ["text_id"] if "text_id" in h.columns else ["source_id", "generator", "gen_model"]
    items = items.merge(h[on + ["human_p_ki", "human_signed", "human_n"]], on=on, how="left")
    return items, judges, feats


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _rank(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank().to_numpy()


def corr_cols(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of each column of A with b (NaN-safe per column)."""
    out = np.full(A.shape[1], np.nan)
    for i in range(A.shape[1]):
        m = ~np.isnan(A[:, i]) & ~np.isnan(b)
        if m.sum() > 3 and np.std(A[m, i]) > 0 and np.std(b[m]) > 0:
            out[i] = pd.Series(A[m, i]).corr(pd.Series(b[m]), method="spearman")
    return out


def within(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Centre columns within cell (prompt_type: human / naive / optimized)."""
    out = df.copy()
    out[cols] = df[cols] - df.groupby("prompt_type")[cols].transform("mean")
    return out


def lens_stats(items: pd.DataFrame, judges: list[str], human: str,
               FEATURES: list[str] = FEATURES) -> dict:
    """All lens-model statistics for one (possibly bootstrapped) item set."""
    res = {}
    # ecological validity: always on the raw cues (centring within class removes it)
    valid = corr_cols(items[FEATURES].to_numpy(float), items["is_ai"].to_numpy(float))
    res["_valid"] = valid
    for variant in ("raw", "within"):
        cols = FEATURES + judges + [human]
        d = items if variant == "raw" else within(items, cols)
        C = d[FEATURES].to_numpy(float)
        util = {k: corr_cols(C, d[k].to_numpy(float)) for k in judges + [human]}
        for k in judges + [human]:
            if variant == "raw":
                res[f"achievement|{k}"] = corr_cols(items[[k]].to_numpy(float),
                                                    items["is_ai"].to_numpy(float))[0]
                res[f"sim_validity|{k}"] = _profile_r(util[k], valid)
            if k != human:
                res[f"sim_human_{variant}|{k}"] = _profile_r(util[k], util[human])
                res[f"agree_{variant}|{k}"] = corr_cols(d[[k]].to_numpy(float),
                                                        d[human].to_numpy(float))[0]
        res[f"_util_{variant}"] = util
    return res


def _profile_r(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def bootstrap(items: pd.DataFrame, judges: list[str], human: str, n_boot: int,
              seed: int, feats: list[str] = FEATURES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = items["source_id"].unique()
    idx = items.groupby("source_id").indices
    rows = []
    for _ in range(n_boot):
        draw = rng.choice(groups, size=len(groups), replace=True)
        b = items.iloc[np.concatenate([idx[g] for g in draw])].reset_index(drop=True)
        s = lens_stats(b, judges, human, feats)
        rows.append({k: v for k, v in s.items() if not k.startswith("_")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--features", type=Path, required=True,
                    help="lab course features_raw.csv (only feature columns are read)")
    ap.add_argument("--human", type=Path, default=config.DATA_DIR / "human_items_chat.csv",
                    help="from export_human_items.R (chat or journalism_seg)")
    ap.add_argument("--human-measure", default="human_p_ki",
                    choices=["human_p_ki", "human_signed"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    items, judges, feats = load_items(args.out_dir, args.features, args.human)
    n_all = len(items)
    items = items.dropna(subset=[args.human_measure, feats[0]]).reset_index(drop=True)
    print(f"{n_all} model-judged items; {len(items)} with human judgments and cues "
          f"({items.groupby('prompt_type').size().to_dict()}); judges: {judges}")

    print(f"{len(feats)} cues")
    point = lens_stats(items, judges, args.human_measure, feats)
    boot = bootstrap(items, judges, args.human_measure, args.n_boot, args.seed, feats)

    # differences between model conditions (each vs. the base model of its preset)
    for k in [c for c in boot.columns if "|" in c]:
        stat, judge = k.split("|")
        if "_base_" in judge:
            continue
        base = judge.split("_")[0] + "_base_raw"
        if f"{stat}|{base}" in boot.columns:
            boot[f"{stat}|{judge}-minus-base"] = boot[k] - boot[f"{stat}|{base}"]
            point[f"{stat}|{judge}-minus-base"] = point[k] - point[f"{stat}|{base}"]

    summary = pd.DataFrame([
        dict(stat=k.split("|")[0], judge=k.split("|")[1], estimate=point[k],
             lo=boot[k].quantile(0.025), hi=boot[k].quantile(0.975))
        for k in boot.columns])
    summary.to_csv(args.out_dir / "lens_summary.csv", index=False)

    prof = pd.DataFrame({"feature": feats, "validity": point["_valid"]})
    for variant in ("raw", "within"):
        for k, v in point[f"_util_{variant}"].items():
            prof[f"util_{variant}|{k}"] = v
    prof.to_csv(args.out_dir / "lens_profiles.csv", index=False)

    order = ["achievement", "sim_validity", "agree_raw", "agree_within",
             "sim_human_raw", "sim_human_within"]
    summary["stat"] = pd.Categorical(summary["stat"], order)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.2f}".format):
        print(summary.sort_values(["stat", "judge"]).to_string(index=False))
    show = prof.reindex(prof["validity"].abs().sort_values(ascending=False).index).head(12)
    with pd.option_context("display.width", 250, "display.float_format", "{:+.2f}".format):
        print("\nTop-12 cues by |ecological validity| (raw utilization):")
        print(show[["feature", "validity"] + [c for c in prof.columns if c.startswith("util_raw")]]
              .to_string(index=False))
    print(f"\n-> {args.out_dir / 'lens_summary.csv'}, lens_profiles.csv")


if __name__ == "__main__":
    main()
