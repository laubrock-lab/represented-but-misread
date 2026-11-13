"""Chat messages: human vs. three kinds of AI message (naive, style-matched, optimized =
style-matched + revised after the students' feedback). Separates the contributions of
style matching and feedback revision to the graded reversal (Discussion).

Answers for human / naive / optimized come from outputs/chat_tw25 (main corpus); answers
for the style-matched messages (same two generators: Claude Opus 4.6, GPT-5.4; same
truncation to 25 words) from outputs/chat_sm_tw25 (prepare_texts.py --all-generators).
Both answer orders averaged.

Per model condition (preset x variant x format):
    auc_<cond>   AUC of "AI" (dv) for <cond> messages vs. human messages
                 (< .5 = the AI messages are judged MORE human than the human ones)
    dv_<cond>    mean dv (log P("KI") - log P("Mensch")) per condition
CIs: grouped bootstrap over source groups (human original + all its AI versions).

Usage:  python chat_style_conditions.py
Output: outputs/chat_style_conditions.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

CONDS = ["naive", "style_matched", "optimized"]


def answers(corpus: str, preset: str, variant: str, fmt: str) -> pd.DataFrame | None:
    sfx = "_neutral" if fmt == "chat" else ""
    fs = [config.OUTPUT_DIR / corpus / f"{preset}_{variant}_{fmt}_{o}{sfx}.csv" for o in ("mk", "km")]
    if not all(f.exists() for f in fs):
        return None
    d = pd.concat([pd.read_csv(f, usecols=["text_id", "source_id", "is_ai", "dv", "p_mass"]) for f in fs])
    return d.groupby(["text_id", "source_id", "is_ai"], as_index=False).agg(dv=("dv", "mean"), p_mass=("p_mass", "mean"))


def main() -> None:
    meta = pd.concat([pd.read_parquet(config.REPO / "data" / f"texts_{c}.parquet",
                                      columns=["text_id", "prompt_type", "gen_model"])
                      for c in ("chat_tw25", "chat_sm_tw25")]).drop_duplicates("text_id")
    rng = np.random.default_rng(1)
    rows = []
    for preset in config.MODEL_PRESETS:
        for variant in ("base", "instruct"):
            for fmt in ("raw", "chat"):
                a, b = answers("chat_tw25", preset, variant, fmt), answers("chat_sm_tw25", preset, variant, fmt)
                if a is None or b is None:
                    continue
                d = pd.concat([a, b]).merge(meta, on="text_id")
                groups = d.groupby("source_id").indices
                keys = np.array(list(groups), dtype=object)

                def stats(dd):
                    hum = dd[dd.prompt_type == "human"].dv
                    out = {}
                    for c in CONDS:
                        ai = dd[dd.prompt_type == c].dv
                        y = np.r_[np.zeros(len(hum)), np.ones(len(ai))]
                        out[f"auc_{c}"] = roc_auc_score(y, np.r_[hum, ai])
                        out[f"dv_{c}"] = ai.mean()
                    out["dv_human"] = hum.mean()
                    return out

                r = stats(d)
                boots = [stats(d.iloc[np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])])
                         for _ in range(1000)]
                b = pd.DataFrame(boots)
                for c in CONDS:
                    r[f"auc_{c}_lo"], r[f"auc_{c}_hi"] = b[f"auc_{c}"].quantile([.025, .975])
                # paired differences: style_matched - naive, optimized - style_matched
                for x, y in (("style_matched", "naive"), ("optimized", "style_matched")):
                    diff = b[f"auc_{x}"] - b[f"auc_{y}"]
                    r[f"diff_{x}_vs_{y}"] = r[f"auc_{x}"] - r[f"auc_{y}"]
                    r[f"diff_{x}_vs_{y}_lo"], r[f"diff_{x}_vs_{y}_hi"] = diff.quantile([.025, .975])
                rows.append(dict(preset=preset, variant=variant, fmt=fmt, p_mass=d.p_mass.mean()) | r)
    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUT_DIR / "chat_style_conditions.csv", index=False)
    show = ["preset", "variant", "fmt", "p_mass", "auc_naive", "auc_style_matched", "auc_optimized",
            "diff_style_matched_vs_naive", "diff_style_matched_vs_naive_lo", "diff_style_matched_vs_naive_hi",
            "diff_optimized_vs_style_matched", "diff_optimized_vs_style_matched_lo", "diff_optimized_vs_style_matched_hi"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(res[show].to_string(index=False))


if __name__ == "__main__":
    main()
