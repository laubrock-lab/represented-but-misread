"""Pilot cited in the Introduction: commercial models as judges of the short chats.

In the lab course from which the chat corpus stems, four commercial models judged each chat
message via their APIs (one message per request, temperature 0), with five prompt versions
(system prompts: neutral, expert; with and without a request to analyze the style first; one
version with a note on the two kinds of error and the share of AI texts). Each answer consists
of a choice ("Mensch" / "KI") and a confidence from 0 to 100. These judgments were collected
before and independently of the analyses of this paper; the per-message file is not part of
this repository, because its identifiers refer to unpublished texts.

This script condenses the per-message file into one row per judge model and prompt version:
AUC of the signed confidence (confidence for "KI", minus confidence for "Mensch") with a
bootstrap CI that resamples source messages (a human message and the AI messages derived from
it), the AUC of the binary choice, and the share of "KI" answers overall, for AI messages
(hits), and for human messages (false alarms).

Usage:  python pilot_llm_judge_summary.py [--input llm_judgments.csv] [--n-boot 2000]
        -> outputs/pilot_llm_judge.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

DEFAULT = config.LABCOURSE_DIR / "python" / "output" / "llm_judge" / "llm_judgments.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, default=DEFAULT, help="per-message judgments of the lab course")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    d = pd.read_csv(a.input)
    assert d[["choice", "confidence"]].notna().all().all() and set(d.choice) <= {"KI", "Mensch"}
    d["ai"] = (d.choice == "KI").astype(int)
    d["score"] = np.where(d.ai == 1, d.confidence, -d.confidence)
    rng = np.random.default_rng(1)
    rows = []
    for (judge, prompt), g in d.groupby(["judge_model", "prompt_version"]):
        y, s, b = g.is_ai.to_numpy(), g.score.to_numpy(), g.ai.to_numpy()
        idx = list(pd.Series(np.arange(len(g))).groupby(g.source_id.to_numpy()).indices.values())
        boot = []
        while len(boot) < a.n_boot:
            i = np.concatenate([idx[k] for k in rng.integers(0, len(idx), len(idx))])
            if y[i].min() != y[i].max():
                boot.append(roc_auc_score(y[i], s[i]))
        rows.append(dict(judge_model=judge, prompt_version=prompt, n=len(g), n_human=int((y == 0).sum()),
                         n_sources=len(idx), auc=roc_auc_score(y, s), lo=np.quantile(boot, .025),
                         hi=np.quantile(boot, .975), auc_binary=roc_auc_score(y, b), share_ai_answers=b.mean(),
                         hit_rate=b[y == 1].mean(), false_alarm_rate=b[y == 0].mean()))
    r = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / "pilot_llm_judge.csv"
    r.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(r.to_string(index=False))
        print(r.groupby("judge_model").agg(auc_min=("auc", "min"), auc_max=("auc", "max"), lo_min=("lo", "min"),
                                           share_ai_max=("share_ai_answers", "max")))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
