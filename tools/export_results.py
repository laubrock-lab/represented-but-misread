"""Copy the tables of derived results behind the manuscript from outputs/ (not tracked) to
results/ (tracked, public).

Only aggregate tables are exported. Per-text model outputs of the authorship corpora are not,
because their identifiers refer to unpublished texts. Every exported file is checked: no
per-item identifier column, no free-text column, and no author code of the chat corpus (the
codes are derived from the identifiers in outputs/ and are never printed).

Usage: python tools/export_results.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT, RES = REPO / "outputs", REPO / "results"
PATTERNS = [
    "meta_multilevel.csv", "meta_prompts*.csv", "meta_contrasts.csv", "contrasts*.csv", "promptvar_summary.csv",
    "sdt_invariance_*.csv", "sdt_invariance_table.md", "base_rate_shift*.csv", "base_rate_shift_table.md",
    "base_rate_scale*.csv", "baselines*.csv", "cka_textbook.csv", "checkpoint_transfer_*.csv",
    "probe_inner_cv_check.csv", "steer_nulls_summary.csv", "steer_cosines.csv", "steer_cache_check.csv", "review_polarity.csv",
    "review_convergence.csv", "pilot_llm_judge*.csv", "truth_stages_contrasts.csv", "stages_summary.csv", "lens_opspam*.csv",
    "fewshot/fewshot_summary.csv", "fewshot/fewshot_interaction.csv",
    "stitch/stitch_summary.csv", "stitch/stitch_neg_cities.csv", "stitch/*/perplexity.csv",
    "patch/patch_summary.csv", "patch/sweep_check.csv", "patch_sweep/patch_sweep_summary.csv",
    "training_study/answers_summary.csv", "training_study/contrasts_ts.csv", "training_study/heldout_breakdown.csv",
    "training_study/formality_*.csv", "training_study/transfer_ts.csv", "training_study/prediction_test.csv", "training_study/prediction_ngram_control.csv",
    "truth_*/stages.csv", "truth_neg_cities/sfirst_answers_summary.csv", "sft_check/*_counts.csv", "figdata/*.csv",
]
ITEM_COLS = {"text_id", "source_id", "item_id", "pair_id", "text", "segment", "user", "assistant"}


def author_codes() -> list[str]:
    codes = set()
    for f in (OUT / "chat_tw25").glob("*_raw_mk.csv"):
        codes |= {str(s).split("__")[0] for s in pd.read_csv(f, usecols=["source_id"]).source_id}
        break
    return sorted(codes)


def main() -> None:
    codes = author_codes()
    assert codes, "chat identifiers not found; cannot check for author codes"
    pat = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, codes)) + r")(?![A-Za-z0-9])")
    files = sorted({f for p in PATTERNS for f in OUT.glob(p)})
    if RES.exists():
        shutil.rmtree(RES)
    kept, skipped = [], []
    for f in files:
        txt = f.read_text(errors="replace")
        reason = None
        if pat.search(txt):
            reason = "contains an author code"
        elif f.suffix == ".csv":
            d = pd.read_csv(f, nrows=500, dtype=str, engine="python", on_bad_lines="skip")
            if ITEM_COLS & set(map(str.lower, d.columns)):
                reason = "per-item identifier column"
            else:
                for c in (c for c in d.columns if c != "note"):   # "note" is dropped on export
                    v = d[c].dropna()
                    if len(v) and v.str.len().mean() > 80 and v.str.count(" ").mean() > 6:
                        reason = f"free-text column {c}"
        if reason:
            skipped.append((f.relative_to(OUT), reason))
            continue
        dst = RES / f.relative_to(OUT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        drop = [c for c in ("note", "run_old", "run_new", "file_old", "file_new") if c in d.columns]
        if f.suffix == ".csv" and drop:      # remarks and run descriptions (hosts, job ids, paths), not data
            pd.read_csv(f).drop(columns=drop).to_csv(dst, index=False)
        else:
            shutil.copy2(f, dst)
        kept.append(f.relative_to(OUT))
    (RES / "README.md").write_text(
        "# Derived results\n\nAggregate tables written by the analysis scripts (see the code overview in the\n"
        "top-level README), from which the numbers in the manuscript are taken. Paths mirror\n"
        "`outputs/`. Per-text model outputs of the authorship corpora are not included, because\n"
        "their identifiers refer to unpublished texts. Exported by `tools/export_results.py`.\n")
    print(f"exported {len(kept)} files, {sum((RES / k).stat().st_size for k in kept) / 1e6:.1f} MB")
    for k, r in skipped:
        print("skipped:", k, "|", r)


if __name__ == "__main__":
    sys.exit(main())
