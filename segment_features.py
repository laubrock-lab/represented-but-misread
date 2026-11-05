"""Compute the lab course text features for arbitrary items (e.g. journalism segments).

Uses `extract_features` (and optionally the LeoLM `ModelScorer`) from the
lab course pipeline unchanged, so the cues are defined exactly as in the lab course
analyses. Input: a parquet from prepare_texts.py; the feature text is the
column `segment` if present (body only, without headline), else `text`.
Output: <out_dir>/features_<stem>.csv with text_id, keys and features only —
no text.

Needs spaCy + de_core_news_lg + sentence-transformers (+ textstat, wordfreq);
on this machine: miniconda3/envs/textrpp_condaenv.

Usage
-----
    python segment_features.py data/texts_journalism_seg.parquet
    python segment_features.py data/texts_journalism_seg.parquet --with-lm   # + LeoLM features
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", type=Path)
    ap.add_argument("--with-lm", action="store_true",
                    help="also compute LeoLM perplexity/entropy/Binoculars features (slow)")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    labcourse_py = config.LABCOURSE_DIR / "python"
    sys.path.insert(0, str(labcourse_py))
    # lab course's config.py clashes with ours by name: load it explicitly from its path
    spec = importlib.util.spec_from_file_location("labcourse_config", labcourse_py / "config.py")
    labcourse_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labcourse_config)
    import spacy  # noqa: E402
    from sentence_transformers import SentenceTransformer  # noqa: E402
    from feature_extractor import extract_features  # noqa: E402

    df = pd.read_parquet(args.texts)
    col = "segment" if "segment" in df.columns else "text"
    nlp = spacy.load(labcourse_config.SPACY_MODEL)
    smodel = SentenceTransformer(labcourse_config.SENTENCE_MODEL)

    keys = [c for c in ("text_id", "source_id", "generator", "gen_model", "prompt_type",
                        "is_ai", "seg_idx") if c in df.columns]
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        rows.append({k: getattr(r, k) for k in keys} | extract_features(getattr(r, col), nlp, smodel))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df)}")
    feats = pd.DataFrame(rows)

    if args.with_lm:
        from model_scorer import ModelScorer  # noqa: E402
        scorer = ModelScorer(labcourse_config.SCORER_MODEL_A, labcourse_config.SCORER_MODEL_B,
                             device=labcourse_config.DEVICE, max_tokens=labcourse_config.MAX_TOKENS)
        lm = pd.DataFrame([scorer.score(t) for t in df[col]])
        scorer.close()
        feats = pd.concat([feats, lm], axis=1)

    stem = args.texts.stem.removeprefix("texts_")
    out_dir = args.out_dir or config.OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"features_{stem}.csv"
    feats.to_csv(path, index=False)
    print(f"{len(feats)} items x {feats.shape[1] - len(keys)} features -> {path}")


if __name__ == "__main__":
    main()
