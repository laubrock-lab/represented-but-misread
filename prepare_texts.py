"""Export the lab course stimulus texts to local, gitignored parquet files.

Reads the Excel files from the internal lab course repository and writes one
long-format table per corpus to data/ with the columns

    text_id, source_id, generator, gen_model, prompt_type, is_ai, text

The journalism texts are confidential: they are only ever processed locally
(open-weight models, no API calls) and must never be committed or published.

Usage
-----
    python prepare_texts.py                 # both corpora, experiment subset
    python prepare_texts.py --corpus chat   # chats only
    python prepare_texts.py --all-generators
"""

from __future__ import annotations

import argparse

import pandas as pd

import config

# (column in xlsx, generator label, is_ai, prompt_type)
CHAT_COLS_EXPERIMENT = [
    ("human_text",         "human_text",         0, "human"),
    ("basic",              "basic",              1, "naive"),
    ("feedback_optimized", "feedback_optimized", 1, "optimized"),
]
CHAT_COLS_ALL = CHAT_COLS_EXPERIMENT + [
    ("style_matched", "style_matched", 1, "style_matched"),
]
CHAT_EXPERIMENT_MODELS = ["gpt-5.4", "claude-opus-4-6"]

JOURNALISM_COLS_EXPERIMENT = [
    ("original_text",                               "human_text",                   0, "human",     ""),
    ("generated_text_openai_naive",                 "openai_naive",                 1, "naive",     "openai"),
    ("generated_text_openai_style_matched_refined", "openai_style_matched_refined", 1, "optimized", "openai"),
]
JOURNALISM_COLS_ALL = JOURNALISM_COLS_EXPERIMENT + [
    ("generated_text_openai_style_matched",            "openai_style_matched",            1, "style_matched", "openai"),
    ("generated_text_anthropic_naive",                 "anthropic_naive",                 1, "naive",         "anthropic"),
    ("generated_text_anthropic_style_matched",         "anthropic_style_matched",         1, "style_matched", "anthropic"),
    ("generated_text_anthropic_style_matched_refined", "anthropic_style_matched_refined", 1, "optimized",     "anthropic"),
]


def _clean(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s or None


def load_chats(all_generators: bool) -> pd.DataFrame:
    df = pd.read_excel(config.CHATS_XLSX)
    if not all_generators:
        df = df[df["model_name"].isin(CHAT_EXPERIMENT_MODELS)]
    cols = CHAT_COLS_ALL if all_generators else CHAT_COLS_EXPERIMENT

    rows, seen_human = [], set()
    for _, row in df.iterrows():
        sid = str(row["source_id"])
        for col, gen, is_ai, ptype in cols:
            text = _clean(row.get(col))
            if text is None:
                continue
            if not is_ai:
                # one human original per source, shared across generator models
                if sid in seen_human:
                    continue
                seen_human.add(sid)
                gen_model = ""
            else:
                gen_model = str(row["model_name"])
            rows.append(dict(source_id=sid, generator=gen, gen_model=gen_model,
                             prompt_type=ptype, is_ai=is_ai, text=text))
    return _finish(rows)


def load_journalism(all_generators: bool) -> pd.DataFrame:
    df = pd.read_excel(config.JOURNALISM_XLSX)
    cols = JOURNALISM_COLS_ALL if all_generators else JOURNALISM_COLS_EXPERIMENT
    rows = []
    for _, row in df.iterrows():
        sid = str(row.get("Text_ID", ""))
        for col, gen, is_ai, ptype, gen_model in cols:
            text = _clean(row.get(col))
            if text is None:
                continue
            rows.append(dict(source_id=sid, generator=gen, gen_model=gen_model,
                             prompt_type=ptype, is_ai=is_ai, text=text))
    return _finish(rows)


def _finish(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out.insert(0, "text_id", out["source_id"] + "__" + out["generator"] + "__"
               + out["gen_model"].replace("", "human"))
    if out["text_id"].duplicated().any():
        raise ValueError("Duplicate text_ids — check the source tables.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=["chat", "journalism", "both"], default="both")
    ap.add_argument("--all-generators", action="store_true",
                    help="include generators/models not used in the human experiment")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    loaders = {"chat": load_chats, "journalism": load_journalism}
    corpora = ["chat", "journalism"] if args.corpus == "both" else [args.corpus]
    suffix = "_all" if args.all_generators else ""
    for corpus in corpora:
        df = loaders[corpus](args.all_generators)
        path = config.DATA_DIR / f"texts_{corpus}{suffix}.parquet"
        df.to_parquet(path, index=False)
        counts = df.groupby(["is_ai", "prompt_type"]).size().to_dict()
        print(f"{corpus}: {len(df)} texts, {df['source_id'].nunique()} sources "
              f"-> {path.relative_to(config.REPO)}\n  counts (is_ai, prompt_type): {counts}")


if __name__ == "__main__":
    main()
