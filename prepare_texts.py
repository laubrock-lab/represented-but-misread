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
    python prepare_texts.py --corpus chat --truncate-words 20    # length control
    python prepare_texts.py --corpus journalism --windows 80     # length control
    python prepare_texts.py --corpus journalism_seg              # items as in the human experiment

Length control: --truncate-words N keeps the first N words of every text and
drops texts with fewer than N words; --windows N cuts every text into
non-overlapping N-word windows (the remainder is discarded). Windows keep
their source_id, so grouped CV still treats the source as the unit.
Original whitespace and line breaks inside the kept span are preserved.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
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


# Items excluded in the lab course preprocessing (analysis/preprocessing.qmd). The identifiers
# refer to unpublished texts and are therefore read from a local file that is not tracked
# (data/segment_exclude.csv with the columns source_id, generator).
def _segment_exclude() -> list[tuple[str, str]]:
    f = Path(__file__).resolve().parent / "data" / "segment_exclude.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f, dtype=str)
    return list(zip(d.source_id, d.generator))


SEGMENT_EXCLUDE = _segment_exclude()


def load_journalism_segments(with_headline: bool = True) -> pd.DataFrame:
    """The journalism items exactly as presented in the segment experiment.

    Union of the 32 PsychoPy conditions files (ABAB/BABA x list A/B x 8
    variants), deduplicated by trial_id. Participants saw headline +
    subheadline above every segment; with_headline=True reproduces that in the
    model input (text), while `segment` holds the body only (for features).
    """
    seg_dir = config.JOURNALISM_XLSX.parent
    files = sorted(seg_dir.glob("segment_trials_ABAB_*.xlsx")) + \
        sorted(seg_dir.glob("segment_trials_BABA_*.xlsx"))
    if len(files) != 32:
        print(f"warning: expected 32 segment conditions files, found {len(files)}")
    cols = ["trial_id", "source_id", "seg_idx", "headline", "subheadline",
            "generator", "prompt_type", "chat_message", "is_ai"]
    raw = pd.concat([pd.read_excel(f, usecols=cols) for f in files], ignore_index=True)
    # the same trial_id must carry the same content in every file
    n_variants = raw.groupby("trial_id")[["chat_message", "headline"]].nunique().max().max()
    if n_variants > 1:
        raise ValueError("A trial_id has differing content across conditions files.")
    df = raw.drop_duplicates("trial_id").copy()
    for src, gen in SEGMENT_EXCLUDE:
        df = df[~((df["source_id"] == src) & (df["generator"] == gen))]

    def compose(r) -> str:
        parts = [_clean(r["headline"]), _clean(r["subheadline"])] if with_headline else []
        head = "\n".join(p for p in parts if p)
        body = _clean(r["chat_message"]) or ""
        return f"{head}\n\n{body}" if head else body

    out = pd.DataFrame(dict(
        text_id=df["trial_id"].astype(str),
        source_id=df["source_id"].astype(str),
        generator=df["generator"].astype(str),
        gen_model=np.where(df["is_ai"].astype(int) == 1, "openai", ""),
        prompt_type=df["prompt_type"].fillna("human").replace("", "human").astype(str),
        is_ai=df["is_ai"].astype(int),
        seg_idx=df["seg_idx"].astype(int),
        text=df.apply(compose, axis=1),
        segment=df["chat_message"].map(lambda x: _clean(x) or ""),
    ))
    out.loc[out["is_ai"] == 0, "prompt_type"] = "human"
    # Diagnostic without showing text: do human and AI items of a source share the headline?
    h = df.assign(ai=df["is_ai"].astype(int)).groupby(["source_id", "ai"])["headline"].first().unstack()
    same = (h[0] == h[1]).sum() if {0, 1} <= set(h.columns) else 0
    print(f"headline identical for human and AI version in {same}/{len(h)} sources")
    return out.reset_index(drop=True)


def _finish(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out.insert(0, "text_id", out["source_id"] + "__" + out["generator"] + "__"
               + out["gen_model"].replace("", "human"))
    if out["text_id"].duplicated().any():
        raise ValueError("Duplicate text_ids — check the source tables.")
    return out


def _word_spans(text: str) -> list[re.Match]:
    return list(re.finditer(r"\S+", text))


def truncate_words(df: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for r in df.to_dict("records"):
        w = _word_spans(r["text"])
        if len(w) >= n:
            rows.append(r | dict(text=r["text"][:w[n - 1].end()]))
    return pd.DataFrame(rows)


def windows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for r in df.to_dict("records"):
        w = _word_spans(r["text"])
        for k in range(len(w) // n):
            rows.append(r | dict(text_id=f"{r['text_id']}__w{k:02d}", parent_id=r["text_id"],
                                 window=k, text=r["text"][w[k * n].start():w[k * n + n - 1].end()]))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=["chat", "journalism", "journalism_seg", "both"],
                    default="both",
                    help="journalism_seg = the segment items exactly as shown to participants")
    ap.add_argument("--no-headline", action="store_true",
                    help="journalism_seg: give the model the segment body only")
    ap.add_argument("--all-generators", action="store_true",
                    help="include generators/models not used in the human experiment")
    lc = ap.add_mutually_exclusive_group()
    lc.add_argument("--truncate-words", type=int, metavar="N",
                    help="keep the first N words; drop shorter texts")
    lc.add_argument("--windows", type=int, metavar="N",
                    help="non-overlapping N-word windows per text")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    loaders = {"chat": load_chats, "journalism": load_journalism,
               "journalism_seg": lambda _: load_journalism_segments(not args.no_headline)}
    corpora = ["chat", "journalism"] if args.corpus == "both" else [args.corpus]
    suffix = "_all" if args.all_generators else ""
    if args.corpus == "journalism_seg" and args.no_headline:
        suffix += "_nohead"
    if args.truncate_words:
        suffix += f"_tw{args.truncate_words}"
    elif args.windows:
        suffix += f"_win{args.windows}"
    for corpus in corpora:
        df = loaders[corpus](args.all_generators)
        n_before = df.groupby("prompt_type").size().to_dict()
        if args.truncate_words:
            df = truncate_words(df, args.truncate_words)
            print(f"{corpus}: truncated to {args.truncate_words} words; texts per type "
                  f"before -> after: {n_before} -> {df.groupby('prompt_type').size().to_dict()}")
        elif args.windows:
            df = windows(df, args.windows)
            print(f"{corpus}: {args.windows}-word windows per type: "
                  f"{df.groupby('prompt_type').size().to_dict()} (from {n_before} texts)")
        path = config.DATA_DIR / f"texts_{corpus}{suffix}.parquet"
        df.to_parquet(path, index=False)
        counts = df.groupby(["is_ai", "prompt_type"]).size().to_dict()
        print(f"{corpus}: {len(df)} texts, {df['source_id'].nunique()} sources "
              f"-> {path.relative_to(config.REPO)}\n  counts (is_ai, prompt_type): {counts}")


if __name__ == "__main__":
    main()
