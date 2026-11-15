"""Cue-level lens analysis of the authorship judgments on the chat messages.

Which text properties drive the models' (reversed) judgments, and does post-training
change their weighting? Do the models use the properties that the students flagged as
AI-like (polish) -- and with which sign?

Items: the chat corpus (human, naive, optimized; outputs/chat_tw25) plus the
style-matched messages without feedback revision (outputs/chat_sm_tw25), all truncated
to 25 words, generators Claude Opus 4.6 and GPT-5.4.

Cues (two sets):
  labcourse  the lab course text features (feature_extractor.py; without the LM-based ones):
          length, sentence length, lexical diversity, readability, word frequency,
          part-of-speech shares, syntactic depth, semantic coherence, ...
  polish  derived from the students' feedback on what sounded AI-like:
          dashes, lower-case sentence starts, doubled punctuation ("??", ".."),
          exclamation and question marks, share of English words, mean word frequency
          (elevated vocabulary), share of long words

Judges: the answers (dv) of every model condition (preset x stage x format), and on the
cluster additionally out-of-fold probe scores from the stored activations (last prompt
token, ~67 % depth, GroupKFold by source) = the stage-1 "judge" of the two-lens model.

Per judge: lens model equation (lens_opspam.lme: ra, Re, Rs, G, C, cos_w), and per cue
the validity r(cue, is_ai) and the utilization r(cue, judgment) (Spearman).

Usage
-----
    python cue_lens_chats.py features    # LOCAL (lab course code, spaCy): writes cue table, no text
    python cue_lens_chats.py analyze [--probes]   # anywhere; --probes needs the activations (cluster)
Outputs: outputs/chat_cues/cues_chat_tw25_all.csv (cues per text_id, no text),
         outputs/chat_cues/lens_judges.csv, outputs/chat_cues/cue_profiles.csv
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config

OUT = config.OUTPUT_DIR / "chat_cues"
LABCOURSE = ["word_count", "sent_count", "mean_sent_len_tokens", "std_sent_len_tokens", "burstiness_cv",
          "ttr", "mtld", "punct_density", "mean_word_len", "hapax_ratio", "flesch_reading_ease_de",
          "wiener_sachtextformel", "pos_noun_ratio", "pos_verb_ratio", "pos_adj_ratio", "pos_adv_ratio",
          "pos_propn_ratio", "ne_density", "dep_depth_mean", "dep_depth_max", "dep_depth_std",
          "embedding_variance", "mean_adjacent_cosine_sim", "global_coherence", "noun_overlap_ratio",
          "mean_word_freq_zipf", "std_word_freq_zipf"]
POLISH = ["dash_rate", "lowercase_start_share", "doubled_punct_rate", "exclam_rate", "question_rate",
          "english_word_share", "long_word_share"]
STAGES = ["base", "sft", "rl", "dpo", "instruct"]


# --------------------------------------------------------------------------- features (local)
def polish_cues(text: str) -> dict:
    from wordfreq import zipf_frequency
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", text)
    nw = max(len(words), 1)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    starts = [s for s in sents if s[0].isalpha()]
    eng = [w for w in words if len(w) > 2 and zipf_frequency(w.lower(), "en") > zipf_frequency(w.lower(), "de") + 1.0]
    return dict(dash_rate=len(re.findall(r"[–—]|\s-\s", text)) / nw,
                lowercase_start_share=(sum(s[0].islower() for s in starts) / len(starts)) if starts else 0.0,
                doubled_punct_rate=len(re.findall(r"[!?.]{2,}", text)) / nw,
                exclam_rate=text.count("!") / nw, question_rate=text.count("?") / nw,
                english_word_share=len(eng) / nw, long_word_share=sum(len(w) > 12 for w in words) / nw)


def features() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = config.REPO / "data" / "texts_chat_all_tw25.parquet"
    subprocess.run([sys.executable, "segment_features.py", str(src), "--out-dir", str(OUT)], check=True)
    f = pd.read_csv(OUT / "features_chat_all_tw25.csv")
    t = pd.read_parquet(src, columns=["text_id", "text"])
    pol = pd.DataFrame([dict(text_id=i) | polish_cues(x) for i, x in zip(t.text_id, t.text)])
    f = f.merge(pol, on="text_id")
    f.to_csv(OUT / "cues_chat_tw25_all.csv", index=False)      # text_id, keys, cues -- no text
    print(f"{len(f)} items x {len(LABCOURSE) + len(POLISH)} cues -> {OUT / 'cues_chat_tw25_all.csv'}")


# --------------------------------------------------------------------------- analysis
def answers(corpus, preset, variant, fmt):
    sfx = "_neutral" if fmt == "chat" else ""
    fs = [config.OUTPUT_DIR / corpus / f"{preset}_{variant}_{fmt}_{o}{sfx}.csv" for o in ("mk", "km")]
    if not all(f.exists() for f in fs):
        return None
    d = pd.concat([pd.read_csv(f, usecols=["text_id", "dv"]) for f in fs])
    n = d.groupby("text_id").size()
    return d[d.text_id.isin(n[n == 2].index)].groupby("text_id").dv.mean()


def probe_scores(preset, variant, items: pd.DataFrame) -> pd.Series | None:
    """OOF probe scores at ~67 % depth from the stored activations (main corpus only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    f = config.OUTPUT_DIR / "chat_tw25" / f"{preset}_{variant}_raw_mk_acts.npz"
    if not f.exists():
        return None
    z = np.load(f)
    A = z["acts_last"]
    layer = round(.67 * (A.shape[1] - 1))
    ids = z["text_id"].astype(str)
    y, g = z["is_ai"], z["source_id"].astype(str)
    s = np.zeros(len(ids))
    for tr, te in GroupKFold(10).split(A, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"))
        s[te] = m.fit(A[tr, layer], y[tr]).decision_function(A[te, layer])
    return pd.Series(s, index=ids)


def analyze(with_probes: bool) -> None:
    from lens_opspam import lme
    cues = pd.read_csv(OUT / "cues_chat_tw25_all.csv")
    cues = cues[(cues.gen_model.isin(["gpt-5.4", "claude-opus-4-6"])) | (cues.is_ai == 0)]
    names = [c for c in LABCOURSE + POLISH if c in cues.columns and cues[c].std() > 0]
    judges = {}
    for preset in config.MODEL_PRESETS:
        for v in STAGES:
            for fmt in ("raw", "chat"):
                a, b = answers("chat_tw25", preset, v, fmt), answers("chat_sm_tw25", preset, v, fmt)
                if a is not None:
                    judges[f"answer|{preset}|{v}|{fmt}"] = pd.concat([a, b]) if b is not None else a
                    judges[f"answer300|{preset}|{v}|{fmt}"] = a     # same 300 items as the probes
            if with_probes:
                p = probe_scores(preset, v, cues)
                if p is not None:
                    judges[f"probe|{preset}|{v}|raw"] = p
    rows, prof = [], []
    for name, J in judges.items():
        d = cues.set_index("text_id").join(J.rename("J"), how="inner").dropna(subset=["J"])
        X = d[names].to_numpy(float)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        y, j = d.is_ai.to_numpy(float), d.J.to_numpy(float)
        r, _, _ = lme(y, j, X, d.source_id.to_numpy())
        kind, preset, v, fmt = name.split("|")
        rows.append(dict(kind=kind, preset=preset, variant=v, fmt=fmt, n=len(d)) | r)
        for k, c in enumerate(names):
            prof.append(dict(kind=kind, preset=preset, variant=v, fmt=fmt, cue=c,
                             cue_set="polish" if c in POLISH else "labcourse",
                             validity=spearmanr(X[:, k], y)[0], utilization=spearmanr(X[:, k], j)[0]))
    res, prof = pd.DataFrame(rows), pd.DataFrame(prof)
    res.to_csv(OUT / "lens_judges.csv", index=False)
    prof.to_csv(OUT / "cue_profiles.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.2f}".format):
        print(res[["kind", "preset", "variant", "fmt", "n", "ra", "Re", "Re_cv", "Rs", "Rs_cv", "G", "G_cv", "C"]]
              .to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["features", "analyze"])
    ap.add_argument("--probes", action="store_true", help="add probe judges (needs activations)")
    a = ap.parse_args()
    features() if a.mode == "features" else analyze(a.probes)


if __name__ == "__main__":
    main()
