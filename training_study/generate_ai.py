"""Training study, step 2: AI-generated counterparts for the GNAD10 items (cluster, GPU).

Each item is assigned one generator (alternating Qwen3 8B / Gemma 3 12B, instruct
versions). The model writes a German news text for an Austrian daily on the item's topic
(first sentence of the original article) and section, somewhat longer than the target
length; the output is truncated to the same number of words as the paired human text.

Usage (cluster):  python training_study/generate_ai.py --generator qwen3-8b
                  python training_study/generate_ai.py --generator gemma3-12b
Output: data/training_study/ai_<generator>.parquet (item_id, ai_text)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402
from judge_logits import load_model, pick_device  # noqa: E402

DATA = REPO / "data" / "training_study"
PROMPT = ("Schreibe einen deutschsprachigen Nachrichtentext für eine österreichische "
          "Tageszeitung, Rubrik {section}, mit etwa {words} Wörtern. Thema: {topic}\n"
          "Gib nur den Fließtext aus, ohne Überschrift und ohne Kommentar.")


def first_words(text: str, n: int) -> str | None:
    spans = list(re.finditer(r"\S+", text))
    return text[:spans[n - 1].end()] if len(spans) >= n else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", required=True, choices=["qwen3-8b", "gemma3-12b"])
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    items = pd.read_parquet(DATA / "items.parquet")
    items = items[items.generator == a.generator].reset_index(drop=True)
    if a.limit:
        items = items.head(a.limit)
    device = pick_device(a.device)
    tok, model = load_model(config.MODEL_PRESETS[a.generator]["instruct"], device, torch.bfloat16)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(1)
    out = []
    for start in range(0, len(items), a.batch):
        b = items.iloc[start:start + a.batch]
        prompts = []
        for r in b.itertuples(index=False):
            msgs = [{"role": "user", "content": PROMPT.format(section=r.section, words=int(r.length * 1.4) + 20,
                                                               topic=r.topic)}]
            try:
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False))
            except TypeError:
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=int(b.length.max() * 2.6) + 60, do_sample=True,
                               temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id,
                               remove_invalid_values=True)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(b.itertuples(index=False), texts):
            t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
            out.append(dict(item_id=r.item_id, generator=a.generator, ai_full=t,
                            ai_text=first_words(t, int(r.length))))
        print(f"{start + len(b)}/{len(items)}", flush=True)
    df = pd.DataFrame(out)
    print(f"too short (dropped later): {df.ai_text.isna().sum()} of {len(df)}")
    df.to_parquet(DATA / f"ai_{a.generator}{'_smoke' if a.limit else ''}.parquet")


if __name__ == "__main__":
    main()
