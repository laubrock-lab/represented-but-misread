"""Central configuration: model presets, prompts, answer candidates, paths."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"          # gitignored — never commit texts
OUTPUT_DIR = REPO / "outputs"     # gitignored — judgments + activations

# Source materials live in the (internal) lab course repository. Override with
# the LABCOURSE_DIR environment variable if it moves.
LABCOURSE_DIR = Path(os.environ.get(
    "LABCOURSE_DIR",
    "/path/to/labcourse/experiment_v1",
))
CHATS_XLSX = LABCOURSE_DIR / "material" / "chats" / "unified" / "unified.xlsx"
JOURNALISM_XLSX = LABCOURSE_DIR / "material" / "journalism" / "combined" / "combined_w.xlsx"

# ---------------------------------------------------------------------------
# Model presets: each family as a base / instruction-tuned pair.
# Gemma and Llama are gated on Hugging Face: accept the licence on the model
# page and run `hf auth login` once.
# ---------------------------------------------------------------------------
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "gemma3-4b":  {"base": "google/gemma-3-4b-pt",    "instruct": "google/gemma-3-4b-it"},
    "gemma3-12b": {"base": "google/gemma-3-12b-pt",   "instruct": "google/gemma-3-12b-it"},
    "llama31-8b": {"base": "meta-llama/Llama-3.1-8B", "instruct": "meta-llama/Llama-3.1-8B-Instruct"},
    "qwen25-7b":  {"base": "Qwen/Qwen2.5-7B",         "instruct": "Qwen/Qwen2.5-7B-Instruct"},
}

# ---------------------------------------------------------------------------
# Prompts (German, matching the lab course LLM-judge wording where possible).
# {opts} is filled with the two answer options in the order under test.
# ---------------------------------------------------------------------------
ORDERS = {
    "mk": ("Mensch", "KI"),   # "Mensch oder KI"
    "km": ("KI", "Mensch"),   # "KI oder Mensch" — counterbalances position bias
}

# Completion-style prompt: identical for base and instruct weights, so that a
# base-vs-instruct difference cannot be due to the chat template.
RAW_TEMPLATE = (
    "Im Folgenden steht ein Text. Er wurde entweder von einem Menschen "
    "geschrieben oder von einer KI generiert.\n\n"
    "Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Frage: Wurde dieser Text von einem Menschen geschrieben oder von einer "
    "KI generiert? Antworte mit einem Wort ({opts}).\n"
    "Antwort:"
)

# Chat-style prompt (instruct models only), via the model's chat template.
SYSTEM_PROMPTS = {
    "neutral": "Du bewertest Texte.",
    "expert": (
        "Du bist ein Experte für die Erkennung von KI-generierten Texten. "
        "Du hast umfangreiches Wissen über stilistische Unterschiede zwischen "
        "menschlich verfassten und KI-generierten Texten, einschließlich Muster "
        "in Wortwahl, Satzstruktur, Kohärenz und anderen sprachlichen Merkmalen."
    ),
}
CHAT_USER_TEMPLATE = (
    "Lies den folgenden Text sorgfältig. Wurde dieser Text von einem Menschen "
    "geschrieben oder von einer KI generiert?\n\n"
    "Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Antworte ausschließlich mit einem Wort: {opts}."
)

# Answer strings whose probabilities are summed per class. Variants with and
# without a leading space cover both completion and chat contexts; duplicate
# token sequences are removed per tokenizer.
CANDIDATES = {
    "Mensch": ["Mensch", " Mensch"],
    "KI":     ["KI", " KI"],
}
