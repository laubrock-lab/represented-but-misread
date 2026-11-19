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
    # Qwen3 (Apr 2025): standard dense transformer (Qwen3ForCausalLM, 36 layers),
    # base + instruct published, Apache 2.0, not gated. The instruct model is a
    # hybrid reasoner; judge_logits passes enable_thinking=False on the chat
    # template, and the raw completion prompt is unaffected by thinking mode.
    "qwen3-8b":   {"base": "Qwen/Qwen3-8B-Base",      "instruct": "Qwen/Qwen3-8B"},
    # Families with published post-training stages (base -> SFT -> DPO -> final):
    # which stage changes sensitivity and criterion? OLMo 2 is mostly English;
    # Apertus 1.5 covers German well but is multimodal and needs the swiss-ai
    # transformers fork (see slurm/stages.sh).
    "olmo2-7b":   {"base": "allenai/OLMo-2-1124-7B", "sft": "allenai/OLMo-2-1124-7B-SFT",
                   "dpo": "allenai/OLMo-2-1124-7B-DPO", "instruct": "allenai/OLMo-2-1124-7B-Instruct"},
    "apertus15-8b": {"base": "swiss-ai/Apertus-v1.5-8B-base", "sft": "swiss-ai/Apertus-v1.5-8B-SFT",
                     "rl": "swiss-ai/Apertus-v1.5-8B-RL",  # SFT -> RLVR -> DPO -> SDPO (final)
                     "dpo": "swiss-ai/Apertus-v1.5-8B-DPO", "instruct": "swiss-ai/Apertus-v1.5-8B"},
    # OLMo 3 7B: two post-training lines from the same base (open data). Think: SFT
    # on reasoning traces -> DPO -> RLVR; Instruct: SFT on chat/instruction data ->
    # DPO -> RLVR. RL-Zero: RLVR applied directly to the base, without SFT.
    "olmo3-7b-think":    {"base": "allenai/Olmo-3-1025-7B", "sft": "allenai/Olmo-3-7B-Think-SFT",
                          "dpo": "allenai/Olmo-3-7B-Think-DPO", "instruct": "allenai/Olmo-3-7B-Think"},
    "olmo3-7b-instruct": {"base": "allenai/Olmo-3-1025-7B", "sft": "allenai/Olmo-3-7B-Instruct-SFT",
                          "dpo": "allenai/Olmo-3-7B-Instruct-DPO", "instruct": "allenai/Olmo-3-7B-Instruct"},
    "olmo3-7b-rlzero":   {"base": "allenai/Olmo-3-1025-7B", "rl": "allenai/Olmo-3-7B-RL-Zero-Mix"},
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
    "geschrieben oder von einer KI generiert.{prior}\n\n"
    "Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Frage: Wurde dieser Text von einem Menschen geschrieben oder von einer "
    "KI generiert? Antworte mit einem Wort ({opts}).\n"
    "Antwort:"
)

# Prompt variants for the robustness check; answer words
# and scoring unchanged. "qfirst": the question precedes the text, so that the text
# positions are processed with the task in context; "para": reworded instruction and
# question. Selected with --prompt-variant (judge_logits.py, patch_positions.py).
RAW_VARIANTS = {
    "default": RAW_TEMPLATE,
    "qfirst": (
        "Im Folgenden steht ein Text. Er wurde entweder von einem Menschen "
        "geschrieben oder von einer KI generiert.{prior}\n\n"
        "Frage: Wurde der folgende Text von einem Menschen geschrieben oder von einer "
        "KI generiert? Antworte mit einem Wort ({opts}).\n\n"
        "Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Antwort:"
    ),
    "para": (
        "Der folgende Text stammt entweder von einer Person oder von einem "
        "KI-System.{prior}\n\n"
        "Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Frage: Hat ein Mensch diesen Text verfasst, oder hat eine KI ihn erzeugt? "
        "Antworte mit einem Wort ({opts}).\n"
        "Antwort:"
    ),
}


def set_prompt_variant(name: str) -> None:
    """Switch the raw template globally (build_prompt reads config.RAW_TEMPLATE)."""
    global RAW_TEMPLATE
    RAW_TEMPLATE = RAW_VARIANTS[name]


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
    "Lies den folgenden Text sorgfältig.{prior} Wurde dieser Text von einem Menschen "
    "geschrieben oder von einer KI generiert?\n\n"
    "Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Antworte ausschließlich mit einem Wort: {opts}."
)

# Base-rate instruction (key = stated percentage of AI texts). One sentence,
# identical in structure for every level, inserted into both prompt formats;
# "none" leaves the prompts unchanged. The order in which AI and humans are
# mentioned is counterbalanced (prior_order "ai" / "hu"), because the models
# react strongly to the order of mention (cf. answer-order bias).
# Bayesian prediction: dv shifts by the prior log-odds, ln(p/(1-p)), for every
# text, i.e. +/-ln 4 = +/-1.39 for 80 % / 20 % relative to 50 %; d' unchanged.
PRIORS = ["none", "50", "20", "80"]
PRIOR_ORDERS = {
    "ai": " In dieser Studie wurden etwa {ai} % der Texte von einer KI generiert "
          "und etwa {hu} % von Menschen geschrieben.",
    "hu": " In dieser Studie wurden etwa {hu} % der Texte von Menschen geschrieben "
          "und etwa {ai} % von einer KI generiert.",
}


def prior_sentence(prior: str, prior_order: str = "ai") -> str:
    if prior == "none":
        return ""
    ai = int(prior)
    return PRIOR_ORDERS[prior_order].format(ai=ai, hu=100 - ai)


# Answer strings whose probabilities are summed per class. Variants with and
# without a leading space cover both completion and chat contexts; duplicate
# token sequences are removed per tokenizer.
CANDIDATES = {
    "Mensch": ["Mensch", " Mensch"],
    "KI":     ["KI", " KI"],
}
