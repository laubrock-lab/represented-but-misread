# Prompt templates for the generation of the AI texts

These are the templates with which the AI counterparts of two corpora were generated
(Methods, Section 2.1). They contain placeholders only. The texts that filled the
placeholders (the students' chat messages and their feedback, the newspaper articles, the
key facts and style profiles derived from them) cannot be shared and are not part of this
repository.

## Chat messages (`chat/`)

German templates in the syntax of Python's `string.Template`; placeholders start with `$`.

| File | Step | Placeholders |
|---|---|---|
| `basic_de.txt` | basic ("naive") message | `$chat_setting`, `$topic`, `$length_hint` |
| `style_match_de.txt` | style-matched message | additionally `$style_references` (other messages of the same student, on other topics) |
| `revise_de.txt` | revision according to feedback ("optimized") | additionally `$previous_draft`, `$feedback_text` |

System prompt for all three steps:

> Du schreibst kurze, glaubwürdige deutsche Chatnachrichten für ein sozialwissenschaftliches
> Experiment. Halte dich exakt an die Vorgaben im Nutzerprompt und gib nur die Nachricht
> selbst aus.

`$length_hint` was "etwa 1 bis 4 kurze Sätze". `$chat_setting` placed the model in a private
German group chat with friends or family.

## Newspaper articles (`news/`)

English templates with a system and a user part; placeholders in braces.

| File | Step |
|---|---|
| `prompt_gemma_topic.txt` | a locally run model extracts topic summary, key facts, length and paragraph count from an article (`{TARGET_TEXT}`) |
| `prompt_gemma_style.txt` | a locally run model derives a style profile from the author's other articles (`{N}`, `{TRAINING_TEXTS}`) |
| `prompt_frontier_naive.txt` | a frontier model writes an article from topic and key facts |
| `prompt_frontier_style_matched.txt` | the same, additionally following the style profile (`{STYLE_PROFILE}`) |
| `prompt_frontier_naive_search.txt`, `prompt_frontier_style_matched_search.txt` | variants in which the model first searches the web for context |

The step in which the style-matched articles were revised to avoid surface features that
students had named as typical of AI text is described in the manuscript; its prompt
contains the students' list of features and is not included.
