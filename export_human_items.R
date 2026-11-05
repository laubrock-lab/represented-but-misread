# Export item-level aggregates of the human judgments (lab course experiments).
#
# Reads the preprocessed trial tables from the internal lab course repository and
# writes one row per item to data/human_items_<corpus>.csv (gitignored):
#   chat:            source_id, generator, gen_model, n, n_participants, p_ki,
#                    mean_conf, signed_conf
#   journalism_seg:  text_id (= trial_id), source_id, generator, seg_idx, ...
# signed_conf = mean of (+confidence if "KI", -confidence if "Mensch"), a graded
# human judgment on a -100..100 scale. In the chat experiment, human originals
# were shown in both model groups and are pooled per source_id.
# Only non-text columns are selected; no text content is read or written.
#
# Usage:  Rscript export_human_items.R [chat|journalism_seg]
#         (LABCOURSE_DIR overrides the repository location)

suppressPackageStartupMessages(library(dplyr))

corpus <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(corpus)) corpus <- "chat"
labcourse <- Sys.getenv("LABCOURSE_DIR",
                     "/path/to/labcourse/experiment_v1")
pre <- file.path(labcourse, "analysis", "preprocessed")

summarise_items <- function(d, ...) {
  d |>
    filter(choice %in% c("KI", "Mensch")) |>
    mutate(ki = as.integer(choice == "KI"),
           signed = ifelse(ki == 1, confidence, -confidence)) |>
    group_by(...) |>
    summarise(n = n(), n_participants = n_distinct(participant),
              p_ki = mean(ki), mean_conf = mean(confidence, na.rm = TRUE),
              signed_conf = mean(signed, na.rm = TRUE), .groups = "drop")
}

if (corpus == "chat") {
  items <- readRDS(file.path(pre, "chat_trials.rds")) |>
    select(participant, source_id, generator, model_group, choice, confidence) |>
    mutate(gen_model = case_when(
      generator == "human_text" ~ "",
      model_group == "gpt"      ~ "gpt-5.4",
      model_group == "claude"   ~ "claude-opus-4-6")) |>
    summarise_items(source_id, generator, gen_model)
} else if (corpus == "journalism_seg") {
  items <- readRDS(file.path(pre, "journ_trials.rds")) |>
    select(participant, trial_id, source_id, generator, seg_idx, choice, confidence) |>
    # same exclusion as in preprocessing.qmd
    filter(!(source_id == "<excluded-item-id>" & generator == "openai_naive")) |>
    summarise_items(trial_id, source_id, generator, seg_idx) |>
    rename(text_id = trial_id)
} else {
  stop("corpus must be 'chat' or 'journalism_seg'")
}

dir.create("data", showWarnings = FALSE)
out <- file.path("data", paste0("human_items_", corpus, ".csv"))
write.csv(items, out, row.names = FALSE)
cat(out, ":", nrow(items), "items;", sum(items$n), "judgments; median n per item:",
    median(items$n), "\n")
print(items |> group_by(generator) |>
        summarise(items = n(), p_ki = round(mean(p_ki), 3), .groups = "drop"))
