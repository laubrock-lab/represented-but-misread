# Export item-level aggregates of the human judgments (lab course chat experiment).
#
# Reads the preprocessed trial table from the internal lab course repository and
# writes one row per text to data/human_items_chat.csv (gitignored):
#   source_id, generator, gen_model, n, p_ki, mean_conf, signed_conf
# signed_conf = mean of (+confidence if "KI", -confidence if "Mensch"), a graded
# human judgment on a -100..100 scale. Human originals were shown in both
# model groups and are pooled per source_id. No text content is read or written.
#
# Usage:  Rscript export_human_items.R   (LABCOURSE_DIR overrides the repo location)

suppressPackageStartupMessages(library(dplyr))

labcourse <- Sys.getenv("LABCOURSE_DIR",
                     "/path/to/labcourse/experiment_v1")
trials <- readRDS(file.path(labcourse, "analysis", "preprocessed", "chat_trials.rds")) |>
  select(participant, source_id, generator, model_group, choice, confidence)

items <- trials |>
  filter(choice %in% c("KI", "Mensch")) |>
  mutate(
    gen_model = case_when(
      generator == "human_text" ~ "",
      model_group == "gpt"      ~ "gpt-5.4",
      model_group == "claude"   ~ "claude-opus-4-6"
    ),
    ki = as.integer(choice == "KI"),
    signed = ifelse(ki == 1, confidence, -confidence)
  ) |>
  group_by(source_id, generator, gen_model) |>
  summarise(n = n(), n_participants = n_distinct(participant),
            p_ki = mean(ki), mean_conf = mean(confidence, na.rm = TRUE),
            signed_conf = mean(signed, na.rm = TRUE), .groups = "drop")

dir.create("data", showWarnings = FALSE)
write.csv(items, "data/human_items_chat.csv", row.names = FALSE)
cat(nrow(items), "items;", sum(items$n), "judgments; median n per item:",
    median(items$n), "\n")
print(items |> group_by(generator) |>
        summarise(items = n(), p_ki = round(mean(p_ki), 3), .groups = "drop"))
