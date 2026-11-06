# Figure: represented vs. said, and alignment of the answer with the class information,
# over depth. Reads outputs/alignment_profile.csv (from alignment_profile.py).
#
# Usage (from the repo root):  Rscript plots/fig_alignment.R [gemma3-4b|gemma3-12b]
# Output: outputs/figures/fig_alignment_<preset>.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(patchwork); library(ggrepel)
})

preset <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(preset)) preset <- "gemma3-4b"

# --- tokens (validated reference palette, first three categorical slots; light surface)
surface <- "#fcfcfb"; ink <- "#0b0b0b"; ink2 <- "#52514e"; gridc <- "#e6e5e0"
cond_levels <- c("base", "instruct", "instruct, chat template")
pal <- setNames(c("#2a78d6", "#eb6834", "#1baf7a"), cond_levels)

corpus_labels <- c(
  chat             = "Chat messages, full length (length confounded)",
  chat_tw25        = "Chat messages, truncated to 25 words",
  journalism_win80 = "Journalism, 80-word windows",
  journalism_seg   = "Journalism, segments as shown to participants"
)

d <- read_csv("outputs/alignment_profile.csv", show_col_types = FALSE) |>
  filter(.data$preset == !!preset) |>
  mutate(cond = factor(case_when(variant == "base" ~ cond_levels[1],
                                 fmt == "raw" ~ cond_levels[2],
                                 TRUE ~ cond_levels[3]), levels = cond_levels),
         corpus = factor(corpus, levels = names(corpus_labels), labels = corpus_labels))
if (nrow(d) == 0) stop("No rows for preset ", preset)

ref <- d |> distinct(corpus, cond, output_d)
ends <- d |> group_by(corpus, cond) |> filter(layer == max(layer)) |> ungroup() |>
  filter(corpus == levels(corpus)[2])          # direct labels in one panel only

# same x scale in both columns; extra room on the right for the direct labels
nl <- max(d$n_layers)
xscale <- scale_x_continuous(limits = c(1, nl * 1.25), breaks = pretty(c(0, nl), 4),
                             expand = expansion(mult = c(0.02, 0)))

th <- theme_minimal(base_size = 10) +
  theme(plot.background  = element_rect(fill = surface, colour = NA),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = gridc, linewidth = 0.3),
        axis.text  = element_text(colour = ink2),
        axis.title = element_text(colour = ink2),
        strip.text = element_text(colour = ink, face = "bold", hjust = 0, size = 9),
        plot.title = element_text(colour = ink, face = "bold", size = 11),
        plot.subtitle = element_text(colour = ink2, size = 8.5),
        legend.text = element_text(colour = ink),
        legend.position = "top")

p1 <- ggplot(d, aes(layer, probe_d, colour = cond)) +
  geom_hline(yintercept = 0, colour = ink2, linewidth = 0.3) +
  geom_hline(data = ref, aes(yintercept = output_d, colour = cond),
             linetype = "22", linewidth = 0.5, show.legend = FALSE) +
  geom_line(linewidth = 0.7) +
  geom_point(data = ends, size = 1.8, show.legend = FALSE) +
  geom_text_repel(data = ends, aes(label = cond), colour = ink, size = 2.8, show.legend = FALSE,
                  nudge_x = 2, direction = "y", hjust = 0, segment.colour = gridc,
                  min.segment.length = 0, seed = 1) +
  facet_wrap(~corpus, ncol = 1) +
  scale_colour_manual(values = pal, name = NULL, drop = FALSE) +
  xscale +
  labs(title = "Represented vs. said",
       subtitle = "solid: linear-probe d′ at the answer position (cross-validated)\ndashed: d′ of the model’s own answer",
       x = "layer", y = "d′") +
  th

p2 <- ggplot(d, aes(layer, alignment, colour = cond)) +
  geom_hline(yintercept = 0, colour = ink2, linewidth = 0.3) +
  geom_line(linewidth = 0.7) +
  facet_wrap(~corpus, ncol = 1) +
  scale_y_continuous(limits = c(-1, 1), breaks = c(-1, -0.5, 0, 0.5, 1)) +
  xscale +
  scale_colour_manual(values = pal, name = NULL, drop = FALSE) +
  labs(title = "Does the answer follow the class information?",
       subtitle = "correlation across dimensions: class effect (AI − human) vs.\nwithin-class decision effect (rated more vs. less AI-like)",
       x = "layer", y = "alignment r") +
  th

fig <- (p1 | p2) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = paste0("Gemma 3 ", toupper(sub("gemma3-", "", preset)), ": the class is represented at the answer position, but the answer does not use it"),
    caption = "Probe d′ is capped at the resolution of the AUC (≈ 3.1–3.6). Alignment < 0: dimensions that carry the class information push the judgment the wrong way.",
    theme = theme(plot.title = element_text(colour = ink, face = "bold", size = 12),
                  plot.caption = element_text(colour = ink2, size = 8, hjust = 0),
                  plot.background = element_rect(fill = surface, colour = NA))) &
  theme(legend.position = "top")

dir.create("outputs/figures", showWarnings = FALSE, recursive = TRUE)
base <- file.path("outputs/figures", paste0("fig_alignment_", preset))
ggsave(paste0(base, ".png"), fig, width = 10, height = 10, dpi = 200, bg = surface)
ggsave(paste0(base, ".pdf"), fig, width = 10, height = 10, device = cairo_pdf, bg = surface)
cat("->", paste0(base, c(".png", ".pdf")), "\n")
