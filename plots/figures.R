# Paper figures 1-3 (data figures). Reads outputs/figdata/*.csv (from plots/fig_data.py),
# writes paper/figures/fig{1,2,3}_*.{pdf,png}.
# Colour roles (validated with the dataviz palette checker, light mode):
#   base models / base checkpoint   = blue   #2a78d6
#   post-trained / SFT checkpoint   = orange #eb6834
#   baselines, references           = ink    #52514e / grey #b8b6b0
# Usage:  Rscript plots/figures.R

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr); library(patchwork)
})

D   <- "outputs/figdata"
OUT <- "paper/figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; INK <- "#52514e"; GREY <- "#b8b6b0"
corpus_levels <- c("Chats (25 words)", "News segments", "Textbook summaries")
family_levels <- c("Gemma 3 4B", "Gemma 3 12B", "Qwen3 8B", "OLMo 2 7B", "Apertus 1.5 8B", "OLMo 3 7B Think")

theme_paper <- function(base = 8.5) {
  theme_minimal(base_size = base) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(colour = "#e6e5e1", linewidth = 0.3),
          strip.text = element_text(face = "bold", hjust = 0),
          axis.title = element_text(colour = INK),
          axis.text = element_text(colour = INK),
          legend.position = "bottom", legend.title = element_blank(),
          plot.tag = element_text(face = "bold"))
}
chance <- function() geom_vline(xintercept = 0, colour = GREY, linewidth = 0.4)
save_fig <- function(p, name, w, h) {
  ggsave(file.path(OUT, paste0(name, ".pdf")), p, width = w, height = h, device = cairo_pdf)
  ggsave(file.path(OUT, paste0(name, ".png")), p, width = w, height = h, dpi = 200, bg = "white")
}

# ---------------------------------------------------------------- Figure 1
f1 <- read_csv(file.path(D, "fig1_info.csv"), show_col_types = FALSE) |>
  mutate(corpus = factor(corpus, corpus_levels),
         group = factor(group, c("Text and representation", "Model answers")),
         measure = factor(measure, rev(c("Word n-grams", "Character n-grams", "Surface statistics",
                                         "Input embedding (probe)", "Model activations (probe)",
                                         "EditLens (trained detector)",
                                         "Answers: base models", "Answers: post-trained models"))))
info <- filter(f1, group == "Text and representation")
ans  <- filter(f1, group == "Model answers") |>
  mutate(stage = ifelse(grepl("base", measure), "Base models", "Post-trained models"))
ans_med <- ans |> group_by(corpus, group, measure, stage) |> summarise(d = median(d), .groups = "drop")
ceil <- distinct(f1, corpus, ceiling)

p1 <- ggplot() +
  chance() +
  geom_vline(data = ceil, aes(xintercept = ceiling), colour = GREY, linetype = "dashed", linewidth = 0.3) +
  geom_text(data = filter(ceil, corpus == corpus_levels[1]) |> mutate(group = factor("Text and representation", levels(f1$group))),
            aes(x = ceiling, y = Inf, label = "ceiling"), colour = INK, size = 2.3, vjust = 1.4, hjust = 1.1) +
  geom_errorbar(data = info, aes(y = measure, xmin = lo, xmax = hi), width = 0, colour = INK, linewidth = 0.5,
                orientation = "y") +
  geom_point(data = info, aes(y = measure, x = d), colour = INK, size = 1.8) +
  geom_point(data = ans, aes(y = measure, x = d, colour = stage), size = 1.4, alpha = 0.6,
             position = position_jitter(height = 0.15, width = 0, seed = 1)) +
  geom_point(data = ans_med, aes(y = measure, x = d, fill = stage), shape = 21, colour = "white",
             size = 3, stroke = 0.6) +
  scale_colour_manual(values = c("Base models" = BLUE, "Post-trained models" = ORANGE)) +
  scale_fill_manual(values = c("Base models" = BLUE, "Post-trained models" = ORANGE), guide = "none") +
  facet_grid(group ~ corpus, scales = "free_y", space = "free_y") +
  labs(x = "Sensitivity d′ (AI vs. human texts)", y = NULL) +
  theme_paper() + theme(strip.text.y = element_blank())
save_fig(p1, "fig1_information_vs_answers", 7, 3.6)

# ---------------------------------------------------------------- Figure 2
m <- read_csv(file.path(D, "fig2_models.csv"), show_col_types = FALSE) |>
  filter(in_format) |>
  mutate(corpus = factor(corpus, corpus_levels), family = factor(family, rev(family_levels)))
con <- read_csv(file.path(D, "fig2_contrasts.csv"), show_col_types = FALSE) |>
  mutate(corpus = factor(corpus, corpus_levels), family = factor(family, rev(family_levels)),
         star = ifelse(p_holm < .05, "*", ""))
wide <- m |> filter(fmt == "raw") |> select(corpus, family, stage, d) |>
  pivot_wider(names_from = stage, values_from = d)
pts <- m |> mutate(key = case_when(stage == "Base" ~ "Base (raw prompt)",
                                   fmt == "raw" ~ "Post-trained (raw prompt)",
                                   TRUE ~ "Post-trained (chat prompt)"))
p2a <- ggplot() +
  chance() +
  geom_segment(data = wide, aes(y = family, yend = family, x = Base, xend = `Post-trained`),
               colour = GREY, linewidth = 0.6) +
  geom_point(data = pts, aes(y = family, x = d, colour = key, shape = key), size = 2.1, stroke = 0.7) +
  geom_text(data = filter(con, star != ""), aes(y = family, x = 2.05, label = star),
            vjust = 0.75, size = 3.5, colour = INK) +
  scale_x_continuous(limits = c(-1.4, 2.2), breaks = c(-1, 0, 1, 2)) +
  scale_colour_manual(values = c("Base (raw prompt)" = BLUE, "Post-trained (raw prompt)" = ORANGE,
                                 "Post-trained (chat prompt)" = ORANGE)) +
  scale_shape_manual(values = c("Base (raw prompt)" = 16, "Post-trained (raw prompt)" = 16,
                                "Post-trained (chat prompt)" = 1)) +
  facet_wrap(~ corpus, nrow = 1) +
  labs(x = "Sensitivity d′ of the answers", y = NULL) +
  theme_paper()

fs <- read_csv(file.path(D, "fig2_fewshot.csv"), show_col_types = FALSE) |>
  mutate(corpus = factor(corpus, corpus_levels), family = factor(family, family_levels),
         kf = factor(k, c(0, 2, 8, 16)))
p2b <- ggplot(fs, aes(x = kf, y = d, colour = stage, group = stage)) +
  geom_hline(yintercept = 0, colour = GREY, linewidth = 0.4) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.4, alpha = 0.6, na.rm = TRUE) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.6) +
  scale_colour_manual(values = c("Base" = BLUE, "Post-trained" = ORANGE),
                      labels = c("Base model", "Post-trained model")) +
  facet_grid(corpus ~ family, labeller = labeller(corpus = c("Chats (25 words)" = "Chats",
                                                             "News segments" = "News",
                                                             "Textbook summaries" = "Textbook"))) +
  labs(x = "Number of labelled examples in the prompt", y = "Sensitivity d′") +
  theme_paper() + theme(panel.grid.major.y = element_line(colour = "#e6e5e1", linewidth = 0.3),
                        panel.grid.major.x = element_blank())
p2 <- p2a / p2b + plot_layout(heights = c(1.15, 1)) + plot_annotation(tag_levels = "A")
save_fig(p2, "fig2_answers_and_fewshot", 7, 6.4)

# ---------------------------------------------------------------- Figure 3
st <- read_csv(file.path(D, "fig3_stitch.csv"), show_col_types = FALSE) |>
  filter(component == "all") |>
  mutate(dir = ifelse(lower == "base", "Base layers below, SFT layers above",
                                       "SFT layers below, base layers above"))
refs <- st |> distinct(family, parent_base, parent_sft) |>
  pivot_longer(c(parent_base, parent_sft), names_to = "parent", values_to = "d") |>
  mutate(label = ifelse(parent == "parent_base", "base model", "SFT model"))
p3a <- ggplot(st, aes(x = layers, y = d, colour = dir, fill = dir)) +
  geom_hline(yintercept = 0, colour = GREY, linewidth = 0.4) +
  geom_hline(data = refs, aes(yintercept = d, linetype = label), colour = INK, linewidth = 0.3,
             inherit.aes = FALSE) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.5) +
  scale_colour_manual(values = c("Base layers below, SFT layers above" = BLUE,
                                 "SFT layers below, base layers above" = ORANGE)) +
  scale_fill_manual(values = c("Base layers below, SFT layers above" = BLUE,
                               "SFT layers below, base layers above" = ORANGE), guide = "none") +
  scale_linetype_manual(values = c("base model" = "dashed", "SFT model" = "dotted"), name = NULL) +
  scale_x_continuous(breaks = c(0, 8, 16, 24, 32)) +
  facet_wrap(~ family, nrow = 1) +
  labs(x = "Number of lower layers taken from the other checkpoint", y = "Sensitivity d′") +
  guides(colour = guide_legend(order = 1, nrow = 1), linetype = guide_legend(order = 2, nrow = 1)) +
  theme_paper() + theme(panel.grid.major.y = element_line(colour = "#e6e5e1", linewidth = 0.3),
                        legend.box = "horizontal", legend.spacing.x = unit(0.3, "in"))

pa <- read_csv(file.path(D, "fig3_patch.csv"), show_col_types = FALSE) |>
  filter(layer == 16) |>
  mutate(set = factor(set, c("none", "pre", "text", "post", "last", "all"),
                      c("unpatched", "instruction", "text", "question", "final token", "all")),
         dir = ifelse(donor == "base", "Base states into the SFT model", "SFT states into the base model"))
p3b <- ggplot(pa, aes(x = set, y = d, colour = dir)) +
  geom_hline(yintercept = 0, colour = GREY, linewidth = 0.4) +
  geom_pointrange(aes(ymin = lo, ymax = hi), position = position_dodge(width = 0.5),
                  size = 0.25, linewidth = 0.5) +
  scale_colour_manual(values = c("Base states into the SFT model" = BLUE,
                                 "SFT states into the base model" = ORANGE)) +
  facet_wrap(~ family, nrow = 1) +
  labs(x = "Positions whose layer-16 states were exchanged", y = "Sensitivity d′") +
  theme_paper() + theme(panel.grid.major.y = element_line(colour = "#e6e5e1", linewidth = 0.3),
                        panel.grid.major.x = element_blank())
p3 <- p3a / p3b + plot_annotation(tag_levels = "A")
save_fig(p3, "fig3_stitching_patching", 7, 6.2)
cat("figures written to", OUT, "\n")
