# Figure: cross-corpus transfer as a function of relative network depth, for
# both model families. Reads outputs/transfer/transfer_curve.csv (Gemma 3 4B/12B)
# and outputs/transfer/transfer_curve_qwen3.csv (Qwen3 8B), both produced by
# probe_transfer.py --mode curve.
#
# Only the four forward pairs (chat/chat_tw25 -> journalism_seg/journalism_win80)
# are shown; reverse pairs are confounded with authorship and are reported
# separately. Within each family, rows are averaged over all pairs and all
# conditions available (six for Gemma 3, three for Qwen3); ribbons are 95% CIs
# across the pair x condition rows (layers within a row are correlated, so the
# band is descriptive).
#
# Usage (from the repo root):  Rscript plots/fig_transfer_depth.R
# Output: paper/figures/fig_transfer_depth.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr)
})

surface <- "#fcfcfb"; ink <- "#0b0b0b"; ink2 <- "#52514e"; gridc <- "#e6e5e0"
site_labels <- c(last = "Last token", textmean = "Text mean")
pal <- setNames(c("#2a78d6", "#eb6834"), unname(site_labels))

forward_src <- c("chat", "chat_tw25")
forward_tgt <- c("journalism_seg", "journalism_win80")

read_curve <- function(path, family) {
  read_csv(path, show_col_types = FALSE) |>
    filter(.data$method %in% c("probe", "diff"),
           .data$independent,
           .data$source %in% forward_src,
           .data$target %in% forward_tgt) |>
    mutate(family = family,
           site = factor(.data$site, levels = names(site_labels),
                         labels = unname(site_labels)),
           method = factor(.data$method, levels = c("probe", "diff"),
                           labels = c("Supervised probe",
                                      "Mean difference (unsupervised)")),
           depth = (floor(.data$layer_frac * 20) + 0.5) / 20 * 100)
}

d <- bind_rows(
  read_curve("outputs/transfer/transfer_curve.csv", "Gemma 3 (4B, 12B)"),
  read_curve("outputs/transfer/transfer_curve_qwen3.csv", "Qwen3 (8B)")
) |>
  mutate(family = factor(.data$family, levels = c("Gemma 3 (4B, 12B)", "Qwen3 (8B)")))

if (nrow(d) == 0) stop("No curve rows found in outputs/transfer/")

s <- d |>
  group_by(family, method, site, depth) |>
  summarise(m = mean(d_auc), n = n(),
            se = sd(d_auc) / sqrt(n()),
            lo = m - qt(0.975, n() - 1) * se,
            hi = m + qt(0.975, n() - 1) * se, .groups = "drop")

th <- theme_minimal(base_size = 10) +
  theme(plot.background  = element_rect(fill = surface, colour = NA),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = gridc, linewidth = 0.3),
        axis.text  = element_text(colour = ink2),
        axis.title = element_text(colour = ink2),
        strip.text = element_text(colour = ink, face = "bold", hjust = 0, size = 9),
        strip.background = element_blank(),
        panel.spacing.x = unit(1.4, "lines"),
        panel.spacing.y = unit(0.8, "lines"),
        legend.text = element_text(colour = ink),
        legend.title = element_blank(),
        legend.position = "top")

p <- ggplot(s, aes(depth, m, colour = site, fill = site)) +
  geom_hline(yintercept = 0, colour = ink2, linewidth = 0.3) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.13, colour = NA) +
  geom_line(linewidth = 0.7) +
  facet_grid(family ~ method) +
  scale_colour_manual(values = pal) +
  scale_fill_manual(values = pal) +
  scale_x_continuous(breaks = seq(0, 100, 25), limits = c(0, 100),
                     expand = expansion(mult = c(0.01, 0.01))) +
  labs(x = "Relative depth (%)", y = "Transfer d\u2032") +
  th

dir.create("paper/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("paper/figures/fig_transfer_depth.png", p, width = 7.2, height = 5.2,
       dpi = 300, bg = surface)
ggsave("paper/figures/fig_transfer_depth.pdf", p, width = 7.2, height = 5.2,
       bg = surface, device = cairo_pdf)
cat("-> paper/figures/fig_transfer_depth.{png,pdf}\n")
