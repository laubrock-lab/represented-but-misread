# Figure: cross-corpus transfer as a function of relative network depth.
# Reads outputs/transfer/transfer_curve.csv (probe_transfer.py --mode curve).
#
# All four forward pairs (chat/chat_tw25 -> journalism_seg/journalism_win80),
# both sites, and all conditions are averaged per 5% depth bin; ribbons are
# 95% CIs across the individual pair x condition rows (layers within a row are
# correlated, so the band is descriptive).
#
# Usage (from the repo root):  Rscript plots/fig_transfer_depth.R
# Output: paper/figures/fig_transfer_depth.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(patchwork)
})

surface <- "#fcfcfb"; ink <- "#0b0b0b"; ink2 <- "#52514e"; gridc <- "#e6e5e0"
site_levels <- c("last", "textmean")
site_labels <- c(last = "Last token", textmean = "Text mean")
pal <- setNames(c("#2a78d6", "#eb6834"), unname(site_labels))

d <- read_csv("outputs/transfer/transfer_curve.csv", show_col_types = FALSE) |>
  filter(.data$method %in% c("probe", "diff"),
         .data$site %in% site_levels,
         .data$independent) |>
  mutate(site = factor(.data$site, levels = site_levels, labels = site_labels),
         method = factor(.data$method, levels = c("probe", "diff"),
                         labels = c("Supervised probe", "Mean difference (unsupervised)")),
         depth = (floor(.data$layer_frac * 20) + 0.5) / 20 * 100)
if (nrow(d) == 0) stop("No rows in outputs/transfer/transfer_curve.csv")

s <- d |>
  group_by(method, site, depth) |>
  summarise(m = mean(d_auc), n = n(), se = sd(d_auc) / sqrt(n()),
            lo = m - qt(0.975, n() - 1) * se,
            hi = m + qt(0.975, n() - 1) * se, .groups = "drop")

th <- theme_minimal(base_size = 10) +
  theme(plot.background  = element_rect(fill = surface, colour = NA),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = gridc, linewidth = 0.3),
        axis.text  = element_text(colour = ink2),
        axis.title = element_text(colour = ink2),
        strip.text = element_text(colour = ink, face = "bold", hjust = 0, size = 9),
        legend.text = element_text(colour = ink),
        legend.title = element_blank(),
        legend.position = "top")

plot_panel <- function(which) {
  ds <- filter(s, method == which)
  ggplot(ds, aes(depth, m, colour = site, fill = site)) +
    geom_hline(yintercept = 0, colour = ink2, linewidth = 0.3) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.7) +
    scale_colour_manual(values = pal) +
    scale_fill_manual(values = pal) +
    scale_x_continuous(breaks = seq(0, 100, 25), limits = c(0, 100),
                       expand = expansion(mult = c(0.01, 0.01))) +
    labs(x = "Relative depth (%)", y = "Transfer d′") +
    th
}

p <- (plot_panel(levels(s$method)[1]) + plot_panel(levels(s$method)[2])) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Cross-corpus transfer peaks in mid-depth layers",
    subtitle = "Mean over four forward pairs and six conditions; ribbons are 95% CIs across rows",
    theme = theme(
      plot.background = element_rect(fill = surface, colour = NA),
      plot.title = element_text(colour = ink, face = "bold", size = 11),
      plot.subtitle = element_text(colour = ink2, size = 8.5))) &
  theme(legend.position = "top")

dir.create("paper/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("paper/figures/fig_transfer_depth.png", p, width = 7.2, height = 3.6,
       dpi = 300, bg = surface)
ggsave("paper/figures/fig_transfer_depth.pdf", p, width = 7.2, height = 3.6,
       bg = surface, device = cairo_pdf)
cat("-> paper/figures/fig_transfer_depth.{png,pdf}\n")
