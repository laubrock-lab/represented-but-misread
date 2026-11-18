# Authorship: layer-by-position patching on the textbook summaries (Section 3.4).
# d' of the recipient checkpoint after its residual stream was replaced by the donor's at one
# set of positions and one layer; base vs. SFT, both directions, OLMo 2 7B and Apertus 1.5 8B
# (outputs/patch_sweep/patch_sweep_summary.csv, patch_sweep_summary.py).
# Shown is d' itself (not the share transferred): the difference between the checkpoints is
# small here (about 0.4-0.5), and hybrids can lie outside the range of the two checkpoints.
# Usage: Rscript plots/fig_authorship_sweep.R -> paper/figures/fig_authorship_sweep.{pdf,png}
suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(readr) })
BLUE <- "#3B6FB6"; ORANGE <- "#E08A2B"; GREY <- "grey45"
fam <- c("olmo2-7b" = "OLMo 2 7B", "apertus15-8b" = "Apertus 1.5 8B")
sets <- c(text = "text", post = "after the text", last = "final position", all = "all positions")
s <- read_csv("outputs/patch_sweep/patch_sweep_summary.csv", show_col_types = FALSE) |>
  filter(corpus == "textbook", preset %in% names(fam), set %in% names(sets),
         is.na(variant) | variant %in% c("", "default")) |>
  mutate(family = fam[preset], set = factor(sets[set], levels = sets),
         rec = ifelse(recipient == "base", "base model with SFT states", "SFT model with base states"))
ref <- s |> distinct(family, rec, d_recipient, d_donor) |>
  tidyr::pivot_longer(c(d_recipient, d_donor), names_to = "which", values_to = "d") |>
  mutate(which = ifelse(which == "d_recipient", "unpatched recipient", "donor"))
p <- ggplot(s, aes(layer, d, colour = set)) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
  geom_hline(data = ref, aes(yintercept = d, linetype = which), colour = GREY, linewidth = 0.4) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.3, alpha = 0.45, position = position_dodge(width = 1.2)) +
  geom_line(linewidth = 0.5, position = position_dodge(width = 1.2)) +
  geom_point(size = 1.1, position = position_dodge(width = 1.2)) +
  facet_grid(family ~ rec, scales = "free_y") +
  scale_colour_manual(values = c("text" = "#7A3E9D", "after the text" = ORANGE, "final position" = "#2A9D8F", "all positions" = "grey25"),
                      name = "states exchanged at") +
  scale_linetype_manual(values = c("unpatched recipient" = "dotted", "donor" = "dashed"), name = NULL) +
  scale_x_continuous(breaks = c(2, 6, 10, 14, 20, 24, 28, 32)) +
  labs(x = "layer at which states are exchanged", y = "d′ (answers)") +
  theme_minimal(base_size = 8.5) +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom", legend.box = "vertical",
        legend.key.height = unit(3, "mm"), strip.text = element_text(face = "bold"))
f <- "paper/figures/fig_authorship_sweep.pdf"
ggsave(f, p, width = 7, height = 5.6, device = cairo_pdf)   # cairo: the prime in d′ is outside Latin-1
ggsave(sub("pdf$", "png", f), p, width = 7, height = 5.6, dpi = 200, bg = "white")
