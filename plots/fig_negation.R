# Figure for Section 3.1 (negated statements): where supervised fine-tuning repairs the readout.
# (A) Model stitching, d' as a function of the number of lower layers taken from one checkpoint
#     (outputs/stitch/stitch_neg_cities.csv, localization_summary.py).
# (B) Activation patching at layers 8 and 16: d' of the recipient after receiving the donor's
#     states at one set of positions (outputs/patch/patch_summary.csv).
# (C) If present: layer-by-position sweep for OLMo 2 (outputs/patch_sweep/patch_sweep_summary.csv,
#     patch_sweep_summary.py), share of the checkpoint difference transferred.
# Usage: Rscript plots/fig_negation.R  -> paper/figures/fig_negation.{pdf,png}
suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(readr); library(patchwork) })
BLUE <- "#3B6FB6"; ORANGE <- "#E08A2B"; GREY <- "grey45"
fam <- c("olmo2-7b" = "OLMo 2 7B", "apertus15-8b" = "Apertus 1.5 8B")
th <- theme_minimal(base_size = 8.5) +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom", legend.key.height = unit(3, "mm"),
        plot.title = element_text(size = 9, face = "bold"), strip.text = element_text(face = "bold"))

st <- read_csv("outputs/stitch/stitch_neg_cities.csv", show_col_types = FALSE) |>
  filter(component == "all") |>
  mutate(family = fam[preset],
         dir = ifelse(lower == "base", "lower layers from base, upper from SFT", "lower layers from SFT, upper from base"))
par <- st |> distinct(family, d_base, d_sft) |>
  tidyr::pivot_longer(c(d_base, d_sft), names_to = "parent", values_to = "d") |>
  mutate(parent = ifelse(parent == "d_base", "base", "SFT"))
pA <- ggplot(st, aes(split, d, colour = dir)) +
  geom_hline(yintercept = 0, colour = GREY, linewidth = 0.3) +
  geom_hline(data = par, aes(yintercept = d, linetype = parent), colour = GREY, linewidth = 0.4) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.4) + geom_line(linewidth = 0.5) + geom_point(size = 1.2) +
  facet_wrap(~family) +
  scale_colour_manual(values = c(BLUE, ORANGE), name = NULL) +
  scale_linetype_manual(values = c(base = "dotted", SFT = "dashed"), name = "unstitched") +
  scale_x_continuous(breaks = c(2, 4, 6, 8, 12, 16, 24)) +
  labs(title = "A  Model stitching", x = "number of lower layers from the first-named checkpoint", y = "d′ (answers)") +
  th + guides(colour = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2))

sets <- c(text = "statement", pre = "before statement", post = "after statement", last = "final position", all = "all positions")
pa <- read_csv("outputs/patch/patch_summary.csv", show_col_types = FALSE) |>
  filter(corpus == "truth_neg_cities", prompt == "default", set %in% names(sets)) |>
  mutate(family = fam[preset], set = factor(sets[set], levels = rev(sets)),
         rec = ifelse(recipient == "base", "base model with SFT states", "SFT model with base states"),
         layer = factor(paste("layer", layer), levels = c("layer 8", "layer 16")))
ref <- pa |> distinct(family, layer, rec, d_recipient, d_donor)
pB <- ggplot(pa, aes(d, set, colour = rec)) +
  geom_vline(xintercept = 0, colour = GREY, linewidth = 0.3) +
  geom_vline(data = ref, aes(xintercept = d_recipient, colour = rec), linetype = "dotted", linewidth = 0.4) +
  geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.4, position = position_dodge(width = 0.5)) +
  geom_point(size = 1.3, position = position_dodge(width = 0.5)) +
  facet_grid(layer ~ family) +
  scale_colour_manual(values = c(BLUE, ORANGE), name = NULL) +
  labs(title = "B  Activation patching", x = "d′ after exchanging states (dotted: unpatched recipient)", y = NULL) + th

fig <- pA / pB + plot_layout(heights = c(1, 1.25)); h <- 6.4
sw <- "outputs/patch_sweep/patch_sweep_summary.csv"
if (file.exists(sw)) {
  s <- read_csv(sw, show_col_types = FALSE) |>
    filter(corpus == "truth_neg_cities", set %in% names(sets))
  fam4 <- c("apertus15-8b" = "Apertus 1.5 8B", "olmo2-7b" = "OLMo 2 7B", "gemma3-4b" = "Gemma 3 4B", "gemma3-12b" = "Gemma 3 12B")
  if ("variant" %in% names(s)) s <- filter(s, is.na(variant) | variant %in% c("", "default"))
  s <- s |> filter(preset %in% names(fam4)) |>
            mutate(family = factor(fam4[preset], levels = fam4), set = factor(sets[set], levels = rev(sets)),
                   rec = ifelse(recipient == "base", "into base", "into post-trained"),
                   share = pmin(pmax(transfer, 0), 1))
  pC <- ggplot(s, aes(factor(layer), set, fill = share)) + geom_tile(colour = "white", linewidth = 0.2) +
    facet_grid(rec ~ family, scales = "free_x") +
    scale_fill_gradient(low = "grey96", high = "#7A3E9D", limits = c(0, 1), name = "share of the\ncheckpoint\ndifference\ntransferred") +
    labs(title = "C  Layer-by-position patching (100 pairs)", x = "layer at which states are exchanged", y = NULL) +
    th + theme(legend.position = "right", legend.key.height = unit(5, "mm"), panel.grid = element_blank(),
               axis.text.x = element_text(size = 5.5))
  fig <- pA / pB / pC + plot_layout(heights = c(1, 1.25, 1.1)); h <- 9.4
}
f <- "paper/figures/fig_negation.pdf"
ggsave(f, fig, width = 7, height = h, device = cairo_pdf)   # cairo: the prime in d′ is outside Latin-1
ggsave(sub("pdf$", "png", f), fig, width = 7, height = h, dpi = 200, bg = "white")
