# Conceptual figure: the "two-stage lens". Brunswik's lens model (criterion -> cues ->
# judgment), split at the model's internal representation into
#   stage 1 (perception):   are the cues encoded?          -> linear probes
#   stage 2 (utilization):  does the question read them?    -> answers (SDT, lens model)
# Post-training acts on stage 2 (orange); stitching and patching locate it, steering
# tests it causally. Writes paper/figures/fig_lens_two_stage.{pdf,png}.
# Usage: Rscript plots/fig_lens.R

suppressPackageStartupMessages({ library(ggplot2) })

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; INK <- "#52514e"; DARK <- "#3d3c39"
GREY <- "#b8b6b0"; TILE <- "#e6e5e1"; LIGHTORANGE <- "#fbd8c8"
OUT <- "paper/figures"
arr <- function(len = 0.06) arrow(length = unit(len, "in"), type = "closed")

# node positions
crit <- c(0.95, 3)
cue_y <- c(4.9, 3.95, 3, 2.05, 1.1)
cue_x <- 3.1
cue_lab <- c("word choice", "first person", "length", "punctuation", "...")
valid <- c(TRUE, TRUE, FALSE, TRUE, TRUE)          # dashed = invalid cue
rep_xy <- c(5.55, 3)
read_xy <- c(7.75, 3)
ans_xy <- c(9.75, 3)

cues <- data.frame(x = cue_x, y = cue_y, lab = cue_lab, valid = valid)
lines_left <- data.frame(x = crit[1] + 0.68, y = crit[2], xend = cue_x - 0.5, yend = cue_y, valid = valid)
lines_mid <- data.frame(x = cue_x + 0.5, y = cue_y, xend = rep_xy[1] - 0.75, yend = rep_xy[2], valid = valid)

box <- function(xy, w, h, fill, col, lab, size = 2.6, face = "plain", lcol = DARK) {
  list(annotate("rect", xmin = xy[1] - w / 2, xmax = xy[1] + w / 2, ymin = xy[2] - h / 2, ymax = xy[2] + h / 2,
                fill = fill, colour = col, linewidth = 0.4),
       annotate("text", x = xy[1], y = xy[2], label = lab, size = size, colour = lcol, fontface = face,
                lineheight = 0.9))
}

p <- ggplot() +
  # stage backgrounds
  annotate("rect", xmin = 4.35, xmax = 6.75, ymin = 0.25, ymax = 5.6, fill = TILE, colour = NA) +
  annotate("rect", xmin = 6.75, xmax = 8.85, ymin = 0.25, ymax = 5.6, fill = LIGHTORANGE, colour = NA) +
  annotate("text", x = 5.55, y = 5.33, label = "Stage 1: perception", size = 2.7, fontface = "bold", colour = DARK) +
  annotate("text", x = 7.8, y = 5.33, label = "Stage 2: utilization", size = 2.7, fontface = "bold", colour = DARK) +
  # lens lines
  geom_segment(data = lines_left, aes(x = x, y = y, xend = xend, yend = yend, linetype = valid),
               colour = INK, linewidth = 0.4) +
  geom_segment(data = lines_mid, aes(x = x, y = y, xend = xend, yend = yend, linetype = valid),
               colour = INK, linewidth = 0.4) +
  scale_linetype_manual(values = c(`TRUE` = "solid", `FALSE` = "22"), guide = "none") +
  # nodes
  box(crit, 1.38, 1.3, "white", INK, "Criterion\n(AI or human?\ntrue or false?\ngenuine or fake?)", size = 2.2) +
  geom_label(data = cues, aes(x = x, y = y, label = lab), size = 2.3, colour = DARK, fill = "white",
             linewidth = 0.3, label.r = unit(0.08, "lines"), label.padding = unit(0.18, "lines")) +
  box(rep_xy, 1.45, 1.25, "white", INK, "Internal\nrepresentation\nof the text", size = 2.35) +
  box(read_xy, 1.55, 1.25, "white", ORANGE, "Readout by\nthe question", size = 2.35) +
  box(ans_xy, 1.05, 0.9, "white", INK, "Answer", size = 2.5, face = "bold") +
  annotate("segment", x = rep_xy[1] + 0.73, xend = read_xy[1] - 0.8, y = 3, yend = 3, colour = DARK,
           linewidth = 0.5, arrow = arr()) +
  annotate("segment", x = read_xy[1] + 0.78, xend = ans_xy[1] - 0.55, y = 3, yend = 3, colour = DARK,
           linewidth = 0.5, arrow = arr()) +
  annotate("text", x = 7.75, y = 2.12, label = "changed by\npost-training", size = 2.3, colour = ORANGE,
           fontface = "bold", lineheight = 0.9) +
  # achievement: bracket over the whole lens
  annotate("segment", x = crit[1], xend = ans_xy[1], y = 6.2, yend = 6.2, colour = INK, linewidth = 0.35,
           linetype = "dashed") +
  annotate("segment", x = c(crit[1], ans_xy[1]), xend = c(crit[1], ans_xy[1]), y = 6.2,
           yend = c(crit[2] + 0.68, ans_xy[2] + 0.48), colour = INK, linewidth = 0.35, linetype = "dashed") +
  annotate("label", x = 5.35, y = 6.2, label = "achievement: does the answer track the criterion?",
           size = 2.3, colour = INK, fill = "white", linewidth = 0) +
  # legend for invalid cues
  annotate("segment", x = 2.35, xend = 2.75, y = 0.55, yend = 0.55, colour = INK, linewidth = 0.4, linetype = "22") +
  annotate("text", x = 2.82, y = 0.55, label = "cue without validity", size = 2.1, colour = INK, hjust = 0) +
  # measures (bottom row)
  annotate("text", x = 2.0, y = -0.05, label = "cue validity\n(text statistics,\nn-gram classifiers)",
           size = 2.15, colour = INK, lineheight = 0.9, vjust = 1) +
  annotate("text", x = 5.55, y = -0.05, label = "decodability\n(linear probes)",
           size = 2.15, colour = INK, lineheight = 0.9, vjust = 1) +
  annotate("text", x = 7.8, y = -0.05, label = "cue utilization\n(signal detection,\nlens model)",
           size = 2.15, colour = INK, lineheight = 0.9, vjust = 1) +
  # causal methods (bottom row 2)
  annotate("segment", x = 5.0, xend = 9.3, y = -1.25, yend = -1.25, colour = BLUE, linewidth = 0.45, arrow = arr()) +
  annotate("text", x = 7.15, y = -1.47, label = "steering: shift the representation, observe the answer",
           size = 2.15, colour = BLUE) +
  annotate("segment", x = 6.85, xend = 8.75, y = -1.85, yend = -1.85, colour = ORANGE, linewidth = 0.45) +
  annotate("segment", x = c(6.85, 8.75), xend = c(6.85, 8.75), y = -1.75, yend = -1.95, colour = ORANGE,
           linewidth = 0.45) +
  annotate("text", x = 7.8, y = -2.08, label = "stitching and patching: where (layer, position)?",
           size = 2.15, colour = ORANGE) +
  coord_cartesian(xlim = c(0.2, 10.4), ylim = c(-2.2, 6.8), clip = "off") +
  theme_void(base_size = 8.5) + theme(plot.margin = margin(6, 6, 6, 6))

dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(OUT, "fig_lens_two_stage.pdf"), p, width = 6.6, height = 3.7, device = pdf)   # base pdf device: Helvetica core font, correct metrics in every viewer
embedFonts(file.path(OUT, "fig_lens_two_stage.pdf"), outfile = file.path(OUT, "fig_lens_two_stage.pdf"),
           options = "-dPDFSETTINGS=/prepress -dEmbedAllFonts=true")   # arXiv/journals: embed all fonts (needs ghostscript)
ggsave(file.path(OUT, "fig_lens_two_stage.png"), p, width = 6.6, height = 3.7, dpi = 200, bg = "white")
cat("written:", file.path(OUT, "fig_lens_two_stage.{pdf,png}"), "\n")
