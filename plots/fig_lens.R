# Conceptual figure: Brunswik's lens model with an internal lens.
#   Lens 1 (encoding):  state of the world -> cues in the text -> internal representation
#                        measured by cue validity (text statistics) and decodability (probes)
#   Lens 2 (readout):   internal representation -> internal features -> answer, weighted by
#                        the question; measured by steering (weight and sign of the measured
#                        class direction) and the answers (signal detection, lens model)
# Post-training changes lens 2 (orange); stitching and patching locate the change.
# Precedents for splitting the judge side: Funder's realistic accuracy model (detection vs.
# utilization); encoding vs. read-out in neuroscience (Panzeri et al., 2017).
# Writes paper/figures/fig_lens_two_stage.{pdf,png}.  Usage: Rscript plots/fig_lens.R

suppressPackageStartupMessages({ library(ggplot2) })

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; INK <- "#52514e"; DARK <- "#3d3c39"
GREY <- "#b8b6b0"; TILE <- "#ecebe7"; LIGHTORANGE <- "#fbd8c8"; MID <- "#8f8d88"
OUT <- "paper/figures"
arr <- function(len = 0.06) arrow(length = unit(len, "in"), type = "closed")

world <- c(0.95, 3); rep_xy <- c(5.05, 3); ans_xy <- c(10.75, 3)
cue_x <- 2.95; cue_y <- c(4.9, 3.95, 3, 2.05, 1.1)
cues <- data.frame(x = cue_x, y = cue_y, lab = c("word choice", "first person", "length", "punctuation", "..."),
                   valid = c(TRUE, TRUE, FALSE, TRUE, TRUE))
feat_x <- 7.75; feat_y <- c(4.55, 3.55, 2.55, 1.55)
feats <- data.frame(x = feat_x, y = feat_y, measured = c(TRUE, FALSE, FALSE, FALSE),
                    lab = c("class direction\n(e.g., AI vs. human)", "feature", "feature", "..."))

l1a <- data.frame(x = world[1] + 0.68, y = world[2], xend = cue_x - 0.5, yend = cue_y, valid = cues$valid)
l1b <- data.frame(x = cue_x + 0.5, y = cue_y, xend = rep_xy[1] - 0.66, yend = rep_xy[2], valid = cues$valid)
l2a <- data.frame(x = rep_xy[1] + 0.66, y = rep_xy[2], xend = feat_x - 0.78, yend = feat_y, measured = feats$measured)
l2b <- data.frame(x = feat_x + 0.78, y = feat_y, xend = ans_xy[1] - 0.55, yend = ans_xy[2], measured = feats$measured)

box <- function(xy, w, h, col, lab, size = 2.35, face = "plain") {
  list(annotate("rect", xmin = xy[1] - w / 2, xmax = xy[1] + w / 2, ymin = xy[2] - h / 2, ymax = xy[2] + h / 2,
                fill = "white", colour = col, linewidth = 0.4),
       annotate("text", x = xy[1], y = xy[2], label = lab, size = size, colour = DARK, fontface = face,
                lineheight = 0.9))
}

p <- ggplot() +
  # lens regions
  annotate("rect", xmin = 1.9, xmax = 5.05, ymin = 0.35, ymax = 5.65, fill = TILE, colour = NA) +
  annotate("rect", xmin = 5.05, xmax = 10.05, ymin = 0.35, ymax = 5.65, fill = LIGHTORANGE, colour = NA) +
  annotate("text", x = 3.5, y = 5.42, label = "Lens 1: encoding", size = 2.75, fontface = "bold", colour = DARK) +
  annotate("text", x = 7.55, y = 5.42, label = "Lens 2: readout", size = 2.75, fontface = "bold", colour = DARK) +
  # lens 1
  geom_segment(data = l1a, aes(x = x, y = y, xend = xend, yend = yend, linetype = valid), colour = INK, linewidth = 0.4) +
  geom_segment(data = l1b, aes(x = x, y = y, xend = xend, yend = yend, linetype = valid), colour = INK, linewidth = 0.4) +
  scale_linetype_manual(values = c(`TRUE` = "solid", `FALSE` = "22"), guide = "none") +
  geom_label(data = cues, aes(x = x, y = y, label = lab), size = 2.25, colour = DARK, fill = "white",
             linewidth = 0.3, label.r = unit(0.08, "lines"), label.padding = unit(0.17, "lines")) +
  # lens 2: measured class direction dark, further features schematic (grey)
  geom_segment(data = l2a, aes(x = x, y = y, xend = xend, yend = yend), colour = ifelse(l2a$measured, DARK, MID),
               linewidth = ifelse(l2a$measured, 0.65, 0.35)) +
  geom_segment(data = l2b, aes(x = x, y = y, xend = xend, yend = yend), colour = ifelse(l2b$measured, DARK, MID),
               linewidth = ifelse(l2b$measured, 0.65, 0.35)) +
  geom_label(data = feats, aes(x = x, y = y, label = lab), size = 2.15, lineheight = 0.88,
             colour = ifelse(feats$measured, DARK, MID), fill = "white",
             fontface = ifelse(feats$measured, "bold", "plain"),
             linewidth = 0.3, label.r = unit(0.08, "lines"), label.padding = unit(0.17, "lines")) +
  # the question sets the weights of lens 2
  annotate("label", x = 9.35, y = 4.95, label = "question\n(task set)", size = 2.2, colour = ORANGE,
           fill = "white", lineheight = 0.9, linewidth = 0.35) +
  annotate("segment", x = 9.35, xend = 9.35, y = 4.55, yend = 3.35, colour = ORANGE, linewidth = 0.45, arrow = arr()) +
  annotate("text", x = 7.95, y = 0.7, label = "weights and signs set by the question;\nchanged by post-training",
           size = 2.15, colour = ORANGE, fontface = "bold", lineheight = 0.9) +
  # nodes
  box(world, 1.42, 1.38, INK, "State of the\nworld\n(ground truth)") +
  box(rep_xy, 1.32, 1.2, INK, "Internal\nrepresentation") +
  box(ans_xy, 1.05, 0.9, INK, "Answer", size = 2.5, face = "bold") +
  # achievement
  annotate("segment", x = world[1], xend = ans_xy[1], y = 6.25, yend = 6.25, colour = INK, linewidth = 0.35, linetype = "dashed") +
  annotate("segment", x = c(world[1], ans_xy[1]), xend = c(world[1], ans_xy[1]), y = 6.25,
           yend = c(world[2] + 0.7, ans_xy[2] + 0.46), colour = INK, linewidth = 0.35, linetype = "dashed") +
  annotate("label", x = 5.85, y = 6.25, label = "achievement: does the answer track the state of the world?",
           size = 2.25, colour = INK, fill = "white", linewidth = 0) +
  # measures
  annotate("text", x = 2.95, y = 0.05, label = "cue validity\n(text statistics,\nn-gram classifiers)",
           size = 2.1, colour = INK, lineheight = 0.9, vjust = 1) +
  annotate("text", x = 5.05, y = 0.05, label = "decodability\n(linear probes,\nprobe transfer)",
           size = 2.1, colour = INK, lineheight = 0.9, vjust = 1) +
  annotate("text", x = 7.75, y = 0.05, label = "utilization of the class direction\n(steering: weight and sign)",
           size = 2.1, colour = INK, lineheight = 0.9, vjust = 1) +
  annotate("text", x = 10.75, y = 0.05, label = "sensitivity of the answers\n(signal detection,\nlens model)",
           size = 2.1, colour = INK, lineheight = 0.9, vjust = 1) +
  # where: stitching and patching
  annotate("segment", x = 5.95, xend = 9.95, y = -1.15, yend = -1.15, colour = ORANGE, linewidth = 0.45) +
  annotate("segment", x = c(5.95, 9.95), xend = c(5.95, 9.95), y = -1.05, yend = -1.25, colour = ORANGE, linewidth = 0.45) +
  annotate("text", x = 7.95, y = -1.42, label = "stitching and patching: where does the readout change (layer, position)?",
           size = 2.1, colour = ORANGE) +
  # legend
  annotate("segment", x = 0.25, xend = 0.6, y = -1.15, yend = -1.15, colour = INK, linewidth = 0.4, linetype = "22") +
  annotate("text", x = 0.67, y = -1.15, label = "cue without validity", size = 2.05, colour = INK, hjust = 0) +
  annotate("segment", x = 0.25, xend = 0.6, y = -1.5, yend = -1.5, colour = MID, linewidth = 0.35) +
  annotate("text", x = 0.67, y = -1.5, label = "internal features: schematic", size = 2.05, colour = INK, hjust = 0) +
  annotate("segment", x = 0.25, xend = 0.6, y = -1.85, yend = -1.85, colour = DARK, linewidth = 0.65) +
  annotate("text", x = 0.67, y = -1.85, label = "class direction: measured", size = 2.05, colour = INK, hjust = 0) +
  coord_cartesian(xlim = c(0.2, 11.4), ylim = c(-1.95, 6.6), clip = "off") +
  theme_void(base_size = 8.5) + theme(plot.margin = margin(6, 6, 6, 6))

dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
f <- file.path(OUT, "fig_lens_two_stage.pdf")
ggsave(f, p, width = 7.0, height = 3.9, device = pdf)   # base pdf device: Helvetica core font, correct metrics in every viewer
embedFonts(f, outfile = f, options = "-dPDFSETTINGS=/prepress -dEmbedAllFonts=true")   # arXiv/journals: embed all fonts (needs ghostscript)
ggsave(file.path(OUT, "fig_lens_two_stage.png"), p, width = 7.0, height = 3.9, dpi = 200, bg = "white")
cat("written:", file.path(OUT, "fig_lens_two_stage.{pdf,png}"), "\n")
