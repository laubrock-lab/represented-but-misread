# Conceptual methods figure: how the answer reads the text, steering, stitching, patching.
# Writes paper/figures/fig0_methods_schematic.{pdf,png}.  Usage: Rscript plots/fig_schematic.R
# Colour roles as in the data figures: blue = base checkpoint, orange = SFT checkpoint.

suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(patchwork) })

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; INK <- "#52514e"; TILE <- "#e6e5e1"; LIGHTBLUE <- "#cfe0f6"; LIGHTORANGE <- "#fbd8c8"
OUT <- "paper/figures"

# positions (x) and layers (y)
groups <- data.frame(group = c("instruction", "text", "question", "answer"),
                     from = c(1, 4, 11, 15), to = c(3, 10, 14, 15))
pos <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i)
  data.frame(x = groups$from[i]:groups$to[i], group = groups$group[i])))
L <- 6
grid_df <- merge(pos, data.frame(y = 1:L))
group_lab <- groups |> mutate(x = (from + to) / 2)

base_panel <- function(fill_fun = function(d) TILE, title) {
  d <- grid_df; d$fill <- fill_fun(d)
  ggplot(d) +
    geom_tile(aes(x = x, y = y, fill = I(fill)), width = 0.86, height = 0.8, colour = NA) +
    geom_text(data = group_lab, aes(x = x, y = 0.15, label = group), size = 2.5, colour = INK) +
    annotate("text", x = 0.1, y = (L + 1) / 2, label = "layers", angle = 90, size = 2.5, colour = INK) +
    annotate("segment", x = 0.4, xend = 0.4, y = 0.8, yend = L + 0.2, colour = INK, linewidth = 0.3,
             arrow = arrow(length = unit(0.06, "in"), type = "closed")) +
    coord_cartesian(xlim = c(0, 16.6), ylim = c(-0.2, L + 1.6), clip = "off") +
    labs(title = title) + theme_void(base_size = 8.5) +
    theme(plot.title = element_text(face = "bold", size = 8.5, colour = "black", hjust = 0),
          plot.margin = margin(4, 6, 4, 6))
}
answer_out <- function(p) p +
  annotate("segment", x = 15, xend = 15, y = L + 0.45, yend = L + 0.95, colour = INK, linewidth = 0.35,
           arrow = arrow(length = unit(0.05, "in"), type = "closed")) +
  annotate("text", x = 15, y = L + 1.35, label = "“Mensch” or “KI”?", size = 2.4, colour = INK)

# A: how the answer reads the text
pA <- base_panel(function(d) ifelse(d$group == "text", "#d9d8d3", TILE),
                 "A  The answer can only read the text via the question") |> answer_out()
pA <- pA +
  annotate("curve", x = 7, xend = 12.5, y = 3.4, yend = 3.6, curvature = -0.3, colour = INK, linewidth = 0.4,
           arrow = arrow(length = unit(0.05, "in"), type = "closed")) +
  annotate("curve", x = 12.5, xend = 15, y = 4.4, yend = 4.6, curvature = -0.3, colour = INK, linewidth = 0.4,
           arrow = arrow(length = unit(0.05, "in"), type = "closed")) +
  annotate("text", x = 9.6, y = 4.8, label = "attention", size = 2.3, colour = INK, fontface = "italic")

# B: steering
pB <- base_panel(function(d) ifelse(d$x == 15 & d$y == 4, "#8a8984", TILE),
                 "B  Steering: shift the answer position") |> answer_out()
pB <- pB +
  annotate("segment", x = 17.4, xend = 15.55, y = 4, yend = 4, colour = INK, linewidth = 0.45,
           arrow = arrow(length = unit(0.06, "in"), type = "closed")) +
  annotate("label", x = 12.2, y = 2.05, label = "+ step along the\nAI–human direction", size = 2.3,
           colour = INK, hjust = 0.5, lineheight = 0.9, fill = "white", label.size = 0) +
  annotate("segment", x = 13.6, xend = 14.7, y = 2.6, yend = 3.75, colour = INK, linewidth = 0.25)

# C: stitching
pC <- base_panel(function(d) ifelse(d$y <= 3, BLUE, ORANGE), "C  Stitching: lower layers from one checkpoint") |>
  answer_out()
pC <- pC +
  annotate("segment", x = 0.6, xend = 15.6, y = 3.5, yend = 3.5, colour = "black", linewidth = 0.4, linetype = "dashed") +
  annotate("text", x = 16.0, y = 2, label = "base", size = 2.4, colour = BLUE, hjust = 0, fontface = "bold") +
  annotate("text", x = 16.0, y = 5, label = "SFT", size = 2.4, colour = ORANGE, hjust = 0, fontface = "bold")

# D: patching
pD <- base_panel(function(d) ifelse(d$y == 4 & d$group == "question", BLUE, LIGHTORANGE),
                 "D  Patching: exchange states at selected positions") |> answer_out()
pD <- pD +
  annotate("rect", xmin = 10.5, xmax = 14.5, ymin = 3.55, ymax = 4.45, fill = NA, colour = "black", linewidth = 0.35) +
  annotate("label", x = 12.5, y = 2.35, label = "SFT model with the base\nmodel's question states", size = 2.3,
           colour = INK, lineheight = 0.9, fill = "white", label.size = 0) +
  annotate("text", x = 16.0, y = 3.0, label = "SFT", size = 2.4, colour = ORANGE, hjust = 0, fontface = "bold")

p <- (pA | pB) / (pC | pD)
ggsave(file.path(OUT, "fig0_methods_schematic.pdf"), p, width = 7, height = 4.0, device = cairo_pdf)
ggsave(file.path(OUT, "fig0_methods_schematic.png"), p, width = 7, height = 4.0, dpi = 200, bg = "white")
cat("written fig0_methods_schematic\n")
