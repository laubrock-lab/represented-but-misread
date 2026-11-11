# Conceptual methods figure: (A) how the answer reads the text, (B) steering in a
# (2-D sketch of the) representation space, (C) stitching, (D) patching.
# Writes paper/figures/fig0_methods_schematic.{pdf,png}.  Usage: Rscript plots/fig_schematic.R
# Colour roles as in the data figures: blue = base checkpoint, orange = SFT checkpoint;
# AI vs. human texts in panel B are coded by shape and grey tone (not by colour).

suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(patchwork) })

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; INK <- "#52514e"; TILE <- "#e6e5e1"; LIGHTORANGE <- "#fbd8c8"
DARK <- "#3d3c39"
OUT <- "paper/figures"
arr <- function(len = 0.055) arrow(length = unit(len, "in"), type = "closed")

# ------------------------------------------------------------ position x layer grid
groups <- data.frame(group = c("instruction", "text", "question", "answer"),
                     from = c(1, 4, 11, 15), to = c(3, 10, 14, 15))
pos <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i)
  data.frame(x = groups$from[i]:groups$to[i], group = groups$group[i])))
L <- 6
grid_df <- merge(pos, data.frame(y = 1:L))
group_lab <- groups |> mutate(x = (from + to) / 2)

grid_panel <- function(fill_fun, title) {
  d <- grid_df; d$fill <- fill_fun(d)
  ggplot(d) +
    geom_tile(aes(x = x, y = y, fill = I(fill)), width = 0.86, height = 0.8, colour = NA) +
    geom_text(data = group_lab, aes(x = x, y = 0.2, label = group), size = 2.4, colour = INK) +
    # axes
    annotate("segment", x = 0.5, xend = 15.6, y = -0.35, yend = -0.35, colour = INK, linewidth = 0.3, arrow = arr()) +
    annotate("text", x = 8, y = -0.8, label = "position in the prompt (token order)", size = 2.4, colour = INK) +
    annotate("segment", x = 0.1, xend = 0.1, y = 0.6, yend = L + 0.4, colour = INK, linewidth = 0.3, arrow = arr()) +
    annotate("text", x = -0.35, y = (L + 1) / 2, label = "layer", angle = 90, size = 2.4, colour = INK) +
    # answer read-out
    annotate("segment", x = 15, xend = 15, y = L + 0.45, yend = L + 0.95, colour = INK, linewidth = 0.35, arrow = arr(0.05)) +
    annotate("text", x = 15, y = L + 1.35, label = "“Mensch” or “KI”?", size = 2.3, colour = INK) +
    coord_cartesian(xlim = c(-0.6, 17), ylim = c(-1.0, L + 1.6), clip = "off") +
    labs(title = title) + theme_void(base_size = 8.5) +
    theme(plot.title = element_text(face = "bold", size = 8.5, hjust = 0),
          plot.margin = margin(4, 8, 4, 8))
}

# A: how the answer reads the text — text -> question -> answer, layer by layer
hiA <- data.frame(x = c(7, 12, 15), y = c(2, 3, 4))
pA <- grid_panel(function(d) TILE, "A  The answer reads the text via the question") +
  geom_tile(data = hiA, aes(x = x, y = y), fill = "#9a9994", width = 0.86, height = 0.8) +
  annotate("curve", x = 7.3, xend = 11.7, y = 2.2, yend = 2.85, curvature = -0.25, colour = DARK,
           linewidth = 0.45, arrow = arr()) +
  annotate("curve", x = 12.3, xend = 14.7, y = 3.2, yend = 3.85, curvature = -0.3, colour = DARK,
           linewidth = 0.45, arrow = arr()) +
  annotate("label", x = 9.2, y = 4.55, label = "attention: later positions\nread earlier positions",
           size = 2.2, colour = INK, lineheight = 0.9, fill = "white", linewidth = 0)

# B: steering in a 2-D sketch of the representation space
set.seed(3)
n <- 28
pts <- rbind(data.frame(x = rnorm(n, -1.1, 0.42), y = rnorm(n, -0.45, 0.42), cls = "human texts"),
             data.frame(x = rnorm(n,  1.1, 0.42), y = rnorm(n,  0.45, 0.42), cls = "AI texts"))
mu <- pts |> group_by(cls) |> summarise(x = mean(x), y = mean(y))
mh <- mu[mu$cls == "human texts", ]; ma <- mu[mu$cls == "AI texts", ]
u <- c(ma$x - mh$x, ma$y - mh$y); u <- u / sqrt(sum(u^2))
p0 <- c(-0.55, 0.75); p1 <- p0 + 1.25 * u
pB <- ggplot(pts, aes(x, y)) +
  geom_abline(intercept = 0.05, slope = -1.4, colour = INK, linetype = "dashed", linewidth = 0.35) +
  annotate("text", x = -1.75, y = 2.05, label = "model answers\n“Mensch”", size = 2.2, colour = INK, lineheight = 0.9) +
  annotate("text", x = 1.75, y = -1.75, label = "model answers\n“KI”", size = 2.2, colour = INK, lineheight = 0.9) +
  geom_point(aes(shape = cls, colour = cls), size = 1.3, stroke = 0.4) +
  annotate("segment", x = mh$x, y = mh$y, xend = ma$x, yend = ma$y, colour = DARK, linewidth = 0.7, arrow = arr(0.07)) +
  annotate("label", x = 0.55, y = -0.62, label = "AI–human direction", size = 2.2, colour = DARK,
           fill = "white", linewidth = 0) +
  annotate("point", x = p0[1], y = p0[2], shape = 23, size = 2.4, fill = "white", colour = DARK, stroke = 0.6) +
  annotate("segment", x = p0[1], y = p0[2], xend = p1[1], yend = p1[2], colour = DARK, linewidth = 0.5,
           linetype = "solid", arrow = arr(0.06)) +
  annotate("point", x = p1[1], y = p1[2], shape = 23, size = 2.4, fill = DARK, colour = DARK) +
  annotate("label", x = -0.15, y = 1.75, label = "one text, shifted\nalong the direction", size = 2.2,
           colour = DARK, lineheight = 0.9, fill = "white", linewidth = 0) +
  annotate("text", x = -1.15, y = -1.35, label = "human texts", size = 2.3, colour = "#8a8984", fontface = "bold") +
  annotate("text", x = 1.2, y = 1.35, label = "AI texts", size = 2.3, colour = "#6b6a66", fontface = "bold") +
  scale_shape_manual(values = c("human texts" = 1, "AI texts" = 17)) +
  scale_colour_manual(values = c("human texts" = "#8a8984", "AI texts" = "#6b6a66")) +
  coord_fixed(xlim = c(-2.4, 2.4), ylim = c(-2.1, 2.3)) +
  labs(title = "B  Steering in the representation space",
       x = "internal dimension 1", y = "internal dimension 2") +
  theme_void(base_size = 8.5) +
  theme(plot.title = element_text(face = "bold", size = 8.5, hjust = 0),
        axis.title.x = element_text(size = 7, colour = INK, margin = margin(t = 3)),
        axis.title.y = element_text(size = 7, colour = INK, angle = 90, margin = margin(r = 3)),
        legend.position = "none",
        panel.border = element_rect(colour = TILE, fill = NA, linewidth = 0.4),
        plot.margin = margin(4, 8, 4, 8))

# C: stitching
pC <- grid_panel(function(d) ifelse(d$y <= 3, BLUE, ORANGE), "C  Stitching: lower and upper layers from different models") +
  annotate("segment", x = 0.6, xend = 15.6, y = 3.5, yend = 3.5, colour = "black", linewidth = 0.4, linetype = "dashed") +
  annotate("text", x = 16.05, y = 2, label = "base", size = 2.4, colour = BLUE, hjust = 0, fontface = "bold") +
  annotate("text", x = 16.05, y = 5, label = "SFT", size = 2.4, colour = ORANGE, hjust = 0, fontface = "bold")

# D: patching
pD <- grid_panel(function(d) ifelse(d$y == 4 & d$group == "question", BLUE, LIGHTORANGE),
                 "D  Patching: exchange states at selected positions") +
  annotate("rect", xmin = 10.5, xmax = 14.5, ymin = 3.55, ymax = 4.45, fill = NA, colour = "black", linewidth = 0.35) +
  annotate("label", x = 12.5, y = 2.3, label = "SFT model with the base\nmodel's question states", size = 2.2,
           colour = INK, lineheight = 0.9, fill = "white", linewidth = 0) +
  annotate("text", x = 16.05, y = 3.0, label = "SFT", size = 2.4, colour = ORANGE, hjust = 0, fontface = "bold")

p <- (pA | pB) / (pC | pD)
ggsave(file.path(OUT, "fig0_methods_schematic.pdf"), p, width = 7.2, height = 5.0, device = cairo_pdf)
ggsave(file.path(OUT, "fig0_methods_schematic.png"), p, width = 7.2, height = 5.0, dpi = 200, bg = "white")
cat("written fig0_methods_schematic\n")
