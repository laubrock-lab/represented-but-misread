# Supplementary figure: layer-wise probe transfer between the base and the final
# post-trained checkpoint. Transfer deficit = within-checkpoint probe d' minus the d' of
# the probe taken from the other checkpoint (both directions averaged). 0 = the same
# direction separates the classes in both checkpoints.
# Reads outputs/checkpoint_transfer_*.csv (checkpoint_transfer.py; numbers only).
# Writes paper/figures/figS2_checkpoint_transfer.{pdf,png}.
# Usage: Rscript plots/fig_checkpoint_transfer.R

suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(readr) })

INK <- "#3d3c39"; GREY <- "#b8b6b0"; OUT <- "paper/figures"
files <- Sys.glob("outputs/checkpoint_transfer_*.csv")
d <- bind_rows(lapply(files, read_csv, show_col_types = FALSE)) |>
  filter(post == "instruct") |>
  distinct(set, preset, post, site, layer, .keep_all = TRUE) |>
  mutate(deficit = ((within_base - base_to_post) + (within_post - post_to_base)) / 2,
         site = factor(ifelse(site == "last", "answer position", "text positions"),
                       levels = c("answer position", "text positions")),
         set = factor(recode(set, chat_tw25 = "Authorship: chats", journalism_seg = "Authorship: news",
                             textbook = "Authorship: textbook", truth_neg_cities = "Negated statements",
                             truth_counterfact = "Facts (counterfact)", truth_opspam = "Fake reviews"),
                      levels = c("Authorship: chats", "Authorship: news", "Authorship: textbook",
                                 "Negated statements", "Facts (counterfact)", "Fake reviews")))

# mean over families on a common depth grid (families differ in their number of layers)
grid <- d |> mutate(depth_bin = round(depth * 10) / 10) |>
  group_by(set, site, depth_bin) |> summarise(deficit = mean(deficit), .groups = "drop")
nfam <- d |> distinct(set, preset) |> count(set, name = "k")

p <- ggplot(d, aes(depth, deficit, colour = site)) +
  geom_hline(yintercept = 0, colour = GREY, linewidth = 0.3) +
  geom_line(aes(group = interaction(preset, site)), linewidth = 0.3, alpha = 0.45) +
  geom_line(data = grid, aes(depth_bin, deficit), linewidth = 1.1) +
  geom_text(data = nfam, aes(x = 0.02, y = 2.55, label = paste0(k, ifelse(k == 1, " family", " families"))),
            inherit.aes = FALSE, hjust = 0, size = 2.5, colour = INK) +
  facet_wrap(~set, nrow = 2) +
  scale_colour_manual(values = c(`answer position` = INK, `text positions` = "#9a9994"), name = NULL) +
  scale_x_continuous("relative depth (layer / number of layers)", breaks = c(0, .5, 1), labels = c("0", ".5", "1")) +
  scale_y_continuous(expression(paste("transfer deficit (", italic(d), "')")), limits = c(-0.3, 2.7)) +
  theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        legend.position = "top", strip.text = element_text(face = "bold", hjust = 0))

dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
f <- file.path(OUT, "figS2_checkpoint_transfer.pdf")
ggsave(f, p, width = 6.6, height = 4.2, device = pdf)
embedFonts(f, outfile = f, options = "-dPDFSETTINGS=/prepress -dEmbedAllFonts=true")
ggsave(file.path(OUT, "figS2_checkpoint_transfer.png"), p, width = 6.6, height = 4.2, dpi = 200, bg = "white")
cat("written:", f, "and .png\n")
