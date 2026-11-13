# Supplementary figure S2: layer-wise probe transfer between the base and the final
# post-trained checkpoint. Transfer deficit = within-checkpoint probe d' minus the d' of
# the probe taken from the other checkpoint (both directions averaged). 0 = the same
# direction separates the classes in both checkpoints.
# Answer position: OLMo 2 and Apertus (the two families present in all six sets)
# highlighted, other families as thin grey lines. Text positions (authorship corpora
# only): mean over families. Number of families in the panel titles.
# Reads outputs/checkpoint_transfer_*.csv (checkpoint_transfer.py; numbers only).
# Writes paper/figures/figS2_checkpoint_transfer.{pdf,png}.
# Usage: Rscript plots/fig_checkpoint_transfer.R

suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(readr) })

INK <- "#3d3c39"; GREY <- "#c9c7c1"; LIGHT <- "#a9a7a1"; OUT <- "paper/figures"
SETS <- c(chat_tw25 = "Authorship: chats", journalism_seg = "Authorship: news",
          textbook = "Authorship: textbook", truth_neg_cities = "Negated statements",
          truth_counterfact = "Facts (counterfact)", truth_opspam = "Fake reviews")
FOCUS <- c(`olmo2-7b` = "OLMo 2 7B", `apertus15-8b` = "Apertus 1.5 8B")

d <- bind_rows(lapply(Sys.glob("outputs/checkpoint_transfer_*.csv"), read_csv, show_col_types = FALSE)) |>
  filter(post == "instruct") |>
  distinct(set, preset, post, site, layer, .keep_all = TRUE) |>
  mutate(deficit = ((within_base - base_to_post) + (within_post - post_to_base)) / 2)
nfam <- d |> distinct(set, preset) |> count(set, name = "k")
d <- d |> left_join(nfam, by = "set") |>
  mutate(panel = factor(paste0(SETS[set], " (", k, " models)"),
                        levels = paste0(SETS[nfam$set[match(names(SETS), nfam$set)]], " (",
                                        nfam$k[match(names(SETS), nfam$set)], " models)")),
         family = ifelse(preset %in% names(FOCUS), FOCUS[preset], "other models"),
         family = factor(family, levels = c(FOCUS, "other models")))

ans <- d |> filter(site == "last")
txt <- d |> filter(site == "textmean") |> mutate(depth_bin = round(depth * 10) / 10) |>
  group_by(panel, depth_bin) |> summarise(deficit = mean(deficit), .groups = "drop") |>
  mutate(family = "text positions (mean)")

p <- ggplot() +
  geom_hline(yintercept = 0, colour = GREY, linewidth = 0.3) +
  geom_line(data = filter(ans, family == "other models"),
            aes(depth, deficit, group = preset, colour = family, linetype = family, linewidth = family)) +
  geom_line(data = txt, aes(depth_bin, deficit, colour = family, linetype = family, linewidth = family)) +
  geom_line(data = filter(ans, family != "other models"),
            aes(depth, deficit, group = preset, colour = family, linetype = family, linewidth = family)) +
  facet_wrap(~panel, nrow = 2) +
  scale_colour_manual(NULL, values = c(`OLMo 2 7B` = INK, `Apertus 1.5 8B` = INK, `other models` = GREY,
                                       `text positions (mean)` = LIGHT)) +
  scale_linetype_manual(NULL, values = c(`OLMo 2 7B` = "solid", `Apertus 1.5 8B` = "42", `other models` = "solid",
                                         `text positions (mean)` = "solid")) +
  scale_linewidth_manual(NULL, values = c(`OLMo 2 7B` = 0.75, `Apertus 1.5 8B` = 0.75, `other models` = 0.35,
                                          `text positions (mean)` = 1.4)) +
  scale_x_continuous("relative depth (layer / number of layers)", breaks = c(0, .5, 1), labels = c("0", ".5", "1")) +
  scale_y_continuous(expression(paste("transfer deficit (", italic(d), "')"))) +
  coord_cartesian(ylim = c(-0.2, 2.75)) +   # clip without dropping data (no broken lines)
  theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        legend.position = "top", legend.key.width = unit(1.4, "lines"),
        strip.text = element_text(face = "bold", hjust = 0))

dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
f <- file.path(OUT, "figS2_checkpoint_transfer.pdf")
ggsave(f, p, width = 6.8, height = 4.3, device = pdf)
embedFonts(f, outfile = f, options = "-dPDFSETTINGS=/prepress -dEmbedAllFonts=true")
ggsave(file.path(OUT, "figS2_checkpoint_transfer.png"), p, width = 6.8, height = 4.3, dpi = 200, bg = "white")
cat("written:", f, "and .png\n")
