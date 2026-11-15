# Multilevel meta-analysis of the post-training effect (final vs. base, identical raw prompt),
# response to review (W6/S6, 16 Nov 2026). The 18 contrasts are not independent: each model
# family contributes three corpora and each corpus six families. Model: Δd′_ij = μ + u_family +
# v_corpus + e_ij + sampling error (SE from the grouped bootstrap CI), REML, t-based inference
# (metafor rma.mv, test = "t"). Also: family-level t-test on the six family means, and the
# sensitivity analyses at answer mass ≥ .5 and for "responsive" contrasts (binary answers vary
# in both conditions).
# Usage: Rscript meta_multilevel.R  -> outputs/meta_multilevel.csv
suppressPackageStartupMessages({ library(metafor); library(dplyr); library(readr) })

d <- read_csv("outputs/contrasts_primary.csv", show_col_types = FALSE) |>
  mutate(se = (hi - lo) / 3.92, family = preset, cell = paste(preset, corpus))
cond <- read_csv("outputs/contrasts_conditions_pmass30.csv", show_col_types = FALSE) |>
  filter(fmt == "raw") |> select(corpus, preset, variant, p_mass)
pm <- cond |> tidyr::pivot_wider(names_from = variant, values_from = p_mass, names_prefix = "pm_") |>
  select(corpus, preset, pm_base, pm_instruct)
d <- d |> left_join(pm, by = c("corpus", "preset")) |>
  mutate(pm_min = pmin(pm_base, pm_instruct))

fit <- function(x, label) {
  if (nrow(x) < 3) return(NULL)
  nf <- n_distinct(x$family); nc <- n_distinct(x$corpus)
  rand <- list(~ 1 | family, ~ 1 | corpus, ~ 1 | cell)
  if (nc < 2) rand <- list(~ 1 | family, ~ 1 | cell)
  if (nf < 2) rand <- list(~ 1 | corpus, ~ 1 | cell)
  m <- rma.mv(diff, se^2, random = rand, data = x, test = "t", method = "REML")
  tibble(analysis = label, k = nrow(x), families = nf, est = as.numeric(m$b), lo = m$ci.lb,
         hi = m$ci.ub, t = m$zval, df = m$ddf[[1]], p = m$pval,
         sd_family = sqrt(m$sigma2[1]), sd_corpus = if (nc >= 2 && nf >= 2) sqrt(m$sigma2[2]) else NA,
         sd_cell = sqrt(tail(m$sigma2, 1)), n_negative = sum(x$diff < 0))
}

fam <- d |> group_by(family) |> summarise(diff = mean(diff), .groups = "drop")
tt <- t.test(fam$diff)
res <- bind_rows(
  fit(d, "multilevel: all 18 contrasts"),
  fit(filter(d, pm_min >= .5), "multilevel: answer mass >= .5 in both conditions"),
  tibble(analysis = "t-test over the 6 family means", k = nrow(fam), families = nrow(fam),
         est = unname(tt$estimate), lo = tt$conf.int[1], hi = tt$conf.int[2],
         t = unname(tt$statistic), df = unname(tt$parameter), p = tt$p.value,
         n_negative = sum(fam$diff < 0)),
  tibble(analysis = "responsive contrasts (binary answers vary in both conditions)",
         k = sum(d$responsive), est = mean(d$diff[d$responsive]),
         n_negative = sum(d$diff[d$responsive] < 0))
)
for (cp in unique(d$corpus)) res <- bind_rows(res, fit(filter(d, corpus == cp), paste("multilevel, corpus:", cp)))
write_csv(res, "outputs/meta_multilevel.csv")
print(as.data.frame(mutate(res, across(where(is.numeric), ~ round(.x, 3)))))
cat("\nfamily means:\n"); print(as.data.frame(mutate(fam, diff = round(diff, 2))))
cat("\ncontrasts with answer mass >= .5 in both conditions:\n")
print(as.data.frame(d |> filter(pm_min >= .5) |> select(corpus, family, diff, pm_base, pm_instruct) |>
  mutate(across(where(is.numeric), ~ round(.x, 2)))))
