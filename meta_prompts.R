# Post-training effect across three prompt wordings (review round 2, 17 Nov 2026).
# Input: outputs/promptvar_summary.csv (promptvar_summary.py): 54 contrasts final - base on the
# identical plain prompt, 6 model families x 3 corpora x 3 prompts (default, paraphrase,
# question first), each with a grouped-bootstrap CI.
#
# Primary inference: t-test over the six family means (each the mean of 9 contrasts). The
# family is the unit that can be regarded as sampled; corpora and prompts have three levels each.
# Secondary: multilevel model  diff = mu + u_family + v_corpus + w_prompt + c_cell + e_obs
# (+ sampling error from the bootstrap CI), REML, t-based inference with containment df
# (metafor rma.mv). Variance components of factors with three levels are poorly estimated and
# are reported for description only. Sampling errors of the three prompts within a cell stem
# from the same texts and are treated as independent, which the cell effect absorbs only partly.
# Also: I2 for this model and for the 18 default-prompt contrasts (multilevel I2: sum of the
# variance components relative to that sum plus the typical sampling variance; Higgins & Thompson),
# per-prompt and per-corpus family-level t-tests, and the same analyses restricted to contrasts
# in which both models place >= 30% of their probability on the two answer words.
# Usage: Rscript meta_prompts.R  -> outputs/meta_prompts.csv, outputs/meta_prompts_family.csv
suppressPackageStartupMessages({ library(metafor); library(dplyr); library(readr) })

d <- read_csv("outputs/promptvar_summary.csv", show_col_types = FALSE) |>
  rename(prompt = variant, family = preset) |>
  mutate(se = (hi - lo) / 3.92, cell = paste(family, corpus), obs = row_number(),
         pm_min = pmin(pm_base, pm_final))
stopifnot(nrow(d) == 54, n_distinct(d$family) == 6)

i2 <- function(m, v) {  # multilevel I2 with the "typical" sampling variance
  k <- length(v); w <- 1 / v
  vt <- (k - 1) * sum(w) / (sum(w)^2 - sum(w^2))
  sum(m$sigma2) / (sum(m$sigma2) + vt)
}
row_ml <- function(m, x, label) tibble(
  analysis = label, k = nrow(x), families = n_distinct(x$family), est = as.numeric(m$b),
  lo = m$ci.lb, hi = m$ci.ub, t = as.numeric(m$zval), df = m$ddf[[1]], p = m$pval,
  i2 = i2(m, x$se^2), n_negative = sum(x$diff < 0),
  var_components = paste(sprintf("%s=%.3f", m$s.names, sqrt(m$sigma2)), collapse = "; "))
row_tt <- function(x, label) {
  fam <- x |> group_by(family) |> summarise(diff = mean(diff), .groups = "drop")
  tt <- t.test(fam$diff)
  tibble(analysis = label, k = nrow(x), families = nrow(fam), est = unname(tt$estimate),
         lo = tt$conf.int[1], hi = tt$conf.int[2], t = unname(tt$statistic),
         df = unname(tt$parameter), p = tt$p.value, n_negative = sum(x$diff < 0),
         var_components = sprintf("families negative: %d of %d", sum(fam$diff < 0), nrow(fam)))
}
ml_all <- function(x) rma.mv(diff, se^2, random = list(~ 1 | family, ~ 1 | corpus, ~ 1 | prompt,
                             ~ 1 | cell, ~ 1 | obs), data = x, test = "t", dfs = "contain", method = "REML")
ml_18 <- function(x) rma.mv(diff, se^2, random = list(~ 1 | family, ~ 1 | corpus, ~ 1 | cell),
                            data = x, test = "t", method = "REML")

res <- bind_rows(
  row_tt(d, "PRIMARY: t-test over 6 family means (3 corpora x 3 prompts)"),
  row_ml(ml_all(d), d, "multilevel: 54 contrasts, family + corpus + prompt + cell + obs"),
  row_tt(filter(d, pm_min >= .3), "t-test over family means, answer mass >= .3 in both models"),
  row_ml(ml_all(filter(d, pm_min >= .3)), filter(d, pm_min >= .3), "multilevel, answer mass >= .3 in both models"))
for (pv in unique(d$prompt)) res <- bind_rows(res, row_tt(filter(d, prompt == pv), paste("t-test over family means, prompt:", pv)))
for (cp in unique(d$corpus)) res <- bind_rows(res, row_tt(filter(d, corpus == cp), paste("t-test over family means (3 prompts), corpus:", cp)))
d0 <- filter(d, prompt == "default")
res <- bind_rows(res, row_ml(ml_18(d0), d0, "multilevel: 18 default-prompt contrasts (I2 for Section 3.x)"))
for (cp in unique(d$corpus)) res <- bind_rows(res, row_tt(filter(d0, corpus == cp), paste("t-test over family values, default prompt, corpus:", cp)))
write_csv(res, "outputs/meta_prompts.csv")

fam <- d |> group_by(family, prompt) |> summarise(diff = mean(diff), .groups = "drop") |>
  tidyr::pivot_wider(names_from = prompt, values_from = diff) |>
  left_join(d |> group_by(family) |> summarise(all = mean(diff), n_negative = sum(diff < 0), k = n()), by = "family")
write_csv(fam, "outputs/meta_prompts_family.csv")

# prompt sensitivity of single cells: change of the contrast between prompts, and sign flips
wide <- d |> select(family, corpus, prompt, diff, lo, hi) |>
  tidyr::pivot_wider(names_from = prompt, values_from = c(diff, lo, hi))
sens <- tibble(
  mean_abs_default_vs_qfirst = mean(abs(wide$diff_default - wide$diff_qfirst)),
  mean_abs_default_vs_para = mean(abs(wide$diff_default - wide$diff_para)),
  mean_abs_effect_default = mean(abs(wide$diff_default)),
  cells_same_sign_all_prompts = sum(sign(wide$diff_default) == sign(wide$diff_qfirst) & sign(wide$diff_default) == sign(wide$diff_para)),
  cells_negative_all_prompts = sum(wide$diff_default < 0 & wide$diff_qfirst < 0 & wide$diff_para < 0),
  cells_ci_below0_all_prompts = sum(wide$hi_default < 0 & wide$hi_qfirst < 0 & wide$hi_para < 0),
  cells = nrow(wide))
write_csv(sens, "outputs/meta_prompts_sensitivity.csv")

options(width = 220)
print(as.data.frame(mutate(res, across(where(is.numeric), ~ round(.x, 3)))))
print(as.data.frame(mutate(fam, across(where(is.numeric), ~ round(.x, 2)))))
print(as.data.frame(mutate(sens, across(where(is.numeric), ~ round(.x, 2)))))
