#!/usr/bin/env Rscript
# ---------------------------------------------------------
# Aggregate Script: sim_20_er_thompson
# ---------------------------------------------------------
suppressPackageStartupMessages({
    library(dplyr)
})

output_dir <- "output"
all_results_file <- file.path(output_dir, "all_results.rds")
summary_csv <- file.path(output_dir, "summary_results.csv")
summary_rds <- file.path(output_dir, "summary_results.rds")

if (!file.exists(all_results_file)) {
    stop("all_results.rds not found. Run retrieve.R first.")
}

all_results <- readRDS(all_results_file)

cat("Aggregating results...\n")

summary_stats <- data.frame(
    n_reps = nrow(all_results),
    mean_accuracy = mean(all_results$accuracy, na.rm = TRUE),
    sd_accuracy = sd(all_results$accuracy, na.rm = TRUE),
    mean_n_selected = mean(all_results$n_selected, na.rm = TRUE),
    mean_tp = mean(all_results$tp, na.rm = TRUE),
    mean_fp = mean(all_results$fp, na.rm = TRUE),
    mean_recall = mean(all_results$recall, na.rm = TRUE),
    mean_precision = mean(all_results$precision, na.rm = TRUE),
    mean_obs_fdr = mean(all_results$obs_fdr, na.rm = TRUE),
    mean_runtime = mean(all_results$runtime, na.rm = TRUE)
)

write.csv(summary_stats, summary_csv, row.names = FALSE)
saveRDS(summary_stats, summary_rds)

cat("\n=== Simulation 20 Summary ===\n")
print(summary_stats)

cat(sprintf("\nSummary saved to %s and %s\n", summary_csv, summary_rds))
