# Aggregate Simulation 18 Results (Thompson Sampling)

library(dplyr)
library(tidyr)
library(purrr)

# Set paths
base_dir <- "."
output_dir <- file.path(base_dir, "output")
summary_file <- "summary_sim18.csv"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir, "\nRun retrieve_sim18.sh first.")
}

# List all RDS files
all_files <- list.files(output_dir, pattern = "\\.rds$", full.names = TRUE)

if (length(all_files) == 0) {
    stop("No .rds files found in ", output_dir)
}

cat(sprintf("Found %d result files. Aggregating...\n", length(all_files)))

results_list <- purrr::map(all_files, function(f) {
    res_data <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(res_data)) {
        return(NULL)
    }

    job_id <- res_data$job_id
    ari <- res_data$ari
    acc <- res_data$acc
    tp <- res_data$tp
    fp <- res_data$fp
    overlap_at_100 <- res_data$overlap_at_100

    # New params
    c_val_target <- res_data$params$C_val
    n_perms <- res_data$params$n_perms

    n_selected <- tp + fp
    # Support size is 10
    recall <- if (tp > 0) tp / 10 else 0
    precision <- if (n_selected > 0) tp / n_selected else 0

    # Calculate Observed FDR = 1 - Precision
    obs_fdr <- 1 - precision

    data.frame(
        job_id = job_id,
        accuracy = acc,
        ari = ari,
        tp = tp,
        fp = fp,
        overlap_at_100 = overlap_at_100,
        n_selected = n_selected,
        recall = recall,
        precision = precision,
        obs_fdr = obs_fdr,
        p = res_data$params$p,
        rho = res_data$params$rho,
        C_val = c_val_target
    )
})

all_results <- bind_rows(results_list)

summary_stats <- data.frame(
    n_reps = nrow(all_results),
    mean_accuracy = mean(all_results$accuracy, na.rm = TRUE),
    mean_ari = mean(all_results$ari, na.rm = TRUE),
    mean_tp = mean(all_results$tp, na.rm = TRUE),
    mean_fp = mean(all_results$fp, na.rm = TRUE),
    mean_overlap_at_100 = mean(all_results$overlap_at_100, na.rm = TRUE),
    mean_recall = mean(all_results$recall, na.rm = TRUE),
    mean_precision = mean(all_results$precision, na.rm = TRUE),
    mean_obs_fdr = mean(all_results$obs_fdr, na.rm = TRUE),
    perfect_acc_count = sum(all_results$accuracy >= 0.999, na.rm = TRUE)
)

write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary saved to %s\n", summary_file))

cat("\n=== Simulation 18 Summary ===\n")
print(summary_stats)

saveRDS(all_results, "all_results_sim18.rds")
