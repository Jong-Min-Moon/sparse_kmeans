# Aggregate Simulation 06 Results (Permutation FDR)

library(dplyr)
library(tidyr)
library(purrr)

# Set paths
base_dir <- "."
output_dir <- file.path(base_dir, "output")
summary_file <- "summary_sim06.csv"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir, "\nRun retrieve_sim06.ps1 first.")
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

    # New params
    fdr_target <- res_data$params$fdr
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
        n_selected = n_selected,
        recall = recall,
        precision = precision,
        obs_fdr = obs_fdr,
        p = res_data$params$p,
        rho = res_data$params$rho
    )
})

all_results <- bind_rows(results_list)

summary_stats <- all_results %>%
    summarise(
        n_reps = n(),
        mean_accuracy = mean(accuracy, na.rm = TRUE),
        mean_ari = mean(ari, na.rm = TRUE),
        mean_tp = mean(tp, na.rm = TRUE),
        mean_fp = mean(fp, na.rm = TRUE),
        mean_recall = mean(recall, na.rm = TRUE),
        mean_precision = mean(precision, na.rm = TRUE),
        mean_obs_fdr = mean(obs_fdr, na.rm = TRUE), # Check if this matches Target FDR (0.1)
        perfect_acc_count = sum(accuracy >= 0.999, na.rm = TRUE)
    )

write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary saved to %s\n", summary_file))

cat("\n=== Simulation 06 Summary ===\n")
print(summary_stats)

saveRDS(all_results, "all_results_sim06.rds")
