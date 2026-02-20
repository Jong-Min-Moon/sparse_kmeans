# Aggregate Simulation 05 Results
# Loads all .rds files from output directory and creates a summary.

library(dplyr)
library(tidyr)
library(purrr)
library(mclust)

# Set paths
base_dir <- "."
output_dir <- file.path(base_dir, "output")
summary_file <- "summary_sim05.csv"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir, "\nRun retrieve_sim05.ps1 first to download results.")
}

# List all RDS files
all_files <- list.files(output_dir, pattern = "\\.rds$", full.names = TRUE)

if (length(all_files) == 0) {
    stop("No .rds files found in ", output_dir)
}

cat(sprintf("Found %d result files. Aggregating...\n", length(all_files)))

# Load and combine data
results_list <- purrr::map(all_files, function(f) {
    res_data <- tryCatch(
        {
            readRDS(f)
        },
        error = function(e) {
            warning(sprintf("Error reading %s: %s", f, e$message))
            NULL
        }
    )

    if (is.null(res_data)) {
        return(NULL)
    }

    # Extract data
    # Structure from driver.R:
    # list(res = res, ari = ari, acc = acc, tp = tp, fp = fp, job_id = job_id, params = list(...))

    job_id <- res_data$job_id
    ari <- res_data$ari
    acc <- res_data$acc
    tp <- res_data$tp
    fp <- res_data$fp

    # Parameters
    p_val <- res_data$params$p
    rho_val <- res_data$params$rho
    sep_val <- res_data$params$sep
    n_val <- res_data$params$n

    # Calculate additional metrics
    n_selected <- tp + fp
    # Support size is 10 in driver.R
    recall <- if (tp > 0) tp / 10 else 0
    precision <- if (n_selected > 0) tp / n_selected else 0

    data.frame(
        job_id = job_id,
        accuracy = acc,
        ari = ari,
        tp = tp,
        fp = fp,
        n_selected = n_selected,
        recall = recall,
        precision = precision,
        p = p_val,
        rho = rho_val,
        sep = sep_val,
        n = n_val
    )
})

all_results <- bind_rows(results_list)

# Calculate overall summaries
summary_stats <- all_results %>%
    summarise(
        n_reps = n(),
        mean_accuracy = mean(accuracy, na.rm = TRUE),
        sd_accuracy = sd(accuracy, na.rm = TRUE),
        median_accuracy = median(accuracy, na.rm = TRUE),
        mean_ari = mean(ari, na.rm = TRUE),
        sd_ari = sd(ari, na.rm = TRUE),
        mean_tp = mean(tp, na.rm = TRUE),
        sd_tp = sd(tp, na.rm = TRUE),
        mean_fp = mean(fp, na.rm = TRUE),
        sd_fp = sd(fp, na.rm = TRUE),
        mean_recall = mean(recall, na.rm = TRUE),
        sd_recall = sd(recall, na.rm = TRUE),
        mean_precision = mean(precision, na.rm = TRUE),
        sd_precision = sd(precision, na.rm = TRUE),
        perfect_acc_count = sum(accuracy >= 0.999, na.rm = TRUE),
        perfect_acc_rate = mean(accuracy >= 0.999, na.rm = TRUE)
    )

# Save summary
write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary saved to %s\n", summary_file))

# Print summary to console
cat("\n=== Simulation 05 (n=200) Summary ===\n")
print(summary_stats)

# Distribution of Accuracy
cat("\n=== Accuracy Distribution ===\n")
cat(sprintf("Min: %.4f\n", min(all_results$accuracy, na.rm = TRUE)))
cat(sprintf("Q1:  %.4f\n", quantile(all_results$accuracy, 0.25, na.rm = TRUE)))
cat(sprintf("Median: %.4f\n", median(all_results$accuracy, na.rm = TRUE)))
cat(sprintf("Q3:  %.4f\n", quantile(all_results$accuracy, 0.75, na.rm = TRUE)))
cat(sprintf("Max: %.4f\n", max(all_results$accuracy, na.rm = TRUE)))

# Save full results for further analysis
saveRDS(all_results, "all_results_sim05.rds")
cat(sprintf("\nFull results saved to all_results_sim05.rds\n"))
