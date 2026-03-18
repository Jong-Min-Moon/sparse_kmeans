# ------------------------------------------------------------------------------
# Script: aggregate_greedy.R
# Purpose: Aggregates results for the cluster_greedy varying p simulation.
# ------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(purrr)

# Set directories
base_dir <- "."
output_dir <- file.path(base_dir, "results_raw")
summary_file <- "summary_greedy.csv"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir)
}

# Recursively locate all .rds files
all_files <- list.files(output_dir, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)

if (length(all_files) == 0) {
    stop("No properly executed .rds files found matching compilation targets in ", output_dir)
}

cat(sprintf("Found %d isolated replication sequence artifacts. Aggregating...\n", length(all_files)))

results_list <- purrr::map(all_files, function(f) {
    res_data <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(res_data)) return(NULL)
    
    data.frame(
        job_id = res_data$job_id,
        accuracy = res_data$accuracy,
        ari = res_data$ari,
        runtime = res_data$runtime,
        n_selected = res_data$n_selected,
        tp = res_data$tp,
        fp = res_data$fp,
        recall = res_data$recall,
        precision = res_data$precision,
        p = res_data$p,
        fdr = res_data$fdr,
        n = res_data$params$n,
        separation = res_data$params$separation
    )
})

all_results <- bind_rows(results_list)

# Compute summary statistics
summary_stats <- all_results %>%
    group_by(p) %>%
    summarize(
        n_reps = n(),
        mean_accuracy = mean(accuracy, na.rm = TRUE),
        mean_ari = mean(ari, na.rm = TRUE),
        mean_tp = mean(tp, na.rm = TRUE),
        mean_fp = mean(fp, na.rm = TRUE),
        mean_n_selected = mean(n_selected, na.rm = TRUE),
        mean_recall = mean(recall, na.rm = TRUE),
        mean_precision = mean(precision, na.rm = TRUE),
        mean_runtime = mean(runtime, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    arrange(p)

write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary aggregation outputs saved to %s\n", summary_file))

cat("\n=== cluster_greedy Performance Summary ===\n")
print(summary_stats)

# Export monolithic RDS
saveRDS(all_results, "all_results_greedy.rds")
