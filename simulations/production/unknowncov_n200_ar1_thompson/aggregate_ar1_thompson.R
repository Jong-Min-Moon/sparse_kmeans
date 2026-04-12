# ------------------------------------------------------------------------------
# Script: aggregate_ar1_thompson.R
# Purpose: Compiles, merges, and validates the output generated across isolated 
#          HPC computational nodes running `unknowncov_n200_ar1_thompson`. 
# Returns: Formatted summary tables suitable for publication-level reporting.
# ------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(purrr)

# Set directories tracking standard outputs inside the unified sim configuration 
base_dir <- "."
output_dir <- file.path(base_dir, "results_raw")
summary_file <- "summary_ar1_thompson.csv"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir, "\nRun retrieve_ar1_thompson.ps1 using Powershell first.")
}

# Recursively locate all generated outputs matching the expected dataset extension
all_files <- list.files(output_dir, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)

if (length(all_files) == 0) {
    stop("No properly executed .rds files found matching compilation targets in ", output_dir)
}

cat(sprintf("Found %d isolated replication sequence artifacts. Aggregating natively...\n", length(all_files)))

results_list <- purrr::map(all_files, function(f) {
    # Fail-safe catching mechanisms allowing incomplete chunks to gracefully crash rather than halting batch extraction
    res_data <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(res_data)) return(NULL)
    
    # Isolate relevant scalars configured natively from the simulation framework driver
    data.frame(
        job_id = res_data$job_id,
        accuracy = res_data$accuracy,
        runtime = res_data$runtime,
        n_selected = res_data$L,
        tp = res_data$tp,
        fp = res_data$fp,
        recall = res_data$recall,
        precision = res_data$precision,
        p = res_data$params$p,
        n = res_data$params$n,
        separation = res_data$params$separation,
        pval = res_data$pval,
        n_step_admm = res_data$params$n_step_admm
    )
})

# Compile frame arrays mapped to single cohesive tables vertically
all_results <- bind_rows(results_list)

# Compute descriptive summary statistical metrics representing true method performance behavior mathematically 
summary_stats <- all_results %>%
    group_by(separation, p) %>%
    summarize(
        n_reps = n(),
        mean_accuracy = mean(accuracy, na.rm = TRUE),
        sd_accuracy = sd(accuracy, na.rm = TRUE),
        mean_tp = mean(tp, na.rm = TRUE),
        mean_fp = mean(fp, na.rm = TRUE),
        mean_n_selected = mean(n_selected, na.rm = TRUE),
        mean_recall = mean(recall, na.rm = TRUE),
        mean_precision = mean(precision, na.rm = TRUE),
        mean_runtime = mean(runtime, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    arrange(separation, p)

write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary aggregation outputs saved to %s\n", summary_file))

cat("\n=== AR(1) Thompson Performance Summary Evaluator ===\n")
print(summary_stats)

# Export monolithic RDS database targeting any final customized post-mortem tracking needs
saveRDS(all_results, "all_results_ar1_thompson.rds")
