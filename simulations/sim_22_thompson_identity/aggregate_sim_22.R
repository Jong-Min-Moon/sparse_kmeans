# ------------------------------------------------------------------------------
# Script: aggregate_sim_22.R
# Purpose: Compiles, merges, and validates the output generated across isolated 
#          HPC computational nodes running `sim_22_thompson_identity`. 
# Returns: Formatted summary tables suitable for publication-level reporting.
# ------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(purrr)

# Set directories tracking standard outputs inside the unified sim configuration 
base_dir <- "."
output_dir <- file.path(base_dir, "results_raw")
summary_file <- "summary_sim_22.csv"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir, "\nRun retrieve_sim_22.ps1 using Powershell first.")
}

# Recursively locate all generated outputs matching the expected dataset extension
all_files <- list.files(output_dir, pattern = "\\.rds$", full.names = TRUE)

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
summary_stats <- data.frame(
    n_reps = nrow(all_results),
    mean_accuracy = mean(all_results$accuracy, na.rm = TRUE),
    sd_accuracy = sd(all_results$accuracy, na.rm = TRUE),
    mean_tp = mean(all_results$tp, na.rm = TRUE),
    mean_fp = mean(all_results$fp, na.rm = TRUE),
    mean_n_selected = mean(all_results$n_selected, na.rm = TRUE),
    mean_recall = mean(all_results$recall, na.rm = TRUE),
    mean_precision = mean(all_results$precision, na.rm = TRUE),
    mean_runtime = mean(all_results$runtime, na.rm = TRUE)
)

write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary aggregation outputs saved to %s\n", summary_file))

cat("\n=== Simulation 22 Performance Summary Evaluator ===\n")
print(summary_stats)

# Export monolithic RDS database targeting any final customized post-mortem tracking needs
saveRDS(all_results, "all_results_sim_22.rds")
