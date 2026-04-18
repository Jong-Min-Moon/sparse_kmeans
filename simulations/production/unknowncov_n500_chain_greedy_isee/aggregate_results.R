# ------------------------------------------------------------------
# aggregate_results.R
# Aggregates Greedy ISEE simulation outputs from HPC.
# Handles the results_raw hierarchy (noise/p/sim_job...)
# ------------------------------------------------------------------
library(dplyr)
library(tidyr)
library(purrr)

# Ensure we interact with the correct results_raw even if called from another CWD natively
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}
base_dir <- "."
output_dir <- file.path(base_dir, "results_raw")
summary_file_prefix <- "summary_hpc"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir, "\nRun retrieve_unknowncov.ps1 first.")
}

# Subdirectories to aggregate
noise_types <- list.dirs(output_dir, full.names = FALSE, recursive = FALSE)

if (length(noise_types) == 0) {
    stop("No noise partitions found in ", output_dir)
}

for (noise in noise_types) {
    noise_dir <- file.path(output_dir, noise)
    cat(sprintf("\n--- Aggregating results for Noise: %s ---\n", noise))
    
    # Locate all generated outputs
    all_files <- list.files(noise_dir, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)
    
    if (length(all_files) == 0) {
        cat(sprintf("No result files found for noise type '%s'.\n", noise))
        next
    }
    
    cat(sprintf("Found %d isolated replication sequence artifacts. Aggregating natively...\n", length(all_files)))
    
    # Read and bind all results
    results_list <- purrr::map(all_files, function(f) {
        res_data <- tryCatch(readRDS(f), error = function(e) NULL)
        if (is.null(res_data)) return(NULL)
        return(res_data)
    })
    
    all_results <- bind_rows(results_list)
    
    # Summary by Dimension (p) and Separation (sep)
    # The columns derived from sim_*.R are:
    # job_id, p, n, sep, rho, accuracy, ari, tp, fp, runtime, n_selected
    
    summary_stats <- all_results %>%
        group_by(p, sep) %>%
        summarize(
            n_reps = n(),
            mean_acc = mean(accuracy, na.rm = TRUE),
            sd_acc = sd(accuracy, na.rm = TRUE),
            mean_ari = mean(ari, na.rm = TRUE),
            sd_ari = sd(ari, na.rm = TRUE),
            mean_tp = mean(tp, na.rm = TRUE),
            mean_fp = mean(fp, na.rm = TRUE),
            mean_runtime = mean(runtime, na.rm = TRUE),
            mean_selected = mean(n_selected, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        arrange(sep, p)
    
    print(summary_stats)
    
    # Save outputs
    save_rds <- sprintf("aggregated_hpc_%s.rds", noise)
    save_summary_rds <- sprintf("%s_%s.rds", summary_file_prefix, noise)
    save_summary_csv <- sprintf("%s_%s.csv", summary_file_prefix, noise)
    
    saveRDS(all_results, save_rds)
    saveRDS(summary_stats, save_summary_rds)
    write.csv(summary_stats, save_summary_csv, row.names = FALSE)
    
    cat(sprintf("Aggregation complete for '%s'. Saved to %s, %s, %s\n", noise, save_rds, save_summary_rds, save_summary_csv))
}

cat("\nGlobal aggregation completed successfully.\n")
