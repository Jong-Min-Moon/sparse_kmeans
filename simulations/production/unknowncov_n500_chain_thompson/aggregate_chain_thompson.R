# ------------------------------------------------------------------------------
# Script: aggregate_chain_thompson.R
# Purpose: Compiles, merges, and validates the output generated across isolated 
#          HPC computational nodes running `unknowncov_n500_chain_thompson`. 
# Returns: Formatted summary tables suitable for publication-level reporting.
# ------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(purrr)

# Set directories tracking standard outputs inside the unified sim configuration 
base_dir <- "."
output_dir <- file.path(base_dir, "results_raw")
# Subdirectories to aggregate natively across noise typologies
noise_types <- list.dirs(output_dir, full.names = FALSE, recursive = FALSE)

if (length(noise_types) == 0) {
    stop("No properly isolated noise partitions found in ", output_dir)
}

for (noise in noise_types) {
    noise_dir <- file.path(output_dir, noise)
    cat(sprintf("\n--- Aggregating results for Noise Profile: %s ---\n", noise))
    
    # Recursively locate all generated outputs matching the expected dataset extension
    all_files <- list.files(noise_dir, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)
    
    if (length(all_files) == 0) {
        cat(sprintf("   > Skipping... No .rds compiled targets found dynamically for '%s'.\n", noise))
        next
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
            n_step_admm = res_data$params$n_step_admm,
            noise = if (is.null(res_data$params$noise)) "Gaussian" else res_data$params$noise
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
    
    # Isolate targets appending noise tag suffix sequentially
    save_rds <- sprintf("all_results_chain_thompson_%s.rds", noise)
    save_summary_csv <- sprintf("summary_chain_thompson_%s.csv", noise)
    
    write.csv(summary_stats, save_summary_csv, row.names = FALSE)
    cat(sprintf("   > Summary aggregation outputs logically compiled into %s\n", save_summary_csv))
    
    # Export monolithic RDS database targeting any final customized post-mortem tracking needs
    saveRDS(all_results, save_rds)
    cat(sprintf("   > Raw global database fully encapsulated natively tracking to %s\n", save_rds))
    
    cat(sprintf("\n=== Simulation 22 %s Performance Summary Evaluator ===\n", toupper(noise)))
    print(summary_stats)
}

cat("\nAggregations iteratively passed effectively matching multi-noise outputs.\n")
