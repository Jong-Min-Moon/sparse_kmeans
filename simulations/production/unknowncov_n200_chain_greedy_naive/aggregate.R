# ------------------------------------------------------------------------------
# aggregate.R
# Compiles unknowncov_n200_chain_greedy_naive simulation outputs from HPC.
#
# Input layout (written by driver_gaussian.R / driver_laplace.R):
#   results_raw/<noise>/p<p>/sim_job<id>_p<p>.rds
#
# Each RDS is a one-row data.frame with columns:
#   job_id, p, n, sep, rho, noise,
#   accuracy, n_selected, tp, fp, recall, precision, runtime
#
# Outputs (per noise type):
#   aggregated_greedy_<noise>.rds  — full flat table of all replicates
#   summary_greedy_<noise>.rds     — summarised by (sep, p)
#   summary_greedy_<noise>.csv     — same, CSV
# ------------------------------------------------------------------------------
library(dplyr)
library(tidyr)
library(purrr)

base_dir   <- "."
output_dir <- file.path(base_dir, "results_raw")

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir,
         "\nRun retrieve.ps1 first.")
}

# Discover noise partitions (sub-directories of results_raw/)
noise_types <- list.dirs(output_dir, full.names = FALSE, recursive = FALSE)

if (length(noise_types) == 0) {
    stop("No noise partitions found in ", output_dir)
}

for (noise in noise_types) {
    noise_dir <- file.path(output_dir, noise)
    cat(sprintf("\n--- Aggregating results for noise: %s ---\n", noise))

    all_files <- list.files(noise_dir, pattern = "\\.rds$",
                            full.names = TRUE, recursive = TRUE)

    if (length(all_files) == 0) {
        cat(sprintf("No result files found for noise type '%s'.\n", noise))
        next
    }

    cat(sprintf("Found %d result file(s). Loading...\n", length(all_files)))

    results_list <- purrr::map(all_files, function(f) {
        res <- tryCatch(readRDS(f), error = function(e) NULL)
        if (is.null(res)) {
            warning(sprintf("Could not read: %s", f))
            return(NULL)
        }
        res  # already a one-row data.frame
    })

    all_results <- bind_rows(results_list)

    cat(sprintf(
        "Loaded %d replicates across %d unique (sep, p) configurations.\n",
        nrow(all_results),
        nrow(distinct(all_results, sep, p))
    ))

    # Summary statistics grouped by (sep, p)
    summary_stats <- all_results %>%
        group_by(sep, p, rho) %>%
        summarize(
            n_reps           = n(),
            mean_accuracy    = mean(accuracy,    na.rm = TRUE),
            sd_accuracy      = sd(accuracy,      na.rm = TRUE),
            mean_n_selected  = mean(n_selected,  na.rm = TRUE),
            mean_tp          = mean(tp,           na.rm = TRUE),
            mean_fp          = mean(fp,           na.rm = TRUE),
            mean_recall      = mean(recall,       na.rm = TRUE),
            mean_precision   = mean(precision,    na.rm = TRUE),
            mean_runtime     = mean(runtime,      na.rm = TRUE),
            .groups = "drop"
        ) %>%
        arrange(sep, p)

    print(summary_stats)

    # Save outputs
    save_full    <- sprintf("aggregated_greedy_%s.rds",  noise)
    save_sum_rds <- sprintf("summary_greedy_%s.rds",     noise)
    save_sum_csv <- sprintf("summary_greedy_%s.csv",     noise)

    saveRDS(all_results,   save_full)
    saveRDS(summary_stats, save_sum_rds)
    write.csv(summary_stats, save_sum_csv, row.names = FALSE)

    cat(sprintf(
        "Aggregation complete for '%s'.\n  -> %s\n  -> %s\n  -> %s\n",
        noise, save_full, save_sum_rds, save_sum_csv
    ))
}

cat("\nGlobal aggregation completed successfully.\n")
