# ------------------------------------------------------------------------------
# Script: aggregate_sim01.R
# Purpose: Compiles, merges, and validates sim01 (cluster_greedy, identity cov)
#          output generated across isolated HPC nodes.
#          Compatible with the results_raw/p{dim}/ directory structure used
#          by sim_22_thompson_identity_laplace and all modern simulations.
# Returns: summary_sim01.csv and all_results_sim01.rds
# ------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(purrr)

base_dir     <- "."
output_dir   <- file.path(base_dir, "results_raw")
summary_file <- "summary_sim01.csv"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir,
         "\nRun retrieve_sim01.ps1 using PowerShell first.")
}

# Recursively locate all .rds output files
all_files <- list.files(output_dir, pattern = "\\.rds$",
                        full.names = TRUE, recursive = TRUE)

if (length(all_files) == 0) {
    stop("No .rds files found in ", output_dir)
}

cat(sprintf("Found %d result file(s). Aggregating...\n", length(all_files)))

# Load and flatten each result object
results_list <- purrr::map(all_files, function(f) {
    res <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(res)) {
        warning(sprintf("Could not read: %s", f))
        return(NULL)
    }
    data.frame(
        job_id     = res$job_id,
        accuracy   = res$accuracy,
        runtime    = res$runtime,
        n_selected = res$L,
        tp         = res$tp,
        fp         = res$fp,
        recall     = res$recall,
        precision  = res$precision,
        p          = res$params$p,
        n          = res$params$n,
        separation = res$params$separation,
        noise      = res$params$noise
    )
})

all_results <- bind_rows(results_list)

cat(sprintf("Successfully loaded %d replicates across %d unique (sep, p, noise) configs.\n",
            nrow(all_results),
            nrow(distinct(all_results, separation, p, noise))))

# Summary statistics — dynamically grouped; no hardcoded p or method names
summary_stats <- all_results %>%
    group_by(separation, p, noise) %>%
    summarize(
        n_reps          = n(),
        mean_accuracy   = mean(accuracy,   na.rm = TRUE),
        sd_accuracy     = sd(accuracy,     na.rm = TRUE),
        mean_tp         = mean(tp,         na.rm = TRUE),
        mean_fp         = mean(fp,         na.rm = TRUE),
        mean_n_selected = mean(n_selected, na.rm = TRUE),
        mean_recall     = mean(recall,     na.rm = TRUE),
        mean_precision  = mean(precision,  na.rm = TRUE),
        mean_runtime    = mean(runtime,    na.rm = TRUE),
        .groups = "drop"
    ) %>%
    arrange(noise, separation, p)

write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary saved to %s\n", summary_file))

cat("\n=== Simulation 01 Performance Summary ===\n")
print(summary_stats)

# Export monolithic RDS for downstream custom analysis
saveRDS(all_results, "all_results_sim01.rds")
cat("Full results saved to all_results_sim01.rds\n")

