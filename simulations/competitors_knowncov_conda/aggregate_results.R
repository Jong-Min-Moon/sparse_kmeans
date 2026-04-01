# ------------------------------------------------------------------
# aggregate_results.R
# Aggregates unified known-covariance simulation outputs.
# ------------------------------------------------------------------
library(dplyr)
library(tidyr)

# Ensure we are in the script's directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

# Simulation settings to aggregate
noise_types <- c("laplace") # Only laplace for now
p_values <- c(200, 500, 1000, 3000, 12000)
sep_values <- c(4)

# Results storage
all_results <- list()

# Process each noise type
for (noise in noise_types) {
    # Path to results for this noise type
    # Changed from absolute/fixed path to relative results folder
    results_dir <- file.path("results", noise)
    if (!dir.exists(results_dir)) {
        cat(sprintf("Skipping noise type '%s': directory not found.\n", noise))
        next
    }
    
    # Pattern matches sim_job{id}_p{dimension}.rds
    files <- list.files(results_dir, pattern = "^sim_job\\d+_p\\d+\\.rds$", full.names = TRUE)
    
    if (length(files) == 0) {
        cat(sprintf("No result files found for noise type '%s'.\n", noise))
        next
    }
    
    cat(sprintf("Aggregating %d files for noise type '%s'...\n", length(files), noise))
    
    # Bind rows cleanly because driver outputs standardized flat dataframes
    all_res <- lapply(files, readRDS)
    df <- bind_rows(all_res)
    
    # Summary by Dimension (p) and Separation (sep)
    summary_df <- df %>%
        group_by(p, sep) %>%
        summarize(
            n_runs = n(),
            # Witten Metrics
            mean_acc_witten = mean(accuracy_witten, na.rm = TRUE),
            sd_acc_witten = sd(accuracy_witten, na.rm = TRUE),
            mean_rt_witten = mean(runtime_witten, na.rm = TRUE),
    
            # Arias Metrics
            mean_acc_arias = mean(accuracy_arias, na.rm = TRUE),
            sd_acc_arias = sd(accuracy_arias, na.rm = TRUE),
            mean_rt_arias = mean(runtime_arias, na.rm = TRUE),
    
            # IF-PCA Metrics
            mean_acc_ifpca = mean(accuracy_ifpca, na.rm = TRUE),
            sd_acc_ifpca = sd(accuracy_ifpca, na.rm = TRUE),
            mean_rt_ifpca = mean(runtime_ifpca, na.rm = TRUE),
            mean_L_ifpca = mean(ifpca_L, na.rm = TRUE),
            .groups = "drop"
        )
    
    print(sprintf("Summary for %s:", noise))
    print(summary_df)
    
    # Save results specifically for this noise type
    saveRDS(df, file = sprintf("aggregated_knowncov_%s.rds", noise))
    saveRDS(summary_df, file = sprintf("summary_knowncov_%s.rds", noise))
    write.csv(summary_df, file = sprintf("summary_knowncov_%s.csv", noise), row.names = FALSE)
    
    cat(sprintf("Aggregation complete for '%s'. Saved to results directory.\n\n", noise))
}
