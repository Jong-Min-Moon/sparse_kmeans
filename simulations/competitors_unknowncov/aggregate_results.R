# ------------------------------------------------------------------
# aggregate_results.R
# Aggregates unified sim_17 multidimensional outputs.
# ------------------------------------------------------------------
library(dplyr)
library(tidyr)

# Ensure we are in the script's directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

results_dir <- "results/"
files <- list.files(results_dir, pattern = "^sim_id\\d+_sep\\d+\\.rds$", full.names = TRUE)

if (length(files) == 0) {
    stop("No result files found in 'results/' directory.")
}

# Bind rows cleanly because driver outputs standardized flat dataframes
all_res <- lapply(files, readRDS)
df <- bind_rows(all_res)

summary_df <- df %>%
    group_by(sep) %>%
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
        mean_L_ifpca = mean(ifpca_L, na.rm = TRUE)
    )

print(summary_df)

saveRDS(df, file = "aggregated_sim17.rds")
saveRDS(summary_df, file = "summary_sim17.rds")
write.csv(summary_df, file = "summary_sim17.csv", row.names = FALSE)

cat("Aggregation complete. Saved to aggregated_sim17.rds, summary_sim17.rds, and summary_sim17.csv.\n")
