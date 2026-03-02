# ------------------------------------------------------------------
# aggregate_results.R
# Aggregates results_raw for sim_18 parameter grid
# ------------------------------------------------------------------
library(dplyr)
library(tidyr)

# Ensure we are in the script's directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

results_dir <- "results_raw/"
files <- list.files(results_dir, pattern = "^sim_id\\d+_sep\\d+_pval\\d+\\.\\d+\\.rds$", full.names = TRUE)

if (length(files) == 0) {
    stop("No result files found in 'results_raw/' directory.")
}

# Bind rows cleanly
all_res <- lapply(files, readRDS)

# Extract nested parameters directly
df_list <- lapply(all_res, function(x) {
    data.frame(
        job_id = x$job_id,
        sep = x$sep,
        pval = x$pval,
        accuracy = x$accuracy,
        L = x$L,
        runtime = x$runtime
    )
})

df <- bind_rows(df_list)

summary_df <- do.call(rbind, lapply(split(df, list(df$sep, df$pval)), function(sub_df) {
    if(nrow(sub_df) == 0) return(NULL)
    data.frame(
        sep = sub_df$sep[1],
        pval = sub_df$pval[1],
        n_runs = nrow(sub_df),
        mean_acc = mean(sub_df$accuracy, na.rm = TRUE),
        sd_acc = sd(sub_df$accuracy, na.rm = TRUE),
        mean_L = mean(sub_df$L, na.rm = TRUE),
        sd_L = sd(sub_df$L, na.rm = TRUE),
        mean_rt = mean(sub_df$runtime, na.rm = TRUE)
    )
}))
summary_df <- summary_df[order(summary_df$sep, -summary_df$pval), ]
rownames(summary_df) <- NULL

print(summary_df)

dir.create("results_aggregated", showWarnings = FALSE)
saveRDS(df, file = "results_aggregated/aggregated_sim18.rds")
write.csv(summary_df, file = "results_aggregated/summary_sim18.csv", row.names = FALSE)

cat("Aggregation complete. Saved to results_aggregated/.\n")
