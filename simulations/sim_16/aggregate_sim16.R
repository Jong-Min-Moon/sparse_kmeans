# Aggregate results for sim_16 (IF-PCA)

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

all_res <- list()

for (f in files) {
    res <- tryCatch(
        {
            readRDS(f)
        },
        error = function(e) {
            warning(paste("Error reading", f))
            NULL
        }
    )

    if (!is.null(res)) {
        all_res[[length(all_res) + 1]] <- data.frame(
            job_id = res$job_id,
            p = res$params$p,
            n = res$params$n,
            sep = res$params$sep,
            rho = res$params$rho,
            ifpca_acc = res$ifpca$acc,
            ifpca_L = res$ifpca$L,
            ifpca_runtime = res$ifpca$runtime
        )
    }
}

df <- bind_rows(all_res)

summary_df <- df %>%
    group_by(sep) %>%
    summarize(
        n_runs = n(),
        mean_ifpca_acc = mean(ifpca_acc, na.rm = TRUE),
        sd_ifpca_acc = sd(ifpca_acc, na.rm = TRUE),
        mean_L = mean(ifpca_L, na.rm = TRUE),
        sd_L = sd(ifpca_L, na.rm = TRUE),
        mean_ifpca_runtime = mean(ifpca_runtime, na.rm = TRUE)
    )

print(summary_df)

saveRDS(df, file = "aggregated_sim16.rds")
saveRDS(summary_df, file = "summary_sim16.rds")
write.csv(summary_df, file = "summary_sim16.csv", row.names = FALSE)

cat("Aggregation complete. Saved to aggregated_sim16.rds, summary_sim16.rds, and summary_sim16.csv.\n")
