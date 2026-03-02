# Aggregate results for sim_15_witten_arias

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
            witten_acc = res$witten$acc,
            witten_runtime = res$witten$runtime,
            arias_acc = res$arias$acc,
            arias_runtime = res$arias$runtime
        )
    }
}

df <- bind_rows(all_res)

summary_df <- df %>%
    group_by(sep) %>%
    summarize(
        n_runs = n(),
        mean_witten_acc = mean(witten_acc, na.rm = TRUE),
        sd_witten_acc = sd(witten_acc, na.rm = TRUE),
        mean_arias_acc = mean(arias_acc, na.rm = TRUE),
        sd_arias_acc = sd(arias_acc, na.rm = TRUE),
        mean_witten_runtime = mean(witten_runtime, na.rm = TRUE),
        mean_arias_runtime = mean(arias_runtime, na.rm = TRUE)
    )

print(summary_df)

saveRDS(df, file = "aggregated_sim15.rds")
saveRDS(summary_df, file = "summary_sim15.rds")
write.csv(summary_df, file = "summary_sim15.csv", row.names = FALSE)

cat("Aggregation complete. Saved to aggregated_sim15.rds, summary_sim15.rds, and summary_sim15.csv.\n")
