# -------------------------------------------------------------
# aggregate_sim_19.R
# Produces summary tables for mean FDR, Acc, Power, and saves diagnostic CSV output.
# -------------------------------------------------------------

library(dplyr)
library(tidyr)

aggregated_dir <- "results_aggregated"
input_file <- file.path(aggregated_dir, "all_results.rds")
summary_output <- file.path(aggregated_dir, "summary_sim_19_greedy_naive_bayes.csv")

if (!file.exists(input_file)) {
    stop("Combined results file not found! Run retrieve_sim_19.R first.")
}

all_results <- readRDS(input_file)

if (nrow(all_results) == 0 || !("sep" %in% colnames(all_results))) {
    stop("Input format invalid or completely empty dataset detected.")
}

cat("Computing parameter grid aggregations natively...\n")

# Use Base R + Dplyr aggregation natively avoiding defunct dependencies from older R installs
summary_df <- do.call(rbind, lapply(split(all_results, list(all_results$sep, all_results$fdr_level)), function(sub_df) {
    if(nrow(sub_df) == 0) return(NULL)
    data.frame(
        sep = sub_df$sep[1],
        fdr_target = sub_df$fdr_level[1],
        n_completed = nrow(sub_df),
        mean_acc = mean(sub_df$accuracy, na.rm = TRUE),
        sd_acc = sd(sub_df$accuracy, na.rm = TRUE),
        mean_L = mean(sub_df$L, na.rm = TRUE),
        mean_empirical_fdr = mean(sub_df$empirical_fdr, na.rm = TRUE),
        sd_empirical_fdr = sd(sub_df$empirical_fdr, na.rm = TRUE),
        mean_power = mean(sub_df$power, na.rm = TRUE),
        sd_power = sd(sub_df$power, na.rm = TRUE),
        mean_runtime = mean(sub_df$runtime, na.rm = TRUE)
    )
}))

summary_df <- summary_df[order(summary_df$sep, summary_df$fdr_target), ]
rownames(summary_df) <- NULL

cat("\n=== sim_19_greedy_naive_bayes Diagnostic Results (30 Jobs Per Sep) ===\n")
print(summary_df, digits = 4)

write.csv(summary_df, file = summary_output, row.names = FALSE)
cat(sprintf("\nAggregation complete. Outputs saved securely in '%s'.\n", summary_output))

# Diagnostic Warning Alert natively 
violations <- summary_df %>% filter(mean_empirical_fdr > fdr_target * 1.5)
if (nrow(violations) > 0) {
    warning("Severe Empirical FDR boundary violations detected!")
    print(violations %>% select(sep, mean_empirical_fdr, fdr_target))
}
