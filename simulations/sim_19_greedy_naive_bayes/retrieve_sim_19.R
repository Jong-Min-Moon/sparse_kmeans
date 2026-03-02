# -------------------------------------------------------------
# retrieve_sim_19.R
# Collects all completed replicate result files sequentially, combining them natively.
# Scrapes SLURM log files as a fallback extracting Runtime + Balanced Accuracy metrics.
# -------------------------------------------------------------
library(dplyr)
library(tidyr)
library(purrr)

# Config to match deploy grid
expected_separations <- c(4, 5, 6)
expected_jobs <- 30
fdr_level <- 0.4
total_expected <- length(expected_separations) * expected_jobs

results_dir <- "results_raw"
logs_dir <- "logs/sim_19_greedy_naive_bayes"
output_combined_file <- "all_results.rds"

if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Build a master grid dataframe to track expectations
grid <- expand.grid(job_id = 1:expected_jobs, sep = expected_separations)

cat(sprintf("Scanning for expected %d result files...\n", nrow(grid)))

results_list <- lapply(1:nrow(grid), function(i) {
  j_id <- grid$job_id[i]
  sp <- grid$sep[i]
  
  rds_file <- file.path(results_dir, sprintf("sim_id%d_sep%d_fdr%.2f.rds", j_id, sp, fdr_level))
  log_file <- file.path(logs_dir, sprintf("sim_id%d_sep%d.out", j_id, sp))
  
  # Try loading native RDS first
  res_data <- tryCatch(if(file.exists(rds_file)) readRDS(rds_file) else NULL, error = function(e) NULL)
  
  if (!is.null(res_data)) {
    return(data.frame(
      job_id = res_data$job_id,
      sep = res_data$sep,
      fdr_level = res_data$fdr_level,
      accuracy = as.numeric(res_data$accuracy),
      L = res_data$L,
      tp = res_data$tp,
      fp = res_data$fp,
      power = res_data$power,
      empirical_fdr = res_data$empirical_fdr,
      runtime = as.numeric(res_data$runtime),
      source = "RDS"
    ))
  }
  
  # Failover to parsing the raw SLURM .out text logs natively
  if (file.exists(log_file)) {
    log_lines <- tryCatch(readLines(log_file, warn = FALSE), error = function(e) character(0))
    
    acc_val <- NA
    rt_val <- NA
    
    # Scrape backward for performance values
    for (line in rev(log_lines)) {
      if (is.na(acc_val) && grepl("Balanced Accuracy \\(Acc\\):", line)) {
        acc_val <- as.numeric(sub(".*Balanced Accuracy \\(Acc\\):\\s*([0-9.]+).*", "\\1", line))
      }
      if (is.na(rt_val) && grepl("Runtime:", line)) {
        rt_val <- as.numeric(sub(".*Runtime:\\s*([0-9.]+)\\s*seconds.*", "\\1", line))
      }
      if (!is.na(acc_val) && !is.na(rt_val)) break
    }
    
    if (!is.na(acc_val) || !is.na(rt_val)) {
      return(data.frame(
        job_id = j_id,
        sep = sp,
        fdr_level = fdr_level,
        accuracy = acc_val,
        L = NA, tp = NA, fp = NA, power = NA, empirical_fdr = NA,
        runtime = rt_val,
        source = "LOG"
      ))
    }
  }
  
  # If both missing/failed, return NA row tracking the dropout natively
  return(data.frame(
    job_id = j_id, sep = sp, fdr_level = fdr_level,
    accuracy = NA, L = NA, tp = NA, fp = NA, power = NA, empirical_fdr = NA,
    runtime = NA, source = "FAILED"
  ))
})

all_results <- bind_rows(results_list)

# Validate summaries directly
success_rds <- sum(all_results$source == "RDS")
success_log <- sum(all_results$source == "LOG")
failed_runs <- sum(all_results$source == "FAILED")

cat(sprintf("\nIntegration Summary:\n- Valid RDS Files: %d\n- Log File Scrapes: %d\n- Total Failures: %d\n", 
            success_rds, success_log, failed_runs))

if (failed_runs > 0) {
  warning(sprintf("Warning: %d runs failed to produce valid RDS or Log scraping outputs.", failed_runs))
}

dir.create("results_aggregated", showWarnings = FALSE)
save_path <- file.path("results_aggregated", output_combined_file)
saveRDS(all_results, save_path)
cat(sprintf("Aggregated dataset successfully saved to '%s'. Proceed to aggregation script.\n", save_path))
