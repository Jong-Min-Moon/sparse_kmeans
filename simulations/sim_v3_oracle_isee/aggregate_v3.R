# Aggregate Simulation v3 Results and Plot Trajectories

library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)

base_dir <- "."
output_dir <- file.path(base_dir, "output")
v2_output_dir <- file.path(base_dir, "../sim_v2_baseline/output") # Example placeholder for v2 results
summary_file <- "summary_sim_v3.csv"
plot_file <- "trajectory_comparison.png"

if (!dir.exists(output_dir)) {
    stop("Output directory not found: ", output_dir, "\nRun retrieve_v3.sh first.")
}

# --- 1. Load v3 Results ---
all_files <- list.files(output_dir, pattern = "\\.rds$", full.names = TRUE)

if (length(all_files) == 0) {
    stop("No .rds files found in ", output_dir)
}

cat(sprintf("Found %d v3 result files. Aggregating...\n", length(all_files)))

results_list <- purrr::map(all_files, function(f) {
    res_data <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(res_data)) return(NULL)

    job_id <- res_data$job_id
    ari <- res_data$ari
    acc <- res_data$acc
    tp <- res_data$tp
    fp <- res_data$fp
    n_selected <- tp + fp
    
    # Sparsity Recovery
    recall <- if (tp > 0) tp / 10 else 0
    precision <- if (n_selected > 0) tp / n_selected else 0
    obs_fdr <- 1 - precision

    # Trajectory
    obj_traj <- res_data$objective_trajectory
    if (is.null(obj_traj)) obj_traj <- rep(NA, 100)

    list(
        metrics = data.frame(
            job_id = job_id,
            accuracy = acc,
            ari = ari,
            tp = tp,
            fp = fp,
            n_selected = n_selected,
            recall = recall,
            precision = precision,
            obs_fdr = obs_fdr
        ),
        trajectory = obj_traj
    )
})

# Separate metrics and trajectories
metrics_list <- lapply(results_list, function(x) x$metrics)
all_results <- bind_rows(metrics_list)

traj_list <- lapply(results_list, function(x) x$trajectory)
traj_mat <- do.call(rbind, traj_list)
mean_traj_v3 <- colMeans(traj_mat, na.rm = TRUE)

# --- 2. Calculate Summary Metrics (Means and SDs) ---
summary_stats <- data.frame(
    n_reps = nrow(all_results),
    mean_accuracy = mean(all_results$accuracy, na.rm = TRUE),
    sd_accuracy = sd(all_results$accuracy, na.rm = TRUE),
    mean_ari = mean(all_results$ari, na.rm = TRUE),
    sd_ari = sd(all_results$ari, na.rm = TRUE),
    mean_tp = mean(all_results$tp, na.rm = TRUE),
    sd_tp = sd(all_results$tp, na.rm = TRUE),
    mean_fp = mean(all_results$fp, na.rm = TRUE),
    sd_fp = sd(all_results$fp, na.rm = TRUE),
    mean_recall = mean(all_results$recall, na.rm = TRUE),
    sd_recall = sd(all_results$recall, na.rm = TRUE),
    mean_obs_fdr = mean(all_results$obs_fdr, na.rm = TRUE),
    sd_obs_fdr = sd(all_results$obs_fdr, na.rm = TRUE),
    perfect_acc_count = sum(all_results$accuracy >= 0.999, na.rm = TRUE)
)

write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("\nSummary saved to %s\n", summary_file))

cat("\n=== Simulation v3 Oracle ISEE Summary ===\n")
print(summary_stats)
saveRDS(all_results, "all_results_sim_v3.rds")

# --- 3. Gather v2 Results (if available) for Trajectory Plot ---
mean_traj_v2 <- rep(NA, length(mean_traj_v3))
if (dir.exists(v2_output_dir)) {
    v2_files <- list.files(v2_output_dir, pattern = "\\.rds$", full.names = TRUE)
    if (length(v2_files) > 0) {
        cat("Found v2 results. Loading for trajectory comparison...\n")
        v2_traj_list <- lapply(v2_files, function(f) {
            res_data <- tryCatch(readRDS(f), error = function(e) NULL)
            if (is.null(res_data) || is.null(res_data$objective_trajectory)) return(rep(NA, 100))
            return(res_data$objective_trajectory)
        })
        v2_traj_mat <- do.call(rbind, v2_traj_list)
        mean_traj_v2 <- colMeans(v2_traj_mat, na.rm = TRUE)
    }
} else {
    cat("Warning: v2 baseline results not found at", v2_output_dir, "\nPlotting v3 trajectory only...\n")
}

# --- 4. Plot Trajectories ---
traj_len <- length(mean_traj_v3)
plot_data <- data.frame(
    Iteration = rep(1:traj_len, 2),
    Objective = c(mean_traj_v3, mean_traj_v2),
    Model = rep(c("v3 (Oracle ISEE)", "v2 (Standard)"), each = traj_len)
)

# Filter out NAs if v2 wasn't loaded
plot_data <- plot_data[!is.na(plot_data$Objective), ]

p <- ggplot(plot_data, aes(x = Iteration, y = Objective, color = Model, linetype = Model)) +
    geom_line(size = 1.2) +
    theme_minimal() +
    labs(
        title = "Convergence Trajectory: Oracle (v3) vs Standard (v2)",
        x = "Iteration",
        y = "Mean Objective Value"
    ) +
    theme(
        legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold")
    )

ggsave(plot_file, plot = p, width = 8, height = 6, dpi = 300)
cat(sprintf("\nTrajectory comparison plot saved to %s\n", plot_file))
