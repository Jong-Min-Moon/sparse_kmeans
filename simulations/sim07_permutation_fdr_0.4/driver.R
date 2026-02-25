# ---------------------------------------------------------
# Single Simulation Run: Permutation FDR (Unknown Covariance, Target 0.4)
# ---------------------------------------------------------
library(methods)
library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)
library(kernlab)
library(matrixStats)

# Arguments for HPC
args <- commandArgs(trailingOnly = TRUE)


if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
        if (args[i] == "--job_id") job_id <- as.integer(args[i + 1])
        if (args[i] == "--fdr") fdr_target <- as.numeric(args[i + 1])
        if (args[i] == "--perms") n_perms <- as.integer(args[i + 1])
    }
}

# Source Code (Adjusted paths for simulations/sim07.../)
# Assuming sim07 is at the same depth as sim06
source("../../code_r/sparse_symmetric_data_generator.R")
source("../../code_r/block_coordinate_optim_greedy_unknowncov_SAM.R")
source("../../code_r/ESSC.R")
source("../../code_r/ISEE_residual_lasso.R")
source("../../code_r/get_intercept_residual_lasso.R")
source("../../code_r/get_cov_small.R")
source("../../code_r/ISEE_bicluster.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/get_cluster_acc.R")
source("../../code_r/utils.R")

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
# MATCHING SIM04 SETTINGS:
# p=400, n=500, K=2, rho=0.45, separation=3, support=1:10
p <- 400
n <- 500
K <- 2
rho_param <- 45
rho <- rho_param / 100
separation <- 3
precision_sparsity <- 2
support <- 1:10
flip <- FALSE

cat(sprintf("--- Simulation Run (Job ID: %d) ---\n", job_id))
cat(sprintf("Params: p=%d, n=%d, sep=%.1f, rho=%.2f\n", p, n, separation, rho))
cat(sprintf("Method: Permutation FDR (Target=%.2f, Perms=%d)\n", fdr_target, n_perms))

# 1. Initialize Generator
generator <- sparse_symmetric_data_generator(
    support = support,
    separation = separation,
    dimension = p,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho,
    flip = flip
)

# 2. Generate Data
set.seed(2025 + job_id)
data_res <- generate_data_from_generator(generator, n, seed = 2025 + job_id)
X <- data_res$X
true_labels <- data_res$labels

# ---------------------------------------------------------
# Run Algorithm (Permutation Based)
# ---------------------------------------------------------
cat("Running block_coordinate_optim_greedy_unknowncov_SAM...\n")

# Register Parallel (Standard)
num_cores <- parallel::detectCores() - 1
if (num_cores < 1) num_cores <- 1
if (Sys.getenv("SLURM_CPUS_PER_TASK") != "") {
    num_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK"))
}
doParallel::registerDoParallel(cores = min(num_cores, 10))

start_time <- Sys.time()
res <- block_coordinate_optim_greedy_unknowncov_SAM(
    X = X,
    K = 2,
    n_iter = 100,
    n_perms = n_perms,
    fdr_target = fdr_target,
    stable_iter = 5,
    true_labels = true_labels
)
end_time <- Sys.time()

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
cat("\n--- Results ---\n")
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Runtime: %.2f seconds\n", runtime))

# Clustering Accuracy (ARI)
ari <- mclust::adjustedRandIndex(res$cluster, true_labels)
acc <- get_cluster_acc(res$cluster, true_labels)
cat(sprintf("Adjusted Rand Index (ARI): %.4f\n", ari))
cat(sprintf("Balanced Accuracy (Acc): %.4f\n", acc))

# Variable Selection
selected_indices <- res$s_hat
cat(sprintf("Number of selected features: %d\n", length(selected_indices)))

# TP/FP
tp <- length(intersect(selected_indices, support))
fp <- length(setdiff(selected_indices, support))

cat(sprintf("True Positives (TP): %d / %d\n", tp, length(support)))
cat(sprintf("False Positives (FP): %d\n", fp))
if (length(support) > 0) {
    cat(sprintf("Recall: %.2f\n", tp / length(support)))
}
if (length(selected_indices) > 0) {
    cat(sprintf("Precision: %.2f\n", tp / length(selected_indices)))
}

# Save Result
dir.create("results", showWarnings = FALSE)
saveRDS(list(
    res = res,
    ari = ari,
    acc = acc,
    tp = tp,
    fp = fp,
    job_id = job_id,
    params = list(p = p, n = n, rho = rho, sep = separation, fdr = fdr_target, n_perms = n_perms)
), file = sprintf("results/sim_id%d.rds", job_id))
