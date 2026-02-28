# ---------------------------------------------------------
# Single Simulation Run: sim14_thompson
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
library(RSpectra)
library(CVXR)
library(cluster)

# Arguments for HPC
args <- commandArgs(trailingOnly = TRUE)

job_id <- 1
C_val_target <- 0.5
n_iter_tvs <- 1000
n_perms <- 300

if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
        if (args[i] == "--job_id") job_id <- as.integer(args[i + 1])
        if (args[i] == "--C_val") C_val_target <- as.numeric(args[i + 1])
        if (args[i] == "--n_iter") n_iter_tvs <- as.integer(args[i + 1])
        if (args[i] == "--perms") n_perms <- as.numeric(args[i + 1])
    }
}

# Source Code
source("../../code_r/sparse_symmetric_data_generator.R")
source("../../code_r/block_coordinate_optim_thompson.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")
source("../../code_r/cluster_spectral.R")
source("../../code_r/ESSC.R")
source("../../code_r/get_cluster_acc.R")

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
# MATCHING SIM13 SETTINGS:
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

cat(sprintf("--- Simulation Run sim14_thompson (Job ID: %d) ---\n", job_id))
cat(sprintf("Params: p=%d, n=%d, sep=%.1f, rho=%.2f\n", p, n, separation, rho))
cat(sprintf("Method: Thompson Sampling (C_val=%.2f, Perms=%d)\n", C_val_target, n_perms))

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
# Run Algorithm (Thompson Sampling)
# ---------------------------------------------------------
cat("Running block_coordinate_optim_thompson...\n")

# Execution flow from sim01 (Silhouette Grid Search)
C_values <- c(C_val_target) # We can use the C_val_target passed from deploy script to satisfy 'same parameter grid'
best_obj <- -Inf
best_res <- NULL
results_all <- list()

cat("Pre-calculating distance matrix for Silhouette index...\n")
dist_x <- dist(t(X))

start_time <- Sys.time()

for (C_val in C_values) {
    cat(sprintf("  Running for C=%.2f...\n", C_val))

    # Ignore covariance structure in optimization (covariance = NULL)
    res <- block_coordinate_optim_thompson(X, K, n_iter = n_iter_tvs, C = C_val, n_perms = n_perms, covariance = NULL, true_cluster = true_labels)

    if (length(unique(res$cluster)) < 2) {
        cat(sprintf("    Warning: Only 1 cluster found for C=%.2f. Setting silhouette to -1.\n", C_val))
        avg_sil <- -1
    } else {
        sil_res <- silhouette(res$cluster, dist_x)
        avg_sil <- mean(sil_res[, "sil_width"])
    }

    cat(sprintf("  C=%.2f => Avg Silhouette Width: %.4f\n", C_val, avg_sil))

    results_all[[as.character(C_val)]] <- list(res = res, sil = avg_sil)

    if (avg_sil > best_obj) {
        best_obj <- avg_sil
        best_res <- res
    }
}

res <- best_res
end_time <- Sys.time()

# ---------------------------------------------------------
# Evaluation (Matching sim13 metrics exactly)
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
# Thompson returns a logical vector in res$selected
selected_indices <- which(res$selected)
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

overlap_at_100 <- NA
cat(sprintf("Overlap at iteration 100: NA (Not applicable for Thompson)\n"))

# Save Result matching sim13 format completely
dir.create("results", showWarnings = FALSE)
saveRDS(list(
    res = res,
    ari = ari,
    acc = acc,
    tp = tp,
    fp = fp,
    overlap_at_100 = overlap_at_100,
    job_id = job_id,
    params = list(p = p, n = n, rho = rho, sep = separation, C_val = C_val_target, n_perms = n_perms)
), file = sprintf("results/sim_id%d.rds", job_id))
