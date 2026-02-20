# ---------------------------------------------------------
# Single Simulation Run: Thompson Sampling v3 (Oracle ISEE)
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

job_id <- 1
n_perms <- 200

if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
        if (args[i] == "--job_id") job_id <- as.integer(args[i + 1])
        if (args[i] == "--perms") n_perms <- as.integer(args[i + 1])
    }
}

# Source Code (Adjusted paths for simulations/sim_v3_oracle_isee/)
source("../../code_r/sparse_symmetric_data_generator.R")
source("../../code_r/block_coordinate_optim_thompson_unknowncov_v3.R")
source("../../code_r/selection_block_greedy_screening.R")
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
# Load C++ Backend
# ---------------------------------------------------------
lib_name <- "selection_utils"
ext <- if (.Platform$OS.type == "windows") ".dll" else ".so"
lib_path <- paste0("../../code_r/", lib_name, ext)
if (file.exists(lib_path)) {
    cat(sprintf("Loading C++ Backend: %s\n", lib_path))
    # Unload if already loaded to prevent caching issues across jobs
    if (lib_name %in% names(getLoadedDLLs())) dyn.unload(lib_path)
    dyn.load(lib_path)
} else {
    warning("C++ backend not found at ", lib_path)
}

# ---------------------------------------------------------
# Data Generation Parameters (Matching sim07)
# ---------------------------------------------------------
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
cat(sprintf("Method: Thompson Sampling v3 (Oracle ISEE on Iter 1)\n"))

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
# Run Algorithm (Thompson Sampling v3)
# ---------------------------------------------------------
cat("Running block_coordinate_optim_thompson_unknowncov_v3...\n")

start_time <- Sys.time()
res <- block_coordinate_optim_thompson_unknowncov_v3(
    X = X,
    K = 2,
    n_iter = 100,
    C = 0.5,
    n_perms = n_perms,
    max_iter_sdp = 4000,
    true_labels = true_labels  # Required for v3 First Iteration ISEE
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
# For v3, res$selected is a logical vector of length p. Convert to indices.
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

# Save Result
dir.create("results", showWarnings = FALSE)
saveRDS(list(
    res = res,
    ari = ari,
    acc = acc,
    tp = tp,
    fp = fp,
    job_id = job_id,
    objective_trajectory = res$objective,
    params = list(p = p, n = n, rho = rho, sep = separation, fdr = fdr_target, n_perms = n_perms)
), file = sprintf("results/sim_id%d.rds", job_id))
