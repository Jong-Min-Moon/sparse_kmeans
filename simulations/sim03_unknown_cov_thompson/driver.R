# ---------------------------------------------------------
# Single Simulation Run: Unknown Covariance (Thompson Sampling)
# ---------------------------------------------------------
library(methods)
library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)

# Arguments for HPC
args <- commandArgs(trailingOnly = TRUE)
job_id <- 1
if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
        if (args[i] == "--job_id") job_id <- as.integer(args[i + 1])
    }
}

# Source Code
source("../../code_r/block_coordinate_optim_thompson_unknowncov.R")
source("../../code_r/ISEE_bicluster.R")
source("../../code_r/get_intercept_residual_lasso.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")

set.seed(2025 + job_id)

# ---------------------------------------------------------
# Data Generation Parameters (Copied from sim02)
# ---------------------------------------------------------
p <- 400
n <- 500
K <- 2
rho <- 0.45

# 1. Precision Matrix Omega (Tridiagonal)
cat("Generating Precision Matrix Omega...\n")
Omega <- matrix(0, p, p)
for (i in 1:p) {
    Omega[i, i] <- 1
    if (i > 1) Omega[i, i - 1] <- rho
    if (i < p) Omega[i, i + 1] <- rho
}
Sigma <- solve(Omega)

# 2. Signal Generation
S_0 <- 1:10
cat("Generating Signal...\n")
v <- rep(0, p)
v[S_0] <- 1
delta <- sqrt(9 / sum(v^2))
mu_diff <- as.numeric(Sigma %*% (v * delta))
mu1 <- mu_diff / 2
mu2 <- -mu_diff / 2

# 3. Generate Data
cat("Generating Data X...\n")
n_c <- n / 2
X1 <- mvrnorm(n_c, mu1, Sigma)
X2 <- mvrnorm(n_c, mu2, Sigma)
X <- t(rbind(X1, X2)) # p x n
true_labels <- c(rep(1, n_c), rep(2, n_c))

# ---------------------------------------------------------
# Run Algorithm
# ---------------------------------------------------------
cat("Running block_coordinate_optim_thompson_unknowncov...\n")

# Register Parallel
if (getDoParWorkers() == 1) {
    if (requireNamespace("doParallel", quietly = TRUE)) {
        doParallel::registerDoParallel(cores = 2)
    } else {
        registerDoSEQ()
    }
}

start_time <- Sys.time()
res <- block_coordinate_optim_thompson_unknowncov(
    X = X,
    K = 2,
    n_iter = 200,
    n_perms = 1000,
    fdr_level = 0.1,
    max_iter_sdp = 4000
)
end_time <- Sys.time()

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
cat("\n--- Results ---\n")
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Runtime: %.2f seconds\n", runtime))

# Clustering Accuracy (ARI)
ari <- adjustedRandIndex(res$cluster, true_labels)
cat(sprintf("Adjusted Rand Index (ARI): %.4f\n", ari))

# Variable Selection
selected_indices <- which(res$selected_features)
cat(sprintf("Number of selected features: %d\n", length(selected_indices)))

tp <- length(intersect(selected_indices, S_0))
fp <- length(setdiff(selected_indices, S_0))
cat(sprintf("True Positives (TP): %d / %d\n", tp, length(S_0)))
cat(sprintf("False Positives (FP): %d\n", fp))

# Save Result
dir.create("results", showWarnings = FALSE)
saveRDS(list(res = res, ari = ari, tp = tp, fp = fp, job_id = job_id),
    file = sprintf("results/sim_id%d.rds", job_id)
)
cat(sprintf("Result saved to results/sim_id%d.rds\n", job_id))
