# ---------------------------------------------------------
# Single Simulation Run: Unknown Covariance (Deterministic)
# ---------------------------------------------------------
library(methods)
library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)
library(kernlab)

# Arguments for HPC
args <- commandArgs(trailingOnly = TRUE)
job_id <- 1
if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
        if (args[i] == "--job_id") job_id <- as.integer(args[i + 1])
    }
}

# Source Code (Adjusted paths for simulations/sim04.../)
source("../../code_r/block_coordinate_optim_deterministic_unknowncov.R")
source("../../code_r/ISEE_bicluster.R")
source("../../code_r/get_intercept_residual_lasso.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")

set.seed(2025 + job_id) # Unique seed per job

# ---------------------------------------------------------
# Data Generation Parameters (Matches Sim 02)
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

# Covariance Sigma
Sigma <- solve(Omega)

# 2. Signal Generation
S_0 <- 1:10
cat("Generating Signal...\n")

# Target: || Omega * (mu1 - mu2) ||^2 = 9 (3^2)
v <- rep(0, p)
v[S_0] <- 1
norm_sq_v <- sum(v^2)
delta <- sqrt(9 / norm_sq_v)

# Set s = v * delta, then mu_diff = Sigma %*% s
mu_diff <- as.numeric(Sigma %*% (v * delta))

# Centered around 0
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
# Run Algorithm (Deterministic)
# ---------------------------------------------------------
cat("Running block_coordinate_optim_deterministic_unknowncov...\n")

# Register Parallel
num_cores <- parallel::detectCores() - 1
if (num_cores < 1) num_cores <- 1
doParallel::registerDoParallel(cores = min(num_cores, 10))

start_time <- Sys.time()
res <- block_coordinate_optim_deterministic_unknowncov(
    X = X,
    K = 2,
    n_iter = 100,
    stable_iter = 10,
    max_iter_sdp = 4000,
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
ari <- adjustedRandIndex(res$cluster, true_labels)
cat(sprintf("Adjusted Rand Index (ARI): %.4f\n", ari))

# Variable Selection
selected_indices <- which(res$selected_features)
cat(sprintf("Number of selected features: %d\n", length(selected_indices)))

# TP/FP
tp <- length(intersect(selected_indices, S_0))
fp <- length(setdiff(selected_indices, S_0))

cat(sprintf("True Positives (TP): %d / %d\n", tp, length(S_0)))
cat(sprintf("False Positives (FP): %d\n", fp))
cat(sprintf("Recall: %.2f\n", tp / length(S_0)))

# Save Result
dir.create("results", showWarnings = FALSE)
saveRDS(list(res = res, ari = ari, tp = tp, fp = fp, job_id = job_id), file = sprintf("results/sim_id%d.rds", job_id))
