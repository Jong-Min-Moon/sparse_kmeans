# Verification script for Deterministic Iterative Clustering (deterministic vs true)
library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)
library(kernlab)

# Source dependencies
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")
source("../../code_r/ISEE_bicluster.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/block_coordinate_optim_deterministic_unknowncov.R")

# Set up parallel
if (!exists("cl")) {
    cl <- makeCluster(max(1, parallel::detectCores() - 1))
    registerDoParallel(cl)
}

# 1. Generate Data (K=2)
set.seed(42)
p <- 100
n <- 200
K <- 2
S_0 <- 10 # 10 signal features
signal_size <- 1.5

cat(sprintf("Generating test data: p=%d, n=%d, K=%d, S_0=%d\n", p, n, K, S_0))

# True labels
true_labels <- c(rep(1, n / 2), rep(2, n / 2))

# Data matrix
X <- matrix(rnorm(p * n), p, n)
X[1:S_0, true_labels == 1] <- X[1:S_0, true_labels == 1] + signal_size
X[1:S_0, true_labels == 2] <- X[1:S_0, true_labels == 2] - signal_size

# 2. Run Deterministic Algorithm
cat("\n--- Running Deterministic Optimization (Universal Threshold) ---\n")
res_det <- block_coordinate_optim_deterministic_unknowncov(
    X = X,
    K = K,
    n_iter = 20,
    stable_iter = 5,
    max_iter_sdp = 1000,
    true_labels = true_labels
)

cat("\nVerification Finished.\n")
cat(sprintf("Final ARI: %.4f\n", mclust::adjustedRandIndex(res_det$cluster, true_labels)))
cat(sprintf("Features Selected: %d\n", sum(res_det$selected_features)))
cat(sprintf("Signal Features captured: %d / %d\n", sum(res_det$selected_features[1:S_0]), S_0))

stopCluster(cl)
