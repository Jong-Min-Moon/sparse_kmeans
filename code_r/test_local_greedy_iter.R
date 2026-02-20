# Test Local Greedy Iteration
library(stats)
library(Rcpp)

# Source dependencies
source("code_r/sdp_kmeans.R")
source("code_r/utils.R")
source("code_r/selection_block_greedy_screening.R")
source("code_r/clustering_block_knowncov.R")
source("code_r/block_coordinate_optim_greedy.R")

# 1. Setup Data
set.seed(42)
p <- 1000
n <- 200
K <- 2
s <- 10
X <- matrix(rnorm(n * p), p, n)
true_labels <- c(rep(1, n / 2), rep(2, n / 2))

# Separation d = 4 (squared d^2 = 16)
# If we have s features, each feature contributes 16/s to the squared distance
# (mu1_j - mu2_j)^2 = 16/s => mu1_j - mu2_j = sqrt(16/s)
# Set mu1_j = sqrt(16/s)/2 and mu2_j = -sqrt(16/s)/2
delta <- sqrt(16 / s) / 2
X[1:s, true_labels == 1] <- X[1:s, true_labels == 1] + delta
X[1:s, true_labels == 2] <- X[1:s, true_labels == 2] - delta

# 2. Run One Iteration
cat("\n--- Running 1 Iteration of Greedy Optimizer locally ---\n")
# Using n_iter=1 to stop after first loop
# stable_iter is large so it doesn't stop prematurely by stability logic
res <- block_coordinate_optim_greedy(X, K, n_iter = 1, stable_iter = 100, fdr_level = 0.4)

cat("\n--- Finished 1 Iteration ---\n")
cat(sprintf("Iterations run: %d\n", res$iter))
# Compute Accuracy (not ARI) as max matching percentage
cat(sprintf("Final Clustering Accuracy: %.4f\n", max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))))
cat(sprintf("Final Adjusted Rand Index: %.4f\n", mclust::adjustedRandIndex(res$cluster, true_labels)))

# Check if selected features in that iteration were reasonable
selected <- selection_block_greedy_screening(X, res$cluster, fdr_level = 0.4, n_perms = 1000)
cat(sprintf("Features selected after 1 iteration: %d\n", sum(selected)))
cat(sprintf("True features in selection: %d/%d\n", sum(selected[1:s]), s))
