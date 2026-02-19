# ---------------------------------------------------------
# Verification Script: Thompson Sampling (Small-Medium)
# ---------------------------------------------------------
library(methods)
library(MASS)
library(mclust)
library(Matrix)
library(foreach)
library(doParallel)
library(glmnet)

# Source Code
source("../../code_r/block_coordinate_optim_thompson_unknowncov.R")
source("../../code_r/ISEE_bicluster.R")
source("../../code_r/get_intercept_residual_lasso.R")
source("../../code_r/clustering_block_knowncov.R")
source("../../code_r/selection_block_greedy_screening.R")
source("../../code_r/sdp_kmeans.R")
source("../../code_r/utils.R")

set.seed(123)

# Small-Medium Dataset Parameters
p <- 100
n <- 200
K <- 2
rho <- 0.2

# Generate Covariance
cat("Generating medium-small data (p=100, n=200)...\n")
Omega <- matrix(0, p, p)
for (i in 1:p) {
    Omega[i, i] <- 1
    if (i > 1) Omega[i, i - 1] <- rho
    if (i < p) Omega[i, i + 1] <- rho
}
Sigma <- solve(Omega)

S_0 <- 1:10
v <- rep(0, p)
v[S_0] <- 1
delta <- sqrt(9 / sum(v^2))
mu_diff <- as.numeric(Sigma %*% (v * delta))
mu1 <- mu_diff / 2
mu2 <- -mu_diff / 2
X1 <- mvrnorm(n / 2, mu1, Sigma)
X2 <- mvrnorm(n / 2, mu2, Sigma)
X <- t(rbind(X1, X2))
true_labels <- c(rep(1, n / 2), rep(2, n / 2))

# Run Thompson Sampling
cat("Running verification of block_coordinate_optim_thompson_unknowncov (FDR=0.1)...\n")
if (requireNamespace("doParallel", quietly = TRUE)) {
    doParallel::registerDoParallel(cores = 2)
}

tryCatch(
    {
        res <- block_coordinate_optim_thompson_unknowncov(
            X = X,
            K = 2,
            n_iter = 50,
            C=0.7,
            n_perms = 500,
            fdr_level = 0.05,
            max_iter_sdp = 1000,
            true_labels = true_labels
        )
        cat("\nVerification SUCCESSFUL. Algorithm ran without errors.\n")
        cat(sprintf("Adjusted Rand Index: %.4f\n", adjustedRandIndex(res$cluster, true_labels)))
    },
    error = function(e) {
        cat("\nVerification FAILED with error:\n")
        print(e)
    }
)
