library(testthat)
library(Matrix)
library(glmnet)
library(foreach)
library(MASS) # For RandIndex if needed (or mclust)

# Check and load RandIndex dependency
if (!requireNamespace("mclust", quietly = TRUE)) {
    # Mock RandIndex if mclust is not available
    RandIndex <- function(c1, c2) {
        return(list(AR = 1.0)) # Mock always 1 for testing flow if package missing
    }
} else {
    library(mclust)
    RandIndex <- function(c1, c2) {
        ar <- adjustedRandIndex(c1, c2)
        return(list(AR = ar))
    }
}

# Source dependencies
source("code_r/get_intercept_residual_lasso.R")
source("code_r/ISEE_bicluster.R")
source("code_r/sdp_kmeans.R")
source("code_r/clustering_block_knowncov.R")
source("code_r/selection_block_greedy_screening.R")
source("code_r/block_coordinate_optim_greedy_unknowncov.R")

test_that("block_coordinate_optim_greedy_unknowncov runs without error on small data", {
    set.seed(123)
    n <- 20
    p <- 50
    K <- 2

    # Generate small synthetic data
    # 2 clusters, signal in first 5 variables
    mu1 <- rep(0, p)
    mu1[1:5] <- 2
    mu2 <- rep(0, p)
    mu2[1:5] <- -2

    X <- matrix(rnorm(n * p), nrow = p, ncol = n)
    true_labels <- rep(1:K, each = n / K)
    X[, true_labels == 1] <- X[, true_labels == 1] + mu1
    X[, true_labels == 2] <- X[, true_labels == 2] + mu2

    # Register parallel for ISEE
    if (getDoParWorkers() == 1) {
        if (requireNamespace("doParallel", quietly = TRUE)) {
            doParallel::registerDoParallel(cores = 2)
        } else {
            registerDoSEQ()
        }
    }

    # Run function
    # Capture output to avoid cluttering test log
    # We expect it might fail due to identified bugs, but we run it to confirm.

    expect_error(
        {
            res <- block_coordinate_optim_greedy_unknowncov(X, K, n_iter = 2, stable_iter = 2)
        },
        NA
    ) # NA implies we expect NO error. If it errors, test fails (observation).
})
