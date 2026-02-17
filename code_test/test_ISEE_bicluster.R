library(testthat)
library(foreach)
# library(doParallel) # Optional: User might register parallel backend externally

# Source functions
if (file.exists("../code_r/ISEE_bicluster.R")) {
    source("../code_r/ISEE_bicluster.R")
    source("../code_r/get_intercept_residual_lasso.R")
} else if (file.exists("code_r/ISEE_bicluster.R")) {
    source("code_r/ISEE_bicluster.R")
    source("code_r/get_intercept_residual_lasso.R")
} else {
    stop("Could not find source files")
}

test_that("ISEE_bicluster returns correct structure", {
    set.seed(123)
    n <- 20
    p <- 10 # Even p
    K <- 2

    # Generate synthetic data
    # Cluster mean vectors
    mu1 <- rep(0, p)
    mu1[1:5] <- 2
    mu2 <- rep(0, p)
    mu2[6:10] <- -2

    mean_mat_true <- matrix(0, nrow = p, ncol = n)
    cluster_est <- rep(1:K, each = n / K)

    mean_mat_true[, cluster_est == 1] <- mu1
    mean_mat_true[, cluster_est == 2] <- mu2

    noise <- matrix(rnorm(n * p), nrow = p, ncol = n)
    x <- mean_mat_true + noise

    # Register sequential backend for testing if none active
    if (getDoParWorkers() == 1) {
        registerDoSEQ()
    }

    res <- ISEE_bicluster(x, cluster_est)

    expect_type(res, "list")
    expect_named(res, c("X_tilde", "Omega_diag_hat"))

    expect_equal(dim(res$X_tilde), c(p, n))
    expect_length(res$Omega_diag_hat, p)

    # Check if X_tilde roughly resembles true mean + some modification (denoised)
    # It's hard to check exact values without knowing the true logic, but dimensions are key.
    # X_tilde = Omega * (Mean + E) = Omega * Omega^-1 * X? No.
    # X_tilde = Omega * (alpha + E).
    # If model is perfect, alpha + E is consistent.

    expect_true(is.numeric(res$X_tilde))
})

test_that("ISEE_bicluster handles odd p", {
    set.seed(123)
    n <- 20
    p <- 11 # Odd p
    K <- 2

    x <- matrix(rnorm(n * p), nrow = p, ncol = n)
    cluster_est <- rep(1:K, each = n / K)

    res <- ISEE_bicluster(x, cluster_est)

    expect_equal(dim(res$X_tilde), c(p, n))
    expect_length(res$Omega_diag_hat, p)
})

print("Test script completed successfully.")
