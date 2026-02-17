library(testthat)
library(Matrix)
library(glmnet)
library(foreach)

# Source necessary files
# Need ISEE_bicluster, sdp_kmeans, clustering_block_unknowncov, get_intercept_residual_lasso
source("code_r/get_intercept_residual_lasso.R")
source("code_r/ISEE_bicluster.R")
source("code_r/sdp_kmeans.R")
source("code_r/clustering_block_knowncov.R")
source("code_r/clustering_block_unknowncov.R")

test_that("clustering_block_unknowncov runs and returns valid output", {
    # Setup minimal valid inputs
    p <- 10
    n <- 20
    K <- 2

    # Mock X
    x <- matrix(rnorm(p * n), nrow = p, ncol = n)

    # Mock Cluster Est
    cluster_est_prev <- rep(1:K, each = n / K)

    # Mock Selected Features (Select all for simplicity)
    selected_features <- rep(TRUE, p)

    # Run
    # Register parallel for ISEE
    if (getDoParWorkers() == 1) registerDoSEQ()

    # Expect successful execution (ISEE prints output, so expect_silent fails)
    res <- run_clustering_block_unknowncov(x, selected_features, K, cluster_est_prev)

    # Check output format
    expect_type(res, "list")
    expect_true("cluster" %in% names(res))
    expect_length(res$cluster, n)
    expect_true(all(res$cluster %in% 1:K))

    # Check failure cases (no features)
    selected_none <- rep(FALSE, p)
    expect_warning(
        {
            res_none <- run_clustering_block_unknowncov(x, selected_none, K, cluster_est_prev)
        },
        "No features selected"
    )

    expect_equal(res_none$cluster, cluster_est_prev)
    expect_true(is.na(res_none$value))
})
