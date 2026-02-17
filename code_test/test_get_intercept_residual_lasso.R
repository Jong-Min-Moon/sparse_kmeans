library(testthat)
library(glmnet)

# Source the function to be tested
# Adjust path as necessary assuming we run from project root or code_test folder
if (file.exists("../code_r/get_intercept_residual_lasso.R")) {
    source("../code_r/get_intercept_residual_lasso.R")
} else if (file.exists("code_r/get_intercept_residual_lasso.R")) {
    source("code_r/get_intercept_residual_lasso.R")
} else {
    stop("Could not find get_intercept_residual_lasso.R")
}

test_that("get_intercept_residual_lasso returns correct structure on valid input", {
    set.seed(123)
    n <- 50
    p <- 10
    predictor <- matrix(rnorm(n * p), nrow = n, ncol = p)
    # True model: y = 2 + 3*x1 + epsilon
    response <- 2 + 3 * predictor[, 1] + rnorm(n)

    result <- get_intercept_residual_lasso(response, predictor)

    expect_type(result, "list")
    expect_named(result, c("intercept", "residual"))
    expect_type(result$intercept, "double")
    expect_length(result$intercept, 1)
    expect_type(result$residual, "double")
    expect_length(result$residual, n)

    # Basic check that residuals are somewhat small (model should fit reasonably well)
    expect_true(mean(result$residual^2) < var(response))
})

test_that("get_intercept_residual_lasso throws error on dimension mismatch", {
    n <- 20
    p <- 5
    predictor <- matrix(rnorm(n * p), nrow = n, ncol = p)
    response <- rnorm(n + 1) # Mismatch

    expect_error(get_intercept_residual_lasso(response, predictor), "Dimensions of response and predictor do not match")
})

test_that("get_intercept_residual_lasso handles constant predictor (fallback)", {
    n <- 30
    p <- 5
    # Constant predictor matrix (zero variance)
    predictor <- matrix(0, nrow = n, ncol = p)
    response <- rnorm(n, mean = 5)

    # The function should catch the glmnet error and return null model
    # Or if glmnet handles it (it might standardize to NA), our code specifically has a tryCatch

    # Warning expectation depends on if glmnet throws error or warning.
    # The code has a warning inside the error handler: "glmnet failed: ... - falling back to simple mean."
    # Let's just check the result structure.

    expect_warning(result <- get_intercept_residual_lasso(response, predictor), "glmnet failed")

    expect_type(result, "list")
    expect_equal(result$intercept, mean(response))
    # Residuals should be centered response
    expect_equal(result$residual, response - mean(response))
})

test_that("get_intercept_residual_lasso works with single feature (fallback)", {
    set.seed(456)
    n <- 40
    p <- 1
    predictor <- matrix(rnorm(n * p), nrow = n, ncol = p)
    response <- 1 + 2 * predictor[, 1] + rnorm(n, sd = 0.1)

    # glmnet requires 2+ columns, so this will trigger fallback warning
    expect_warning(result <- get_intercept_residual_lasso(response, predictor), "glmnet failed")

    expect_type(result, "list")
    expect_length(result$residual, n)
})

test_that("get_intercept_residual_lasso handles data.frame input", {
    n <- 30
    p <- 3
    predictor_df <- as.data.frame(matrix(rnorm(n * p), nrow = n, ncol = p))
    response <- rnorm(n)

    # Should not error
    result <- get_intercept_residual_lasso(response, predictor_df)

    expect_type(result, "list")
    expect_named(result, c("intercept", "residual"))
    expect_length(result$residual, n)
})
