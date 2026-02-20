#' Sparse Symmetric Data Generator
#'
#' Replicates the logic of bandit/sparse_symmetric_data_generator.m
#'
#' @param support Indices of signal features (e.g. 1:10)
#' @param separation Distance between cluster means
#' @param dimension Total number of features (p)
#' @param precision_sparsity Sparsity of the precision matrix (rho-based tridiagonal)
#' @param conditional_correlation The rho value (correlation/precision parameter)
#' @param flip Whether to treat the matrix as covariance instead of precision
#' @export
sparse_symmetric_data_generator <- function(support, separation, dimension, precision_sparsity, conditional_correlation, flip = FALSE) {
    # 1. Generate Sparse Precision Matrix
    sparse_precision_matrix <- diag(1, dimension)

    if (is.character(precision_sparsity)) {
        if (precision_sparsity == "nonsparse") {
            max_for_loop <- dimension - 1
        } else {
            max_for_loop <- 0
        }
    } else if (precision_sparsity >= 2) {
        max_for_loop <- floor(precision_sparsity / 2)
    } else {
        max_for_loop <- 0
    }

    if (max_for_loop > 0) {
        for (i in 1:max_for_loop) {
            val <- conditional_correlation^i
            # Upper diagonal
            for (j in 1:(dimension - i)) {
                sparse_precision_matrix[j, j + i] <- val
                sparse_precision_matrix[j + i, j] <- val
            }
        }
    }

    # 2. Covariance vs Precision
    if (flip) {
        covariance_matrix <- sparse_precision_matrix
        sparse_precision_matrix <- solve(covariance_matrix)
    } else {
        covariance_matrix <- solve(sparse_precision_matrix)
    }

    # 3. Magnitude Calculation
    # magnitude = separation / 2 / sqrt( sum( Sigma[support, support] ) )
    Sigma_S0 <- covariance_matrix[support, support]
    sum_Sigma_S0 <- sum(Sigma_S0)
    magnitude <- separation / 2 / sqrt(sum_Sigma_S0)

    # 4. Mean Vectors
    # sparse_pre_mean_one = indicator vector on support
    sparse_pre_mean_one <- rep(0, dimension)
    sparse_pre_mean_one[support] <- 1

    sparse_pre_mean_0 <- magnitude * sparse_pre_mean_one
    mean_0 <- covariance_matrix %*% sparse_pre_mean_0

    mu1 <- -mean_0 # cluster 1 mean
    mu2 <- mean_0 # cluster 2 mean

    return(list(
        support = support,
        separation = separation,
        dimension = dimension,
        precision_sparsity = precision_sparsity,
        rho = conditional_correlation,
        precision_matrix = sparse_precision_matrix,
        covariance_matrix = covariance_matrix,
        magnitude = magnitude,
        mu1 = as.numeric(mu1),
        mu2 = as.numeric(mu2)
    ))
}

#' Generate data using the generator
#' @param generator Result from sparse_symmetric_data_generator
#' @param n Sample size
#' @param seed Optional seed for reproducibility
#' @export
generate_data_from_generator <- function(generator, n, seed = NULL) {
    if (!is.null(seed)) set.seed(seed)

    X1 <- MASS::mvrnorm(n / 2, generator$mu1, generator$covariance_matrix)
    X2 <- MASS::mvrnorm(n / 2, generator$mu2, generator$covariance_matrix)
    X <- t(rbind(X1, X2))
    labels <- c(rep(1, n / 2), rep(2, n / 2))
    return(list(X = X, labels = labels))
}
