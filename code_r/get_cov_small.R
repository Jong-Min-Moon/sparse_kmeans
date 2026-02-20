#' Get Small Covariance Matrix
#'
#' Estimates the covariance matrix for selected features based on current cluster labels.
#' Replicates the logic of get_cov_small.m
#'
#' @param X Data matrix (p x n)
#' @param cluster_est Current cluster assignments (labels 1 and 2)
#' @param s_hat Indices of selected features
#' @return Estimated covariance matrix (s x s) for selected features
#' @export
get_cov_small <- function(X, cluster_est, s_hat) {
    # Select variables using s_hat
    X_subset <- X[s_hat, , drop = FALSE]

    K <- length(unique(cluster_est))
    n <- ncol(X)

    # Residuals (centered data from both clusters)
    residuals_mat <- matrix(0, nrow = length(s_hat), ncol = n)

    for (k in unique(cluster_est)) {
        mask <- (cluster_est == k)
        if (sum(mask) >= 1) {
            cluster_data <- X_subset[, mask, drop = FALSE]
            cluster_mean <- rowMeans(cluster_data)
            residuals_mat[, mask] <- cluster_data - cluster_mean
        }
    }

    # Compute covariance matrix (s x s)
    # Scale by n-1 as in usual cov
    Sigma_hat <- tcrossprod(residuals_mat) / (n - 1)

    return(Sigma_hat)
}
