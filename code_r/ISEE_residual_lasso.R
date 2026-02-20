#' ISEE Residual Lasso (Separate Slopes, AIC Selection)
#'
#' Performs ISEE transformation using separate Lasso regressions per cluster.
#' Selection of lambda is done via AIC to match MATLAB parity.
#'
#' @param X Data matrix (p x n)
#' @param cluster_est Vector of cluster labels
#' @param K Number of clusters
#' @return List containing:
#'   \item{X_tilde}{Transformed data matrix (p x n)}
#'   \item{Omega_diag}{Diagonal of the estimated precision matrix (p)}
#' @export
ISEE_residual_lasso <- function(X, cluster_est, K) {
    p <- nrow(X)
    n <- ncol(X)

    x_t <- t(X)
    X_tilde <- matrix(0, nrow = p, ncol = n)
    Omega_diag_hat <- numeric(p)
    n_regression <- floor(p / 2)

    # Check for parallel backend
    if (!foreach::getDoParRegistered()) {
        warning("No parallel backend registered. Running sequentially.")
        registerDoSEQ()
    }

    results_list <- foreach::foreach(i = 1:n_regression, .packages = c("glmnet"), .export = c("get_intercept_residual_lasso_aic")) %dopar% {
        rows_idx <- c(2 * i - 1, 2 * i)
        predictor_all <- x_t[, -rows_idx, drop = FALSE]
        E_Al <- matrix(0, 2, n)
        alpha_Al <- matrix(0, 2, K)

        for (c in 1:K) {
            cluster_mask <- (cluster_est == c)
            if (sum(cluster_mask) < 2) next
            predictor_cluster <- predictor_all[cluster_mask, , drop = FALSE]
            for (j in 1:2) {
                row_id <- rows_idx[j]
                response_cluster <- x_t[cluster_mask, row_id]
                res <- get_intercept_residual_lasso_aic(response_cluster, predictor_cluster)
                E_Al[j, cluster_mask] <- res$residual
                alpha_Al[j, c] <- res$intercept
            }
        }

        # Robustness: variance floor
        scatter_mat <- tcrossprod(E_Al)
        Omega_hat_Al <- tryCatch(
            {
                solve(scatter_mat + diag(1e-4, 2)) * n
            },
            error = function(e) {
                diag(1 / pmax(diag(scatter_mat), 1e-4)) * n
            }
        )

        X_tilde_local <- Omega_hat_Al %*% (alpha_Al[, cluster_est] + E_Al)
        list(rows_idx = rows_idx, X_tilde_local = X_tilde_local, diag_local = diag(Omega_hat_Al))
    }

    for (res in results_list) {
        X_tilde[res$rows_idx, ] <- res$X_tilde_local
        Omega_diag_hat[res$rows_idx] <- res$diag_local
    }

    return(list(X_tilde = X_tilde, Omega_diag = Omega_diag_hat))
}
