# ------------------------------------------------------------------
# methods_wrapper.R
# Provides a uniform interface for calling sparse K-means methods.
# ------------------------------------------------------------------
source("../../code_r/competitors_modernized.R")
source("../../code_r/ifpca.R")

#' Run all spatial clustering methods on the same input data
#'
#' @param X Data matrix (n samples x p features expected by generator, handled internally)
#' @param K Number of clusters
#' @param pvalcut p-value cut threshold for IF-PCA
#' @param seed Random seed for reproducibility
#' @return A list containing cluster assignments and runtime for each method
run_all_methods <- function(X, K, pvalcut, seed) {
    p <- ncol(X)
    n <- nrow(X)

    # ---------------------------
    # 1. Witten's Sparse K-Means
    # ---------------------------
    st <- Sys.time()
    witten_res <- tryCatch(
        {
            run_witten(X, K, seed = seed, return_list = TRUE)
        },
        error = function(e) {
            warning(paste("Witten failed:", e$message))
            list(cluster = rep(NA, n), L = NA)
        }
    )
    rt_witten <- as.numeric(difftime(Sys.time(), st, units = "secs"))

    # ---------------------------
    # 2. Arias-Castro Sparse K-Means
    # ---------------------------
    st <- Sys.time()
    arias_res <- tryCatch(
        {
            run_arias(X, K, seed = seed, return_list = TRUE)
        },
        error = function(e) {
            warning(paste("Arias failed:", e$message))
            list(cluster = rep(NA, n), L = NA)
        }
    )
    rt_arias <- as.numeric(difftime(Sys.time(), st, units = "secs"))

    # ---------------------------
    # 3. IF-PCA
    # ---------------------------
    # IF-PCA explicitly expects features in rows (p x n)
    if (ncol(X) > nrow(X)) {
        X_ifpca <- t(X)
    } else {
        X_ifpca <- X
    }

    st <- Sys.time()
    ifpca_res <- tryCatch(
        {
            if_pca(Data = X_ifpca, K = K, rep = 500, nullsimu = TRUE, pvalcut = pvalcut, kmeansrep = 20, per = 1, seed = seed)
        },
        error = function(e) {
            warning(paste("IF-PCA failed:", e$message))
            NULL
        }
    )
    rt_ifpca <- as.numeric(difftime(Sys.time(), st, units = "secs"))

    return(list(
        witten = list(
            cluster = witten_res$cluster,
            L = witten_res$L,
            runtime = rt_witten
        ),
        arias = list(
            cluster = arias_res$cluster,
            L = arias_res$L,
            runtime = rt_arias
        ),
        ifpca = list(
            cluster = if (!is.null(ifpca_res)) ifpca_res$labels else rep(NA, n),
            L = if (!is.null(ifpca_res)) ifpca_res$L else NA,
            runtime = rt_ifpca
        )
    ))
}
