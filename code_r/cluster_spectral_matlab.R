#' MATLAB-style Spectral Clustering (Comparison Variant)
#'
#' This follows the logic in cluster_spectral.m
#' @param X Data matrix (p x n)
#' @param K Number of clusters
#' @export
cluster_spectral_matlab <- function(X, K) {
    p <- nrow(X)
    n <- ncol(X)

    # Affinity matrix (Gram matrix)
    H_hat <- crossprod(X) / n # n x n

    # Eigen decomposition
    eig_res <- eigen(H_hat, symmetric = TRUE)
    d <- eig_res$values
    V <- eig_res$vectors

    # Sort descending (eigen already returns sorted for symmetric)
    tau_n <- 1 / log(n + p)
    delta_n <- tau_n^2

    # Use sorted eigenvectors
    Vs <- V

    # f1 is the "loss of constant-ness" for Vs[,1]
    s1 <- abs(sum(Vs[, 1])) / sqrt(n)
    s2 <- abs(sum(Vs[, 2])) / sqrt(n)

    cat(sprintf(
        "Spectral Debug: d1=%.2f, d2=%.2f, d1/d2=%.3f, tau_n=%.3f, s1=%.3f, s2=%.3f, delta_n=%.3f\n",
        d[1], d[2], d[1] / d[2], tau_n, s1, s2, delta_n
    ))

    # Logic: If d1/d2 close to 1, use both.
    if (d[1] / d[2] < 1 + tau_n) {
        cat("Spectral: Choosing Vs[, 1:2]\n")
        new_data <- Vs[, 1:2, drop = FALSE]
    } else {
        # Pick between Vs1 and Vs2. The one that is LESS constant is usually the signal.
        if (s1 < s2) {
            cat("Spectral: Choosing Vs[, 1] (looks more like signal)\n")
            new_data <- Vs[, 1, drop = FALSE]
        } else {
            cat("Spectral: Choosing Vs[, 2] (Vs1 looks like constant)\n")
            new_data <- Vs[, 2, drop = FALSE]
        }
    }

    # K-means on the selected components
    km <- kmeans(new_data, centers = K, nstart = 20)
    return(as.integer(km$cluster))
}
