# Clustering Block for Unknown Covariance (ISEE)

# source("ISEE_bicluster.R") # Sourced by driver
# source("sdp_kmeans.R")     # Sourced by driver

#' Clustering Block for Unknown Covariance
#'
#' @param x Raw data matrix (p x n)
#' @param selected_features Logical vector of selected features
#' @param K Number of clusters
#' @param cluster_est_prev Previous cluster assignments (used for ISEE initialization)
#' @return New cluster assignments (vector of length n)
run_clustering_block_unknowncov <- function(x, selected_features, K, cluster_est_prev) {
    # 1. ISEE Estimation (The Handshake)
    # Estimate denoised matrix X_tilde based on previous clusters
    cat("Running ISEE in Clustering Block...\n")
    isee_res <- ISEE_bicluster(x, cluster_est_prev)
    X_tilde <- isee_res$X_tilde

    # 2. Sub-matrix Extraction
    # Subset X_tilde using selected_features
    X_tilde_sub <- X_tilde[selected_features, , drop = FALSE]

    # 3. Covariance Calculation (Crucial Step)
    # Calculate sample covariance of original x using selected_features
    x_sub <- x[selected_features, , drop = FALSE]

    # Demean per cluster to remove cluster effect (Pooled Covariance estimation)
    x_sub_centered <- x_sub
    unique_clusters <- unique(cluster_est_prev)
    for (k in unique_clusters) {
        cluster_idx <- (cluster_est_prev == k)
        if (sum(cluster_idx) > 0) {
            cluster_data <- x_sub[, cluster_idx, drop = FALSE]
            cluster_mean <- rowMeans(cluster_data)
            # Subtract cluster mean from each column in that cluster
            x_sub_centered[, cluster_idx] <- sweep(cluster_data, 1, cluster_mean, "-")
        }
    }

    # Calculate covariance of centered data
    cov_sub <- cov(t(x_sub_centered))

    # 4. The Handshake (Delegation)
    # Call run_clustering_block_knowncov
    # x: X_tilde_sub
    # selected_features: all rows of X_tilde_sub (indices 1:nrow)
    # covariance: cov_sub

    res <- run_clustering_block_knowncov(
        X_tilde = X_tilde_sub,
        selected_features = seq_len(nrow(X_tilde_sub)),
        K = K,
        cluster_est_prev = cluster_est_prev,
        covariance = cov_sub
    )

    return(res)
}
