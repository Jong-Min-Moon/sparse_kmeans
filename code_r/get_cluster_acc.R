#' Get Multi-Cluster Accuracy
#'
#' Calculates clustering accuracy for K >= 2 clusters using the Hungarian algorithm
#' to find the optimal label permutation.
#' @param cluster_est Estimated labels (numeric, character, or factor)
#' @param cluster_true True labels (numeric, character, or factor)
#' @return A numeric value between 0 and 1 representing the maximum accuracy
#' @export
get_cluster_acc <- function(cluster_est, cluster_true) {
    # 1. Input validation
    if (length(cluster_est) != length(cluster_true)) {
        stop("Lengths of estimated and true labels must match.")
    }

    # 2. Check for the required package
    if (!requireNamespace("clue", quietly = TRUE)) {
        stop("The 'clue' package is required for the Hungarian algorithm. Please run install.packages('clue').")
    }

    n <- length(cluster_true)

    # 3. Standardize labels to guarantee a square confusion matrix
    # This handles edge cases where the algorithm missed a cluster or predicted extra ones.
    all_levels <- unique(c(as.character(cluster_est), as.character(cluster_true)))
    cluster_est_f <- factor(cluster_est, levels = all_levels)
    cluster_true_f <- factor(cluster_true, levels = all_levels)

    # 4. Create the confusion matrix
    conf_mat <- table(cluster_est_f, cluster_true_f)

    # 5. Apply the Hungarian algorithm
    # We set maximum = TRUE because we want to maximize the sum of correct matches
    optimal_mapping <- clue::solve_LSAP(conf_mat, maximum = TRUE)

    # 6. Calculate accuracy based on the optimal mapping
    # optimal_mapping gives the best column index for each row.
    # We extract and sum those specific matrix cells.
    optimal_matches <- sum(conf_mat[cbind(seq_along(optimal_mapping), optimal_mapping)])

    return(optimal_matches / n)
}
