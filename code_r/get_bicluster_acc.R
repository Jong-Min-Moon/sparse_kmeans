#' Get Bicluster Accuracy
#'
#' Replicate MATLAB logic for accuracy calculation, allowing for label permutations.
#' @param cluster_est Estimated labels
#' @param cluster_true True labels
#' @export
get_bicluster_acc <- function(cluster_est, cluster_true) {
    n <- length(cluster_true)
    # Ensure labels are 1 and 2
    u1 <- unique(cluster_true)
    # Case 1: Direct match
    acc1 <- sum(cluster_est == cluster_true)
    # Case 2: Permutation (swap 1 and 2)
    # This assumes K=2
    cluster_est_flipped <- 3 - cluster_est
    acc2 <- sum(cluster_est_flipped == cluster_true)

    return(max(acc1, acc2) / n)
}
