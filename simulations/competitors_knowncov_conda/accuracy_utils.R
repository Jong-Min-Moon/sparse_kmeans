# ------------------------------------------------------------------
# accuracy_utils.R
# Handles optimal precision validation for clustering assignments.
# ------------------------------------------------------------------
source("../../code_r/get_cluster_acc.R")

#' Compute accuracy for multiple method outputs mapped against true labels
#'
#' @param methods_output The result list from `run_all_methods`
#' @param true_labels Ground truth vector
#' @return A flattened list dataframe row with the accuracies
compute_all_accuracies <- function(methods_output, true_labels) {
    acc_witten <- NA
    acc_arias <- NA
    acc_ifpca <- NA
    acc_bclust <- NA
    acc_scvx <- NA

    # Witten
    if (!any(is.na(methods_output$witten$cluster))) {
        acc_witten <- get_cluster_acc(methods_output$witten$cluster, true_labels)
    }

    # Arias
    if (!any(is.na(methods_output$arias$cluster))) {
        acc_arias <- get_cluster_acc(methods_output$arias$cluster, true_labels)
    }

    # IF-PCA
    if (!any(is.na(methods_output$ifpca$cluster))) {
        acc_ifpca <- get_cluster_acc(methods_output$ifpca$cluster, true_labels)
    }

    # B-Clust
    if (!any(is.na(methods_output$bclust$cluster))) {
        acc_bclust <- get_cluster_acc(methods_output$bclust$cluster, true_labels)
    }

    # SCVX
    if (!any(is.na(methods_output$scvx$cluster))) {
        acc_scvx <- get_cluster_acc(methods_output$scvx$cluster, true_labels)
    }

    return(list(
        acc_witten = acc_witten,
        acc_arias = acc_arias,
        acc_ifpca = acc_ifpca,
        acc_bclust = acc_bclust,
        acc_scvx = acc_scvx
    ))
}
