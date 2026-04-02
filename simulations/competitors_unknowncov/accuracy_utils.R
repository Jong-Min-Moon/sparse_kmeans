# ------------------------------------------------------------------
# accuracy_utils.R
# Handles optimal precision validation for clustering assignments.
# NOTE: accuracy computation is now also available inline in
# run_simulation_methods (code_r/methods_wrapper.R). This file is
# kept for scripts that call compute_all_accuracies directly.
# ------------------------------------------------------------------
source("../../code_r/get_cluster_acc.R")

#' Compute accuracy for multiple method outputs mapped against true labels
#'
#' @param methods_output The result list from `run_simulation_methods`
#' @param true_labels Ground truth vector
#' @return A list with per-method accuracy values
compute_all_accuracies <- function(methods_output, true_labels) {
    acc_witten <- NA
    acc_arias  <- NA
    acc_ifpca  <- NA
    acc_scvx   <- NA

    if (!any(is.na(methods_output$witten$cluster))) {
        acc_witten <- get_cluster_acc(methods_output$witten$cluster, true_labels)
    }

    if (!any(is.na(methods_output$arias$cluster))) {
        acc_arias <- get_cluster_acc(methods_output$arias$cluster, true_labels)
    }

    if (!any(is.na(methods_output$ifpca$cluster))) {
        acc_ifpca <- get_cluster_acc(methods_output$ifpca$cluster, true_labels)
    }

    if (exists("scvx", where = methods_output) &&
        !any(is.na(methods_output$scvx$cluster))) {
        acc_scvx <- get_cluster_acc(methods_output$scvx$cluster, true_labels)
    }

    return(list(
        acc_witten = acc_witten,
        acc_arias  = acc_arias,
        acc_ifpca  = acc_ifpca,
        acc_scvx   = acc_scvx
    ))
}
