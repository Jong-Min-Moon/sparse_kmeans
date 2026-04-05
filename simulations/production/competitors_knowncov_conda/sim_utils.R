# ------------------------------------------------------------------
# sim_utils.R
# Shared utility functions for competitors_knowncov_conda simulations.
# run_simulation_methods is defined in code_r/methods_wrapper.R.
# ------------------------------------------------------------------
source("../../../code_r/data_generator.R")
source("../../../code_r/methods_wrapper.R")

#' Ensure directory exists
ensure_dir <- function(path) {
    if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE, showWarnings = FALSE)
    }
}

#' Generate data using the identity (known-covariance) specification
generate_data_knowncov <- function(n, p, sep, seed, noise_type) {
    support <- 1:10

    spec <- get_specification_identity(
        support    = support,
        separation = sep,
        dimension  = p
    )

    res <- generate_data_from_specification(
        specification = spec,
        n             = n,
        seed          = seed,
        noise         = noise_type
    )

    return(list(X = t(res$X), true_labels = res$labels, spec = spec))
}

#' Check if result already exists AND covers all requested methods.
#'
#' A checkpoint is considered valid only when every requested method has a
#' numeric (non-logical) accuracy entry in the saved data.frame. This prevents
#' silently re-using stale files written during a narrower methods subset
#' (e.g. methods = c("scvx") only), which would leave NAs in the aggregation.
#'
#' @param results_dir  Directory containing checkpoint .rds files
#' @param job_id       Replicate index
#' @param p            Feature dimension
#' @param methods      Character vector of methods that must be present
#'                     (default: all four standard methods)
#' @return TRUE if a valid, complete checkpoint exists; FALSE otherwise
check_progress <- function(results_dir, job_id, p,
                           methods = c("witten", "arias", "ifpca", "scvx", "cvs")) {
    filename <- file.path(results_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
    if (!file.exists(filename)) return(FALSE)

    # Load and validate method coverage
    d <- tryCatch(readRDS(filename), error = function(e) NULL)
    if (is.null(d) || !is.data.frame(d)) return(FALSE)

    for (m in methods) {
        col <- paste0("accuracy_", m)
        # If the column is missing OR stored as logical (== was never computed)
        if (!col %in% names(d) || is.logical(d[[col]])) {
            message(sprintf(
                "[checkpoint] Stale file %s missing numeric results for '%s'. Will re-run.",
                basename(filename), m
            ))
            return(FALSE)
        }
    }
    return(TRUE)
}

#' Save a result data.frame as .rds
save_result <- function(res_df, results_dir, job_id, p) {
    if (is.null(res_df)) return()
    filename <- file.path(results_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
    saveRDS(res_df, file = filename)
}

#' Append a progress message to log file (safe for parallel writes)
log_progress <- function(log_file, msg) {
    cat(msg)
    suppressWarnings(
        tryCatch(
            cat(msg, file = log_file, append = TRUE),
            error = function(e) NULL
        )
    )
}
