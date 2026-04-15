# ------------------------------------------------------------------
# sim_utils.R
# Shared utility functions for unknowncov_n200_ar1_competitors_modern_conda
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

#' Generate data using the AR(1) specification
generate_data_ar1_conda <- function(n, p, sep, seed, noise_type) {
    support <- 1:10
    spec <- get_specification_ar1(
        support    = support,
        separation = sep,
        dimension  = p,
        rho        = 0.8
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
check_progress <- function(results_dir, job_id, p,
                           methods = c("scvx")) {
    filename <- file.path(results_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
    if (!file.exists(filename)) return(FALSE)

    # Load and validate method coverage
    d <- tryCatch(readRDS(filename), error = function(e) NULL)
    if (is.null(d) || !is.data.frame(d)) return(FALSE)

    for (m in methods) {
        col <- paste0("accuracy_", m)
        # If the column is missing OR stored as logical (== was never computed)
        if (!col %in% names(d) || is.logical(d[[col]])) {
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
log_progress <- function(log_file, msg, console_only = FALSE) {
    if (is.null(msg) || msg == "") return()
    
    # Always print to console
    cat(msg)
    
    # Append to file if requested and provided
    if (!console_only && !is.null(log_file) && log_file != "") {
        suppressWarnings(
            tryCatch(
                {
                    con <- file(log_file, open = "a")
                    cat(msg, file = con)
                    close(con)
                },
                error = function(e) NULL
            )
        )
    }
}
