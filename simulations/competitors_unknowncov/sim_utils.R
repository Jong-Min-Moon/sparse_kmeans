# ------------------------------------------------------------------
# sim_utils.R
# Shared utility functions for competitors_unknowncov simulations
# ------------------------------------------------------------------
source("../../code_r/data_generator.R")
source("methods_wrapper.R")
source("accuracy_utils.R")

#' Ensure directory exists
ensure_dir <- function(path) {
    if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE, showWarnings = FALSE)
    }
}

#' Generate data using the new pipeline
generate_data <- function(n, p, sep, seed, noise_type) {
    # Fixed parameters as per requirement
    support <- 1:10
    K <- 2
    rho <- 0.45
    precision_sparsity <- 2
    
    spec <- get_specification_chaingraph(
        support = support,
        separation = sep,
        dimension = p,
        precision_sparsity = precision_sparsity,
        conditional_correlation = rho,
        flip = FALSE
    )
    
    res <- generate_data_from_specification(
        specification = spec,
        n = n,
        seed = seed,
        noise = noise_type
    )
    
    return(list(X = t(res$X), true_labels = res$labels, spec = spec))
}

#' Run methods and return results block
try_method <- function(expr, name) {
    tryCatch({
        expr
    }, error = function(e) {
        warning(sprintf("%s failed: %s", name, e$message))
        return(NA)
    })
}

run_simulation_methods <- function(X, true_labels, K, p, n, sep, rho, job_id, seed) {
    pvalcut <- log(p) / p
    
    st_run <- Sys.time()
    
    methods_out <- tryCatch({
        run_all_methods(X, K, pvalcut, seed = seed)
    }, error = function(e) {
        warning(paste("run_all_methods failed:", e$message))
        NULL
    })
    
    if (is.null(methods_out)) {
         return(list(res_df = NULL, log_msg = sprintf("[%s] Rep %d, p = %d: run_all_methods ENCOUNTERED ERROR\n", Sys.time(), job_id, p)))
    }
    
    acc_out <- compute_all_accuracies(methods_out, true_labels)
    
    res_df <- data.frame(
        job_id = job_id,
        p = p,
        n = n,
        sep = sep,
        rho = rho,
        accuracy_witten = acc_out$acc_witten,
        runtime_witten = as.numeric(methods_out$witten$runtime),
        accuracy_arias = acc_out$acc_arias,
        runtime_arias = as.numeric(methods_out$arias$runtime),
        accuracy_ifpca = acc_out$acc_ifpca,
        ifpca_L = as.numeric(methods_out$ifpca$L),
        runtime_ifpca = as.numeric(methods_out$ifpca$runtime)
    )
    
    log_msg <- sprintf(
        "[%s] Rep %d, p = %d: Witten [feat = %s, acc = %.3f], Arias [feat = %s, acc = %.3f], IF-PCA [feat = %s, acc = %.3f]\n",
        format(Sys.time(), "%Y-%m-%d %H:%M:%S"), job_id, p,
        ifelse(is.na(methods_out$witten$L), "NA", as.character(methods_out$witten$L)), acc_out$acc_witten,
        ifelse(is.na(methods_out$arias$L), "NA", as.character(methods_out$arias$L)), acc_out$acc_arias,
        ifelse(is.na(methods_out$ifpca$L), "NA", as.character(methods_out$ifpca$L)), acc_out$acc_ifpca
    )
    
    return(list(res_df = res_df, log_msg = log_msg))
}

#' Check if result already exists (.rds)
check_progress <- function(results_dir, job_id, p) {
    filename <- file.path(results_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
    return(file.exists(filename))
}

#' Save result as .rds
save_result <- function(res_df, results_dir, job_id, p) {
    if (is.null(res_df)) return()
    filename <- file.path(results_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
    saveRDS(res_df, file = filename)
}

#' Log progress to file
log_progress <- function(log_file, msg) {
    cat(msg)
    # Use suppressWarnings because multiple processes appending to same file might trigger warnings
    suppressWarnings(
        tryCatch(
            cat(msg, file = log_file, append = TRUE),
            error = function(e) NULL
        )
    )
}
