# ------------------------------------------------------------------
# methods_wrapper.R
# Provides a uniform interface for calling sparse K-means competitor methods.
# Defines run_simulation_methods(), which merges method execution, accuracy
# computation, and result formatting into a single pipeline step.
# ------------------------------------------------------------------

# Resolve the directory of this script for robust relative sourcing
script_dir <- "."
if (exists("utils::getSrcDirectory")) {
    d <- utils::getSrcDirectory(function(x) {
        x
    })
    if (nchar(d) > 0) script_dir <- d
}

if (!file.exists(file.path(script_dir, "competitors_modernized.R"))) {
    if (file.exists("code_r/competitors_modernized.R")) {
        script_dir <- "code_r"
    } else if (file.exists("../../code_r/competitors_modernized.R")) {
        script_dir <- "../../code_r"
    }
}

source(file.path(script_dir, "competitors_modernized.R"))
source(file.path(script_dir, "ifpca.R"))
source(file.path(script_dir, "scvx_wrapper.R"))
# source(file.path(script_dir, "clustvarsel_wrapper.R"))
source(file.path(script_dir, "get_cluster_acc.R"))

#' Safe single-method wrapper: returns NA list on failure
#'
#' @param expr  Expression to evaluate
#' @param name  Method name (for warning message)
#' @param n     Number of observations (for NA fallback)
#' @return Result of expr, or a list(cluster = rep(NA, n), L = NA) on error
.try_method <- function(expr, name, n) {
    tryCatch(expr, error = function(e) {
        warning(sprintf("%s failed: %s", name, e$message))
        list(cluster = rep(NA, n), L = NA)
    })
}

#' .safe_acc: compute get_cluster_acc only when clusters are all non-NA
.safe_acc <- function(cluster, true_labels) {
    if (!any(is.na(cluster))) get_cluster_acc(cluster, true_labels) else NA
}

#' Run selected competitor methods and return a simulation result row
#'
#' This function merges what was previously split across run_all_methods()
#' (in methods_wrapper.R) and run_simulation_methods() (in sim_utils.R).
#' Caller gets back a flat data.frame row plus a formatted log message.
#'
#' @param X           Data matrix (n x p)
#' @param true_labels Ground truth cluster vector (length n)
#' @param K           Number of clusters
#' @param p           Feature dimension (number of columns in X)
#' @param n           Sample size (number of rows in X)
#' @param sep         Separation parameter (stored in result, not used here)
#' @param rho         Correlation parameter (stored in result, not used here)
#' @param job_id      Replicate index (stored in result)
#' @param seed        Random seed passed to each method
#' @param methods     Character vector selecting which methods to run.
#'                    Any subset of c("witten", "arias", "ifpca", "scvx", "cvs").
#'                    Defaults to all five.
#' @return list(res_df = data.frame(...), log_msg = character(1))
run_simulation_methods <- function(X, true_labels, K, p, n, sep, rho, job_id, seed,
                                   methods = c("witten", "arias", "ifpca", "scvx")) {
    pvalcut <- log(p) / p

    # ---- 1. Witten's Sparse K-Means ----------------------------------------
    if ("witten" %in% methods) {
        st <- Sys.time()
        witten_res <- .try_method(
            run_witten(X, K, seed = seed, return_list = TRUE),
            "Witten", n
        )
        rt_witten <- as.numeric(difftime(Sys.time(), st, units = "secs"))
    } else {
        witten_res <- list(cluster = rep(NA, n), L = NA)
        rt_witten <- NA
    }

    # ---- 2. Arias-Castro Sparse K-Means ------------------------------------
    if ("arias" %in% methods) {
        st <- Sys.time()
        arias_res <- .try_method(
            run_arias(X, K, seed = seed, return_list = TRUE),
            "Arias", n
        )
        rt_arias <- as.numeric(difftime(Sys.time(), st, units = "secs"))
    } else {
        arias_res <- list(cluster = rep(NA, n), L = NA)
        rt_arias <- NA
    }

    # ---- 3. IF-PCA ---------------------------------------------------------
    # IF-PCA expects features in rows (p x n); transpose from standard (n x p)
    if ("ifpca" %in% methods) {
        X_ifpca <- t(X)
        st <- Sys.time()
        ifpca_res <- tryCatch(
            if_pca(
                Data = X_ifpca, K = K, rep = 500, nullsimu = TRUE,
                pvalcut = pvalcut, kmeansrep = 20, per = 1, seed = seed
            ),
            error = function(e) {
                warning(paste("IF-PCA failed:", e$message))
                NULL
            }
        )
        rt_ifpca <- as.numeric(difftime(Sys.time(), st, units = "secs"))
    } else {
        ifpca_res <- NULL
        rt_ifpca <- NA
    }

    # ---- 4. Sparse Convex Clustering (scvxclustr) --------------------------
    if ("scvx" %in% methods) {
        scvx_res <- run_scvx(X, K, seed)
        rt_scvx <- scvx_res$runtime
    } else {
        scvx_res <- list(
            cluster = rep(NA, n), selected = rep(FALSE, p),
            g1 = NA, g2 = NA, silhouette = NA
        )
        rt_scvx <- NA
    }


    # ---- 6. Accuracy -------------------------------------------------------
    acc_witten <- .safe_acc(witten_res$cluster, true_labels)
    acc_arias <- .safe_acc(arias_res$cluster, true_labels)
    acc_ifpca <- .safe_acc(
        if (!is.null(ifpca_res)) ifpca_res$labels else rep(NA, n),
        true_labels
    )
    acc_scvx <- .safe_acc(scvx_res$cluster, true_labels)
    acc_cvs <- .safe_acc(cvs_res$cluster, true_labels)

    # ---- 7. Result data.frame ----------------------------------------------
    res_df <- data.frame(
        job_id          = job_id,
        p               = p,
        n               = n,
        sep             = sep,
        rho             = rho,
        accuracy_witten = acc_witten,
        runtime_witten  = rt_witten,
        accuracy_arias  = acc_arias,
        runtime_arias   = rt_arias,
        accuracy_ifpca  = acc_ifpca,
        ifpca_L         = if (!is.null(ifpca_res)) as.numeric(ifpca_res$L) else NA,
        runtime_ifpca   = rt_ifpca,
        accuracy_scvx   = acc_scvx,
        runtime_scvx    = rt_scvx,
        accuracy_cvs    = acc_cvs,
        cvs_L           = if (!is.null(cvs_res$L) && !is.na(cvs_res$L)) as.numeric(cvs_res$L) else NA,
        runtime_cvs     = rt_cvs
    )

    # ---- 8. Log message ----------------------------------------------------
    log_msg <- sprintf(
        "[%s] Rep %d, p = %d: Witten [feat=%s, acc=%.3f], Arias [feat=%s, acc=%.3f], IF-PCA [feat=%s, acc=%.3f], SCVX [acc=%.3f], CVS [feat=%s, acc=%.3f]\n",
        format(Sys.time(), "%Y-%m-%d %H:%M:%S"), job_id, p,
        ifelse(is.na(witten_res$L), "NA", as.character(witten_res$L)), acc_witten,
        ifelse(is.na(arias_res$L), "NA", as.character(arias_res$L)), acc_arias,
        ifelse(is.null(ifpca_res) || is.na(ifpca_res$L), "NA",
            as.character(ifpca_res$L)
        ), acc_ifpca,
        acc_scvx,
        ifelse(is.na(cvs_res$L), "NA", as.character(cvs_res$L)), acc_cvs
    )

    list(res_df = res_df, log_msg = log_msg)
}
