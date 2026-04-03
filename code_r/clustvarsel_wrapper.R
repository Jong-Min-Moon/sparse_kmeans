# ------------------------------------------------------------------
# clustvarsel_wrapper.R
# Wrapper for clustvarsel (Scrucca & Raftery 2016, J. Stat. Software 84(1))
# Variable selection for Gaussian model-based clustering.
#
# Design notes for high-dimensional use (p >> n):
#   - clustvarsel is a BIC-based stepwise forward/backward search over
#     all p candidate variables. For p in the thousands this is infeasible.
#   - We first prescreen features using marginal F-ratios (between-group
#     variance proxy: column-wise range of group means / pooled variance)
#     computed *without* using true labels — only column variance ordering
#     is available. We use the unsupervised colVar ranking, which retains
#     the most heterogeneous features. max_prescreen controls the budget.
#   - After prescreening: run clustvarsel (backward/forward headlong search)
#     on the reduced feature matrix.
#   - Final clusters come from Mclust on the selected subset.
#   - If clustvarsel selects zero variables, fall back to Mclust on all
#     prescreened features.
# ------------------------------------------------------------------

# clustvarsel uses compiled internal symbols that are only reachable when the
# package is fully *attached* (library()), not merely loaded (requireNamespace).
# These library() calls must be present in every worker environment that sources
# this file, including doParallel workers started via foreach's .packages arg.
library(mclust)
library(clustvarsel)

#' Run clustvarsel-based clustering as a competitor
#'
#' @param X              n x p data matrix (observations as rows)
#' @param K              Number of clusters (fixed)
#' @param seed           Optional random seed
#' @param prescreen_frac Fraction of columns to retain before running
#'                       clustvarsel (ranked by column variance). Default 0.20
#'                       (top 20%). For p=1000 this gives 200 screened features.
#'                       Keep in mind that the greedy BIC search is O(m^2) in
#'                       the prescreened size m, so large fractions at very
#'                       high p can be slow.
#' @return list with cluster (length-n label vector), L (# selected features),
#'         selected (logical, length p)
run_clustvarsel <- function(X, K = 2, seed = NULL, prescreen_frac = 0.20) {
    if (!is.null(seed)) set.seed(seed)

    n <- nrow(X)
    p <- ncol(X)

    # ------------------------------------------------------------------
    # 1. Unsupervised prescreening — keep the top (prescreen_frac * p)
    #    features by column variance. Provides a cheap unsupervised proxy
    #    for informativeness under identity covariance.
    # ------------------------------------------------------------------
    m <- max(ceiling(prescreen_frac * p), K + 2L)  # screen size, floor at K+2
    if (p > m) {
        col_vars  <- apply(X, 2, var)
        top_idx   <- order(col_vars, decreasing = TRUE)[seq_len(m)]
        X_screen  <- X[, top_idx, drop = FALSE]
    } else {
        top_idx  <- seq_len(p)
        X_screen <- X
    }

    # ------------------------------------------------------------------
    # 2. Run clustvarsel on the prescreened matrix.
    #    - G = K: fix the known number of clusters
    #    - search = "greedy", direction = "forward": starts from an empty
    #      variable set and adds features greedily by BIC improvement.
    #      Faster per step than backward (which starts from the full set).
    #    - emModels2 = "EII": spherical equal-volume (matches identity cov)
    #    - samp = TRUE: subsample hierarchical initialisation for speed
    #    - verbose = FALSE: suppress per-step printed output
    # ------------------------------------------------------------------
    cvs_out <- tryCatch(
        clustvarsel::clustvarsel(
            data       = X_screen,
            G          = K,
            search     = "greedy",
            direction  = "forward",
            emModels1  = c("E"),
            emModels2  = "EII",
            samp       = TRUE,
            sampsize   = max(K + 1, round(n / 2)),
            verbose    = FALSE,
            itermax    = 50
        ),
        error = function(e) {
            warning(paste("clustvarsel failed:", e$message))
            NULL
        }
    )

    # ------------------------------------------------------------------
    # 3. Extract selected features (as indices into X_screen).
    # ------------------------------------------------------------------
    if (!is.null(cvs_out) && length(cvs_out$subset) > 0) {
        sel_screen <- cvs_out$subset  # integer indices in X_screen
    } else {
        # Fallback: use all prescreened features
        sel_screen <- seq_len(ncol(X_screen))
    }

    X_sel <- X_screen[, sel_screen, drop = FALSE]

    # ------------------------------------------------------------------
    # 4. Fit final Mclust model on selected features.
    # ------------------------------------------------------------------
    clust_fit <- tryCatch(
        mclust::Mclust(X_sel, G = K, modelNames = "EII",
                       verbose = FALSE),
        error = function(e) {
            warning(paste("Mclust (post-clustvarsel) failed:", e$message))
            NULL
        }
    )

    if (!is.null(clust_fit)) {
        cluster_labels <- clust_fit$classification
    } else {
        # Last resort: random assignment
        cluster_labels <- sample(seq_len(K), n, replace = TRUE)
    }

    # ------------------------------------------------------------------
    # 5. Map selected indices back to the original p-dimensional space.
    # ------------------------------------------------------------------
    sel_original <- top_idx[sel_screen]
    selected_logical <- logical(p)
    selected_logical[sel_original] <- TRUE

    list(
        cluster  = cluster_labels,
        L        = length(sel_original),
        selected = selected_logical
    )
}
