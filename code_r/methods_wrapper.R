# ------------------------------------------------------------------
# methods_wrapper.R
# Provides a uniform interface for calling sparse K-means methods.
# Now includes silhouette-based grid search for scvxclustr.
# ------------------------------------------------------------------
# Attempt to find the directory where this script resides for robust sourcing
# This works whether sourced or run via Rscript
script_dir <- "."
if (exists("utils::getSrcDirectory")) {
    d <- utils::getSrcDirectory(function(x) {x})
    if (nchar(d) > 0) script_dir <- d
}

# If the above fails (e.g. run via Rscript directly), we can try to find the files in common locations
if (!file.exists(file.path(script_dir, "competitors_modernized.R"))) {
    # Try looking in code_r relative to root or current
    if (file.exists("code_r/competitors_modernized.R")) {
        script_dir <- "code_r"
    } else if (file.exists("../../code_r/competitors_modernized.R")) {
        script_dir <- "../../code_r"
    }
}

source(file.path(script_dir, "competitors_modernized.R"))
source(file.path(script_dir, "ifpca.R"))

#' Run all spatial clustering methods on the same input data
#'
#' @param X Data matrix (n samples x p features expected by generator, handled internally)
#' @param K Number of clusters
#' @param pvalcut p-value cut threshold for IF-PCA
#' @param seed Random seed for reproducibility
#' @return A list containing cluster assignments and runtime for each method
run_all_methods <- function(X, K, pvalcut, seed) {
    p <- ncol(X)
    n <- nrow(X)

    # ---------------------------
    # 1. Witten's Sparse K-Means
    # ---------------------------
    st <- Sys.time()
    witten_res <- tryCatch(
        {
            run_witten(X, K, seed = seed, return_list = TRUE)
        },
        error = function(e) {
            warning(paste("Witten failed:", e$message))
            list(cluster = rep(NA, n), L = NA)
        }
    )
    rt_witten <- as.numeric(difftime(Sys.time(), st, units = "secs"))

    # ---------------------------
    # 2. Arias-Castro Sparse K-Means
    # ---------------------------
    st <- Sys.time()
    arias_res <- tryCatch(
        {
            run_arias(X, K, seed = seed, return_list = TRUE)
        },
        error = function(e) {
            warning(paste("Arias failed:", e$message))
            list(cluster = rep(NA, n), L = NA)
        }
    )
    rt_arias <- as.numeric(difftime(Sys.time(), st, units = "secs"))

    # ---------------------------
    # 3. IF-PCA
    # ---------------------------
    # IF-PCA strictly expects features in rows (p x n)
    # Since X is provided as n x p, we always transpose.
    X_ifpca <- t(X)

    st <- Sys.time()
    ifpca_res <- tryCatch(
        {
            if_pca(Data = X_ifpca, K = K, rep = 500, nullsimu = TRUE, pvalcut = pvalcut, kmeansrep = 20, per = 1, seed = seed)
        },
        error = function(e) {
            warning(paste("IF-PCA failed:", e$message))
            NULL
        }
    )
    rt_ifpca <- as.numeric(difftime(Sys.time(), st, units = "secs"))

    # ---------------------------
    # 4. Sparse Convex Clustering (scvxclustr)
    # ---------------------------
    st_scvx <- Sys.time()
    
    # 4a. SCVX Weights and Step Size
    n_val <- nrow(X)
    assign("n", n_val, envir = .GlobalEnv) # scvxclustr internal dependency
    
    scvx_pipeline <- tryCatch({
        # Weights matching user's established pattern
        w_raw <- scvxclustr::dist_weight(t(X) / sqrt(p), phi = 0.5, dist.type = "euclidean", p = 2)
        w <- cvxclustr::knn_weights(w_raw, k = 5, n = n_val)
        nu_val <- scvxclustr::AMA_step_size(w, n = n_val) / 2
        
        # Grid Search Configuration (Widened for robustness)
        g1_grid <- c(1, 10, 100)
        g2_grid <- c(0.1, 1, 10)
        
        model_results <- list()
        selected_list <- list()
        
        # 4b. Execute Grid
        for (g1 in g1_grid) {
            for (g2 in g2_grid) {
                key <- paste0("g1_", g1, "_g2_", g2)
                fit <- tryCatch({
                    scvxclustr::scvxclust(as.matrix(X), w = w, Gamma1 = g1, Gamma2 = g2, 
                                         Gamma2_weight = rep(1, p), method = "ama", nu = nu_val)
                }, error = function(e) { NULL })
                
                if (!is.null(fit)) {
                    # Extract Labels (1e-3 tolerance)
                    V_mat <- fit$V[[1]]
                    diffs <- apply(V_mat, 2, function(x) norm(as.matrix(x), "f"))
                    conn_ix <- which(diffs < 1e-3)
                    ix_all <- scvxclustr:::vec2tri(which(w > 0), n_val)
                    A_adj <- Matrix::Matrix(0, n_val, n_val, sparse = TRUE)
                    if (length(conn_ix) > 0) A_adj[(ix_all[conn_ix, 2] - 1) * n_val + ix_all[conn_ix, 1]] <- 1
                    
                    # IGRAH BUG WORKAROUND: graph.adjacency crashes on all-zero sparse matrix
                    if (any(A_adj != 0)) {
                        G <- igraph::graph.adjacency(A_adj, mode = 'upper')
                        labels <- igraph::clusters(G)$membership
                    } else {
                        labels <- 1:n_val
                    }
                    
                    # Extract Selected Features (1e-6 tolerance)
                    U_mat <- fit$U[[1]]
                    selected <- colSums(abs(U_mat)^2) > 1e-6
                    
                    model_results[[key]] <- list(cluster = labels, selected = selected, g1 = g1, g2 = g2)
                    selected_list[[key]] <- selected
                }
            }
        }
        
        # 4c. Model Selection (Silhouette on Union of Selected Features)
        best_res <- NULL
        best_sil <- -1
        merged_selected <- rep(FALSE, p)
        
        if (length(selected_list) > 0) {
            merged_selected <- Reduce(`|`, selected_list)
            merged_indices <- which(merged_selected)
            
            if (length(merged_indices) > 0) {
                # Precompute distance matrix on selected features for speed
                dist_mat <- dist(as.matrix(X)[, merged_indices, drop = FALSE])
                
                for (key in names(model_results)) {
                    res_temp <- model_results[[key]]
                    if (length(unique(res_temp$cluster)) > 1 && length(unique(res_temp$cluster)) < n_val) {
                        sil <- cluster::silhouette(res_temp$cluster, dist_mat)
                        if (!is.null(sil) && !any(is.na(sil))) {
                            avg_sil <- mean(sil[, 3])
                            
                            if (avg_sil > best_sil) {
                                best_sil <- avg_sil
                                best_res <- res_temp
                                best_res$silhouette <- avg_sil
                            }
                        }
                    }
                }
            }
        }
        
        if (is.null(best_res)) {
            list(cluster = rep(NA, n_val), selected = merged_selected)
        } else {
            # Update selected to the union as per requirements
            best_res$selected <- merged_selected
            best_res
        }
        
    }, error = function(e) {
        cat(sprintf("   SCVX Master Error: %s\n", e$message))
        list(cluster = rep(NA, n_val), selected = rep(FALSE, p))
    })
    
    rt_scvx <- as.numeric(difftime(Sys.time(), st_scvx, units = "secs"))

    # ---------------------------
    # 5. Result Aggregation
    # ---------------------------
    return(list(
        witten = list(
            cluster = witten_res$cluster,
            L = witten_res$L,
            runtime = rt_witten
        ),
        arias = list(
            cluster = arias_res$cluster,
            L = arias_res$L,
            runtime = rt_arias
        ),
        ifpca = list(
            cluster = if (!is.null(ifpca_res)) ifpca_res$labels else rep(NA, n),
            L = if (!is.null(ifpca_res)) ifpca_res$L else NA,
            runtime = rt_ifpca
        ),
        scvx = list(
            cluster = scvx_pipeline$cluster,
            selected = if (!is.null(scvx_pipeline$selected)) scvx_pipeline$selected else rep(FALSE, p),
            g1 = if (!is.null(scvx_pipeline$g1)) scvx_pipeline$g1 else NA,
            g2 = if (!is.null(scvx_pipeline$g2)) scvx_pipeline$g2 else NA,
            silhouette = if (!is.null(scvx_pipeline$silhouette)) scvx_pipeline$silhouette else NA,
            runtime = rt_scvx
        )
    ))
}
