# ------------------------------------------------------------------
# scvx_wrapper.R
# Sparse Convex Clustering (scvxclustr) grid search + silhouette selection.
# Extracted from methods_wrapper.R to allow independent sourcing in
# conda-isolated environments where scvxclustr is separately installed.
# ------------------------------------------------------------------

#' Run SCVX grid search with silhouette-based model selection
#'
#' @param X Data matrix (n x p, rows = observations)
#' @param K Number of clusters (used only for fallback labelling context)
#' @param seed Random seed (currently unused directly; grid is deterministic)
#' @return A list: cluster, selected, g1, g2, silhouette, runtime
run_scvx <- function(X, K, seed) {
    p <- ncol(X)
    n_val <- nrow(X)

    st_scvx <- Sys.time()

    # scvxclustr has an internal dependency on a global `n` variable
    assign("n", n_val, envir = .GlobalEnv)

    scvx_pipeline <- tryCatch(
        {
            # Weights and AMA step size
            w_raw <- scvxclustr::dist_weight(t(X) / sqrt(p),
                phi = 0.5,
                dist.type = "euclidean", p = 2
            )
            w <- cvxclustr::knn_weights(w_raw, k = 5, n = n_val)
            nu_val <- scvxclustr::AMA_step_size(w, n = n_val) / 2

            # Grid search configuration
            g1_grid <- c(1, 100, 1000)
            g2_grid <- c(0.1, 1, 10)

            model_results <- list()
            selected_list <- list()

            for (g1 in g1_grid) {
                for (g2 in g2_grid) {
                    key <- paste0("g1_", g1, "_g2_", g2)
                    fit <- tryCatch(
                        {
                            scvxclustr::scvxclust(
                                as.matrix(X),
                                w = w,
                                Gamma1 = g1, Gamma2 = g2,
                                Gamma2_weight = rep(1, p),
                                method = "ama", nu = nu_val
                            )
                        },
                        error = function(e) NULL
                    )

                    if (!is.null(fit)) {
                        # Extract cluster labels (connectivity tolerance 1e-3 scaled by sqrt(p))
                        V_mat <- fit$V[[1]]
                        diffs <- apply(V_mat, 2, function(x) norm(as.matrix(x), "f"))
                        conn_ix <- which(diffs < 1e-3 * sqrt(p))
                        ix_all <- scvxclustr:::vec2tri(which(w > 0), n_val)
                        A_adj <- Matrix::Matrix(0, n_val, n_val, sparse = TRUE)
                        if (length(conn_ix) > 0) {
                            A_adj[(ix_all[conn_ix, 2] - 1) * n_val + ix_all[conn_ix, 1]] <- 1
                        }

                        # igraph::graph.adjacency crashes on all-zero sparse matrix
                        if (any(A_adj != 0)) {
                            G <- igraph::graph.adjacency(A_adj, mode = "upper")
                            labels <- igraph::clusters(G)$membership
                        } else {
                            labels <- 1:n_val
                        }

                        # Feature selection mask (L2-norm tolerance 1e-6)
                        U_mat <- fit$U[[1]]
                        selected <- colSums(abs(U_mat)^2) > 1e-6

                        model_results[[key]] <- list(
                            cluster = labels, selected = selected,
                            g1 = g1, g2 = g2
                        )
                        selected_list[[key]] <- selected
                    }
                }
            }

            # Silhouette-based model selection on the union of selected features
            best_res <- NULL
            best_sil <- -1
            merged_selected <- rep(FALSE, p)

            if (length(selected_list) > 0) {
                merged_selected <- Reduce(`|`, selected_list)
                merged_indices <- which(merged_selected)

                if (length(merged_indices) > 0) {
                    dist_mat <- dist(as.matrix(X)[, merged_indices, drop = FALSE])

                    for (key in names(model_results)) {
                        res_temp <- model_results[[key]]
                        n_clust <- length(unique(res_temp$cluster))
                        if (n_clust > 1 && n_clust < n_val) {
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
                best_res$selected <- merged_selected # use union of all selected features

                # Consolidate to exactly K clusters if SCVX over-partitioned.
                # Convex clustering routinely produces many micro-clusters at high p;
                # get_cluster_acc's Hungarian matching breaks down when K_est >> K_true.
                # Fix: k-means on selected features, warm-started from SCVX centroids.
                n_found <- length(unique(best_res$cluster))
                if (n_found > K) {
                    feat_idx <- which(merged_selected)
                    X_sub <- as.matrix(X)[, if (length(feat_idx) > 0) feat_idx else seq_len(p),
                        drop = FALSE
                    ]
                    # Build per-cluster centroids as k-means init
                    clust_ids <- unique(best_res$cluster)
                    init_cents <- do.call(rbind, lapply(clust_ids, function(cl) {
                        colMeans(X_sub[best_res$cluster == cl, , drop = FALSE])
                    }))
                    # Reduce centroids to K via k-means (handles K < n_found centroids)
                    if (nrow(init_cents) >= K) {
                        set.seed(seed)
                        km_init <- tryCatch(
                            kmeans(init_cents, centers = K, nstart = 10, iter.max = 50),
                            error = function(e) NULL
                        )
                        if (!is.null(km_init)) {
                            # Reassign original points to the K merged centroids
                            final_cents <- km_init$centers
                            set.seed(seed)
                            km_final <- tryCatch(
                                kmeans(X_sub, centers = final_cents, nstart = 1, iter.max = 100),
                                error = function(e) NULL
                            )
                            if (!is.null(km_final)) best_res$cluster <- km_final$cluster
                        }
                    }
                }

                best_res
            }
        },
        error = function(e) {
            cat(sprintf("   SCVX Master Error: %s\n", e$message))
            list(cluster = rep(NA, n_val), selected = rep(FALSE, p))
        }
    )

    rt_scvx <- as.numeric(difftime(Sys.time(), st_scvx, units = "secs"))
    scvx_pipeline$runtime <- rt_scvx
    scvx_pipeline
}
