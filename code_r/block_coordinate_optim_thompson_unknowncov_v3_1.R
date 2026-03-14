# Helper functions for Block Coordinate Optimization with Thompson Sampling (Known Covariance)

# Dependencies should be loaded by the driver script
# source("sdp_kmeans.R")
# source("utils.R")
# source("clustering_block_knowncov.R")
# source("selection_block_greedy_screening.R")
# source("cluster_spectral.R")

#' Block Coordinate Optimization with Thompson Sampling (Unknown Covariance, v3_1 Oracle ISEE ALWAYS)
#'
#' Implements the block coordinate ascent with Thompson Sampling for feature selection.
#' Version 3.1 Experimental Control: Force-inputs true cluster assignments into the ISEE step for EVERY iteration.
#'
#' @param X Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Number of iterations (default 10)
#' @param C Confidence parameter for threshold (default 0.5)
#' @param n_perms Number of permutations for reward step (default 100)
#' @param true_labels True cluster assignments (REQUIRED for v3_1)
#' @return List containing cluster assignments, selected features, and metrics
#' @export
block_coordinate_optim_thompson_unknowncov_v3_1 <- function(X, K, n_iter = 100, C = 0.5, n_perms = 200, p_val_threshold = 0.01, max_iter_sdp = 4000, true_labels = NULL) {
    if (!is.matrix(X)) stop("X must be a matrix")
    if (is.null(true_labels)) {
        stop("true_labels must be provided for version 3.1 (Oracle ISEE ALWAYS).")
    }

    p <- nrow(X)

    # Initialization
    alpha_vec <- rep(1, p)
    beta_vec <- rep(1, p)

    # Calculate cutoff threshold
    if (C <= 0 || C >= 1) {
        warning("C should be in (0, 1). Using default 0.5.")
        C <- 0.5
    }
    cutoff <- log(1 / C) / log((C + 1) / C)

    # Metrics
    rand_vec <- numeric(n_iter)
    acc_vec <- numeric(n_iter)
    obj_vec <- numeric(n_iter) # Track Objective Function
    selected_history <- list()
    reward_history <- list()

    # =========================================================
    # 1. INITIALIZATION BLOCK
    # =========================================================
    # Initialization: Sample from Prior (Beta(1,1))
    cat("Running initial sampling from prior (Beta(1,1))...\n")
    theta_init <- rbeta(p, alpha_vec, beta_vec)
    S_hat_now <- which(theta_init > cutoff)

    if (length(S_hat_now) == 0) {
        warning("Initialization selected 0 features. Forcing random selection of 10 features.")
        S_hat_now <- sample(1:p, 10, replace = FALSE)
    }

    cat(sprintf("Initial subset size: %d. Running initial clustering on subset...\n", length(S_hat_now)))

    # Initial clustering (ESSC or similar)
    res_init <- kernlab::specc(t(X), centers = K)
    cluster_est_now <- as.integer(res_init)

    if (!is.null(true_labels)) {
        ari_init <- mclust::adjustedRandIndex(cluster_est_now, true_labels)
        cat(sprintf("Initial Clustering Accuracy (ARI): %.4f\n", ari_init))
    }

    for (iternum in 1:n_iter) {
        cat(sprintf("\n--- Iteration %d (Thompson Sampling v3.1 Oracle ISEE) ---\n", iternum))

        # =========================================================
        # 2. SELECTION BLOCK
        # =========================================================

        # Version 3.1: ALWAYS use true cluster labels for ISEE transformation
        cat("Using true cluster labels for ISEE transformation (Oracle Mode).\n")
        cluster_isee_input <- true_labels

        res_isee <- ISEE_residual_lasso(X, cluster_isee_input, K)
        X_tilde <- res_isee$X_tilde

        # -------------------------------------------------------
        # A. Reward Step
        # -------------------------------------------------------
        # Subset X_tilde based on CURRENT selection indices (S_hat_now)
        X_tilde_sub <- X_tilde[S_hat_now, , drop = FALSE]

        # Run Greedy Screening (Permutation Test) on the subset
        rewards_sub <- select_greedily(X_tilde_sub, cluster_est_now, fdr_level = NULL, n_perms = 10000, p_val_threshold = p_val_threshold)

        # -------------------------------------------------------
        # B. Update Step
        # -------------------------------------------------------
        current_indices <- S_hat_now
        alpha_vec[current_indices] <- alpha_vec[current_indices] + as.numeric(rewards_sub)
        beta_vec[current_indices] <- beta_vec[current_indices] + (1 - as.numeric(rewards_sub))

        cat(sprintf("Rewarded arms: %d / %d selected in this subset\n", sum(rewards_sub), length(current_indices)))

        # --- Enhanced Logging: Top 10 Rewarded vs Un-rewarded Arms ---
        post_mean_all <- alpha_vec / (alpha_vec + beta_vec)

        # Rewarded Arms
        rewarded_indices <- current_indices[which(rewards_sub)]
        if (length(rewarded_indices) > 0) {
            n_show_r <- min(10, length(rewarded_indices))
            rewarded_probs <- post_mean_all[rewarded_indices]
            top_r_idx <- order(rewarded_probs, decreasing = TRUE)[1:n_show_r]
            cat(sprintf("Top %d REWARDED Arms (Posterior Mean):\n", n_show_r))
            print(setNames(rewarded_probs[top_r_idx], rewarded_indices[top_r_idx]))
        }

        # Un-rewarded (but selected) Arms
        unrewarded_indices <- current_indices[which(!rewards_sub)]
        if (length(unrewarded_indices) > 0) {
            n_show_u <- min(10, length(unrewarded_indices))
            unrewarded_probs <- post_mean_all[unrewarded_indices]
            top_u_idx <- order(unrewarded_probs, decreasing = TRUE)[1:n_show_u]
            cat(sprintf("Top %d UN-REWARDED (Selected) Arms (Posterior Mean):\n", n_show_u))
            print(setNames(unrewarded_probs[top_u_idx], unrewarded_indices[top_u_idx]))
        }
        # -------------------------------------------------------------
        rewards_full <- rep(0, p)
        rewards_full[current_indices] <- as.numeric(rewards_sub)
        reward_history[[iternum]] <- rewards_full

        # -------------------------------------------------------
        # C. Choose Step (Thompson Sampling)
        # -------------------------------------------------------
        theta_sample <- rbeta(p, alpha_vec, beta_vec)
        current_selection_logical <- theta_sample > cutoff
        S_hat_next <- which(current_selection_logical)
        n_selected <- length(S_hat_next)

        cat(sprintf("Arms pulled (Features selected for next step): %d / %d\n", n_selected, p))
        selected_history[[iternum]] <- current_selection_logical

        if (n_selected == 0) {
            warning("No features selected. Keeping previous selection for clustering.")
            S_hat_next <- S_hat_now
        }

        # =========================================================
        # 3. CLUSTERING BLOCK
        # =========================================================
        Sigma_hat_small <- get_cov_small(X, cluster_est_now, S_hat_next)
        Sigma_full <- diag(1, p)
        if (length(S_hat_next) > 0) {
            Sigma_full[S_hat_next, S_hat_next] <- Sigma_hat_small
        }

        res_blocking <- run_clustering_block_knowncov(
            X_tilde = X_tilde,
            selected_features = S_hat_next,
            K = K,
            cluster_est_prev = cluster_est_now,
            covariance = Sigma_full
        )
        cluster_est_new <- res_blocking$cluster
        obj_val <- res_blocking$value

        obj_vec[iternum] <- obj_val
        cat(sprintf("SDP Objective Value: %.4f\n", obj_val))

        ri <- mclust::adjustedRandIndex(cluster_est_new, cluster_est_now)
        rand_vec[iternum] <- ri
        cat(sprintf("Adjusted Rand Index change (prev vs now): %.4f\n", ri))

        if (!is.null(true_labels)) {
            ari_true <- mclust::adjustedRandIndex(cluster_est_new, true_labels)
            acc_true <- get_cluster_acc(cluster_est_new, true_labels)
            acc_vec[iternum] <- acc_true
            cat(sprintf("Iteration %d Evaluation: ARI=%.4f, Acc=%.4f\n", iternum, ari_true, acc_true))
        }

        cluster_est_now <- cluster_est_new
        S_hat_now <- S_hat_next
    }

    posterior_mean <- alpha_vec / (alpha_vec + beta_vec)
    final_selection <- posterior_mean > 0.5

    cat(sprintf("\n--- Final Selection ---\n"))
    cat(sprintf("Features selected: %d / %d\n", sum(final_selection), p))

    return(list(
        cluster = cluster_est_now,
        selected = final_selection,
        alpha = alpha_vec,
        beta = beta_vec,
        objective = obj_vec,
        rand_history = rand_vec,
        acc_history = acc_vec
    ))
}
