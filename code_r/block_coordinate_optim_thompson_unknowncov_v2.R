# Helper functions for Block Coordinate Optimization with Thompson Sampling (Known Covariance)

# Dependencies should be loaded by the driver script
# source("sdp_kmeans.R")
# source("utils.R")
# source("clustering_block_knowncov.R")
# source("selection_block_greedy_screening.R")
# source("cluster_spectral.R")
# if (!require(mclust)) install.packages("mclust")
# if (!require(CVXR)) install.packages("CVXR")
# if (!require(MASS)) install.packages("MASS")
# library(mclust)
# library(CVXR)
# library(MASS)

#' Block Coordinate Optimization with Thompson Sampling (Unknown Covariance)
#'
#' Implements the block coordinate ascent with Thompson Sampling for feature selection.
#' Uses clustering_block_knowncov for the clustering step.
#'
#' @param X Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Number of iterations (default 10)
#' @param C Confidence parameter for threshold (default 0.5)
#' @param n_perms Number of permutations for reward step (default 100)
#' @param true_labels True cluster assignments (optional, if provided ARI will be logged)
#' @return List containing cluster assignments, selected features, and metrics
#' @export
block_coordinate_optim_thompson_unknowncov <- function(X, K, n_iter = 100, C = 0.5, n_perms = 200, fdr_level = 0.1, max_iter_sdp = 4000, true_labels = NULL) {
    if (!is.matrix(X)) stop("X must be a matrix")

    p <- nrow(X)
    n <- ncol(X)

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
    obj_vec <- numeric(n_iter) # Track Objective Function
    selected_history <- list()
    reward_history <- list()


    # =========================================================
    # 1. INITIALIZATION BLOCK
    # =========================================================
    # Initialization: Sample from Prior (Beta(1,1))
    cat("Running initial sampling from prior (Beta(1,1))...\n")
    # Sample theta ~ Beta(1,1) (which is Uniform(0,1))
    # alpha_vec and beta_vec are initialized to 1s.


    theta_init <- rbeta(p, alpha_vec, beta_vec)
    S_hat_now <- which(theta_init > cutoff)

    if (length(S_hat_now) == 0) {
        warning("Initialization selected 0 features. Forcing random selection of 10 features.")
        S_hat_now <- sample(1:p, 10, replace = FALSE)
    }

    cat(sprintf("Initial subset size: %d. Running initial clustering on subset...\n", length(S_hat_now)))

    # Create logical vector for clustering function
    current_selection_logical_init <- rep(FALSE, p)
    current_selection_logical_init[S_hat_now] <- TRUE

    # Run Clustering on this initial subset
    # We pass cluster_est_prev = NULL as there is no previous estimate
    res_init <- kernlab::specc(t(X), centers = K)
    cluster_est_now <- as.integer(res_init)

    if (!is.null(true_labels)) {
        ari_init <- mclust::adjustedRandIndex(cluster_est_now, true_labels)
        cat(sprintf("Initial Clustering Accuracy (ARI): %.4f\n", ari_init))
    }
    # Start with all features selected for the first reward assessment

    for (iternum in 1:n_iter) {
        cat(sprintf("\n--- Iteration %d (Thompson Sampling) ---\n", iternum))

        # =========================================================
        # 2. SELECTION BLOCK
        # =========================================================
        # transfomration step
        res_isee <- ISEE_residual_lasso(X, cluster_est_now, K)
        X_tilde <- res_isee$X_tilde
        Omega_diag_hat <- res_isee$Omega_diag
        # -------------------------------------------------------
        # A. Reward Step
        # -------------------------------------------------------
        # Subset X_tilde based on CURRENT selection indices (S_hat_now)
        X_tilde_sub <- X_tilde[S_hat_now, , drop = FALSE]

        # Run Greedy Screening (Permutation Test) on the subset
        # rewards_sub is a LOGICAL vector of length length(S_hat_now)
        rewards_sub <- select_greedily(X_tilde_sub, cluster_est_now, fdr_level = fdr_level, n_perms = n_perms)

        # -------------------------------------------------------
        # B. Update Step
        # -------------------------------------------------------
        # Update Alpha/Beta for SELECTED features

        # Identify indices in full vector corresponding to S_hat_now
        current_indices <- S_hat_now

        # Map logical rewards back to update values
        alpha_vec[current_indices] <- alpha_vec[current_indices] + as.numeric(rewards_sub)
        beta_vec[current_indices] <- beta_vec[current_indices] + (1 - as.numeric(rewards_sub))

        cat(sprintf("Rewarded arms: %d / %d selected in this subset\n", sum(rewards_sub), length(current_indices)))

        # --- Logging Top 20 Features among current selection ---
        post_mean_temp <- alpha_vec[current_indices] / (alpha_vec[current_indices] + beta_vec[current_indices])
        n_show <- min(20, length(current_indices))
        top_indices_local <- order(post_mean_temp, decreasing = TRUE)[1:n_show]
        top_probs <- post_mean_temp[top_indices_local]
        top_indices_global <- current_indices[top_indices_local]

        cat(sprintf("Top %d Features in selected subset (Posterior Mean):\n", n_show))
        print(setNames(top_probs, top_indices_global))
        # -------------------------------

        # Store rewards history (creating full vector for logging)
        rewards_full <- rep(0, p)
        rewards_full[current_indices] <- as.numeric(rewards_sub)
        reward_history[[iternum]] <- rewards_full

        # -------------------------------------------------------
        # C. Choose Step (Thompson Sampling)
        # -------------------------------------------------------
        # Sample theta ~ Beta(alpha, beta) for ALL features
        theta_sample <- rbeta(p, alpha_vec, beta_vec)

        # Select features > cutoff (Logical Vector)
        current_selection_logical <- theta_sample > cutoff

        # Convert to Index Vector for next iteration's S_hat
        S_hat_next <- which(current_selection_logical)
        n_selected <- length(S_hat_next)

        cat(sprintf("Arms pulled (Features selected for next step): %d / %d\n", n_selected, p))
        selected_history[[iternum]] <- current_selection_logical

        # Handle empty selection
        if (n_selected == 0) {
            warning("No features selected. Keeping previous selection for clustering.")
            S_hat_next <- S_hat_now
            current_selection_logical <- rep(FALSE, p)
            current_selection_logical[S_hat_now] <- TRUE
        }

        # =========================================================
        # 3. CLUSTERING BLOCK
        # =========================================================
        # Run Clustering on the NEW logical selection
        # clustering_block_knowncov uses X_tilde and logical vector

        # Clustering Block returns list(cluster, value)
        # 3. Covariance Calculation (Crucial Step)
        # Calculate sample covariance of original x using selected_features

        # Demean per cluster to remove cluster effect (Pooled Covariance estimation)
        x_sub <- X[S_hat_next, , drop = FALSE]
        X_tilde_sub_new <- X_tilde[S_hat_next, , drop = FALSE]
        # Estimate Sigma_hat on s_hat
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

        # Store Objective Value
        obj_vec[iternum] <- obj_val
        cat(sprintf("SDP Objective Value: %.4f\n", obj_val))

        # Compute Rand Index
        ri <- adjustedRandIndex(cluster_est_new, cluster_est_now)
        rand_vec[iternum] <- ri
        cat(sprintf("Adjusted Rand Index change (prev vs now): %.4f\n", ri))

        if (!is.null(true_labels)) {
            ari_true <- mclust::adjustedRandIndex(cluster_est_new, true_labels)
            cat(sprintf("Iteration %d Accuracy (ARI vs True): %.4f\n", iternum, ari_true))
        }

        # Prepare for next iteration
        cluster_est_now <- cluster_est_new
        S_hat_now <- S_hat_next
    }

    # Final Selection based on Posterior Mean > 0.5
    posterior_mean <- alpha_vec / (alpha_vec + beta_vec)
    final_selection <- posterior_mean > 0.5

    cat(sprintf("\n--- Final Selection ---\n"))
    cat(sprintf("Features selected: %d / %d\n", sum(final_selection), p))

    return(list(
        cluster = cluster_est_now,
        selected = final_selection,
        alpha = alpha_vec,
        beta = beta_vec,
        objective = obj_vec
    ))
}
