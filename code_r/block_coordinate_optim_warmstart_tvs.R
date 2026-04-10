# Block Coordinate Optimization with Warmstart TVS (Greedy -> TVS)

library(kernlab)
library(mclust)

# Requires sourcing of:
# source("selection_block_greedy_screening.R")
# source("clustering_block_knowncov.R")
# source("ISEE_residual_lasso.R")
# source("cluster_greedy_ISEE.R") # for get_permutation_fdr_threshold
# source("ESSC.R")
# source("utils.R")

#' Greedy Warmstart TVS
#'
#' Phase 1: Runs Greedy (Permutation FDR) for `n_iter_greedy` iterations.
#' Phase 2: Freezes X_tilde and cluster estimates.
#' Phase 3: Runs TVS for `n_iter_tvs` iterations on the frozen environment.
#'
#' @export
block_coordinate_optim_warmstart_tvs <- function(X, K,
                                                 n_iter_greedy = 100,
                                                 n_iter_tvs = 50,
                                                 n_perms_greedy = 20,
                                                 fdr_target_greedy = 0.4,
                                                 C = 0.5,
                                                 p_val_threshold = 0.01,
                                                 true_labels = NULL) {
    if (!is.matrix(X)) stop("X must be a matrix")
    p <- nrow(X)
    n <- ncol(X)

    start_time <- Sys.time()

    # ---------------------------------------------------------
    # Phase 1: Greedy Initialization & Stabilization
    # ---------------------------------------------------------
    cat("\n=======================================================\n")
    cat("           PHASE 1: GREEDY STABILIZATION\n")
    cat("=======================================================\n")

    cat("Running initial clustering (ESSC)...\n")
    cluster_est_now <- ESSC(X, K)

    if (!is.null(true_labels)) {
        acc_init <- get_cluster_acc(cluster_est_now, true_labels)
        cat(sprintf("Initial Clustering Accuracy: %.4f\n", acc_init))
    }

    greedy_ari_history <- numeric(n_iter_greedy)
    greedy_acc_history <- numeric(n_iter_greedy)
    S_hat_now <- 1:p # Initially assume all

    # Variables to track diagnostic differences
    X_tilde <- matrix(0, p, n)
    X_tilde_prev <- matrix(0, p, n)
    ari_at_100 <- NA
    overlap_at_100 <- NA
    x_tilde_diff_99_100 <- NA

    for (iter_g in 1:n_iter_greedy) {
        cat(sprintf("\n--- Phase 1 Iteration %d / %d ---\n", iter_g, n_iter_greedy))

        # 1. ISEE Update
        cat("Running ISEE...\n")
        res_isee <- ISEE_residual_lasso(X, cluster_est_now, K)
        X_tilde_prev <- X_tilde
        X_tilde <- res_isee$X_tilde
        Omega_diag_hat <- res_isee$Omega_diag

        # Diagnostic: Norm difference at boundary
        if (iter_g == n_iter_greedy) {
            x_tilde_diff_99_100 <- norm(X_tilde - X_tilde_prev, type = "F")
        }

        # 2. Selection Update (Permutation FDR - same as sim07)
        means_mat <- matrix(0, p, K)
        for (c in 1:K) {
            means_mat[, c] <- rowMeans(X_tilde[, cluster_est_now == c, drop = FALSE])
        }
        n1 <- sum(cluster_est_now == 1)
        n2 <- sum(cluster_est_now == 2)
        abs_diff <- abs(means_mat[, 1] - means_mat[, 2]) / sqrt(pmax(Omega_diag_hat, 1e-8)) * sqrt(n1 * n2 / n)

        cat(sprintf("Estimating Delta Threshold via %d permutations...\n", n_perms_greedy))
        perm_res <- get_permutation_fdr_threshold(abs_diff, X_tilde, cluster_est_now, Omega_diag_hat, n_perms_greedy, fdr_target_greedy)

        S_hat_now <- perm_res$s_hat
        if (length(S_hat_now) == 0) {
            cat("WARNING: No features selected. Using Top K features fallback.\n")
            S_hat_now <- order(abs_diff, decreasing = TRUE)[1:K]
        }
        cat(sprintf("Features Selected: %d (Est FDR: %.4f)\n", length(S_hat_now), perm_res$est_fdr))

        # 3. Clustering Update
        cat("Running Clustering Block...\n")
        Sigma_hat_small <- get_cov_small(X, cluster_est_now, S_hat_now)
        Sigma_full <- diag(1, p)
        if (length(S_hat_now) > 0) {
            Sigma_full[S_hat_now, S_hat_now] <- Sigma_hat_small
        }

        res_cluster <- run_clustering_block_knowncov(X_tilde, S_hat_now, K, cluster_est_now, covariance = Sigma_full, max_iter = 4000)
        cluster_est_new <- res_cluster$cluster

        ari_now <- mclust::adjustedRandIndex(cluster_est_new, cluster_est_now)
        greedy_ari_history[iter_g] <- ari_now
        cat(sprintf("Stability Check (ARI vs Prev): %.4f\n", ari_now))

        if (!is.null(true_labels)) {
            acc1 <- sum(cluster_est_new == true_labels) / n
            acc2 <- sum(cluster_est_new != true_labels) / n
            accuracy_now <- max(acc1, acc2)
            greedy_acc_history[iter_g] <- accuracy_now
            cat(sprintf("Clustering Accuracy: %.4f\n", accuracy_now))
        }

        cluster_est_now <- cluster_est_new

        if (iter_g == n_iter_greedy) {
            ari_at_100 <- ari_now
            # Calculate true overlap if ground truth is available? Wait, no true support is passed.
            # We'll calculate overlap with ground truth externally in the driver.R script
            # that's where the true support is known. But we can save the subset.
        }
    }

    # ---------------------------------------------------------
    # Phase 2: Freeze State
    # ---------------------------------------------------------
    cat("\n=======================================================\n")
    cat("           PHASE 2: FREEZING ENVIRONMENT\n")
    cat("=======================================================\n")
    cat("Freezing X_tilde.\n")
    cat("No further ISEE updates will run.\n")

    frozen_X_tilde <- X_tilde

    # ---------------------------------------------------------
    # Phase 3: TVS on Stationary Environment
    # ---------------------------------------------------------
    cat("\n=======================================================\n")
    cat("           PHASE 3: TVS EXACT BOUNDARY SEARCH\n")
    cat("=======================================================\n")

    # Initialization for Bandit
    alpha_vec <- rep(1, p)
    beta_vec <- rep(1, p)

    if (C <= 0 || C >= 1) C <- 0.5
    cutoff <- log(1 / C) / log((C + 1) / C)

    tvs_reward_trajectory <- list()
    tvs_selected_history <- list()
    tvs_turnover_rate <- numeric(n_iter_tvs)
    tvs_ari_history <- numeric(n_iter_tvs)
    tvs_acc_history <- numeric(n_iter_tvs)

    # "Initialize TVS using: Current selected feature set from Greedy"
    S_hat_tvs <- S_hat_now

    for (iter_t in 1:n_iter_tvs) {
        cat(sprintf("\n--- Phase 3 Iteration %d / %d ---\n", iter_t, n_iter_tvs))

        # ISEE disabled in sim11
        # X_tilde is fixed after greedy warm start

        # 1. Reward Step (on frozen environment)
        X_tilde_sub <- frozen_X_tilde[S_hat_tvs, , drop = FALSE]

        # Reward calculation: standard Greedy Screening (as implemented in TVS)
        rewards_sub <- select_greedily(X_tilde_sub, cluster_est_now, fdr_level = NULL, n_perms = 10000, p_val_threshold = p_val_threshold)

        # 2. Update Step
        alpha_vec[S_hat_tvs] <- alpha_vec[S_hat_tvs] + as.numeric(rewards_sub)
        beta_vec[S_hat_tvs] <- beta_vec[S_hat_tvs] + (1 - as.numeric(rewards_sub))

        cat(sprintf("Rewarded arms: %d / %d selected in this subset\n", sum(rewards_sub), length(S_hat_tvs)))

        rewards_full <- rep(0, p)
        rewards_full[S_hat_tvs] <- as.numeric(rewards_sub)
        tvs_reward_trajectory[[iter_t]] <- rewards_full

        # --- Enhanced Logging: Top 10 Rewarded vs Un-rewarded Arms ---
        post_mean_all <- alpha_vec / (alpha_vec + beta_vec)

        # Rewarded Arms
        rewarded_indices <- S_hat_tvs[which(rewards_sub)]
        if (length(rewarded_indices) > 0) {
            n_show_r <- min(10, length(rewarded_indices))
            rewarded_probs <- post_mean_all[rewarded_indices]
            top_r_idx <- order(rewarded_probs, decreasing = TRUE)[1:n_show_r]
            cat(sprintf("Top %d REWARDED Arms (Posterior Mean):\n", n_show_r))
            print(setNames(rewarded_probs[top_r_idx], rewarded_indices[top_r_idx]))
        }

        # Un-rewarded (but selected) Arms
        unrewarded_indices <- S_hat_tvs[which(!rewards_sub)]
        if (length(unrewarded_indices) > 0) {
            n_show_u <- min(10, length(unrewarded_indices))
            unrewarded_probs <- post_mean_all[unrewarded_indices]
            top_u_idx <- order(unrewarded_probs, decreasing = TRUE)[1:n_show_u]
            cat(sprintf("Top %d UN-REWARDED (Selected) Arms (Posterior Mean):\n", n_show_u))
            print(setNames(unrewarded_probs[top_u_idx], unrewarded_indices[top_u_idx]))
        }
        # 3. Choose Step (Thompson Sampling)
        theta_sample <- rbeta(p, alpha_vec, beta_vec)
        current_selection_logical <- theta_sample > cutoff
        S_hat_next <- which(current_selection_logical)

        cat(sprintf("Arms pulled (Features selected for next step): %d / %d\n", length(S_hat_next), p))
        tvs_selected_history[[iter_t]] <- current_selection_logical

        # Calculate Turnover Rate (new features + dropped features) / total features
        turnover <- sum(!(S_hat_next %in% S_hat_tvs)) + sum(!(S_hat_tvs %in% S_hat_next))
        avg_size <- (length(S_hat_tvs) + length(S_hat_next)) / 2
        tvs_turnover_rate[iter_t] <- if (avg_size > 0) turnover / (2 * avg_size) else 0

        if (length(S_hat_next) == 0) {
            warning("No features selected. Keeping previous selection.")
            S_hat_next <- S_hat_tvs
        }

        S_hat_tvs <- S_hat_next

        # 4. Clustering Update
        cat("Running Clustering Block...\n")
        Sigma_hat_small <- get_cov_small(X, cluster_est_now, S_hat_tvs)
        Sigma_full <- diag(1, p)
        if (length(S_hat_tvs) > 0) {
            Sigma_full[S_hat_tvs, S_hat_tvs] <- Sigma_hat_small
        }

        res_cluster <- run_clustering_block_knowncov(frozen_X_tilde, S_hat_tvs, K, cluster_est_now, covariance = Sigma_full, max_iter = 4000)
        cluster_est_new <- res_cluster$cluster

        ari_now <- mclust::adjustedRandIndex(cluster_est_new, cluster_est_now)
        tvs_ari_history[iter_t] <- ari_now
        cat(sprintf("Stability Check (ARI vs Prev): %.4f\n", ari_now))

        if (!is.null(true_labels)) {
            acc1 <- sum(cluster_est_new == true_labels) / n
            acc2 <- sum(cluster_est_new != true_labels) / n
            accuracy_now <- max(acc1, acc2)
            tvs_acc_history[iter_t] <- accuracy_now
            cat(sprintf("Clustering Accuracy: %.4f\n", accuracy_now))
        }

        cluster_est_now <- cluster_est_new
    }

    # Final Selection
    posterior_mean <- alpha_vec / (alpha_vec + beta_vec)
    final_selection <- posterior_mean > cutoff
    Sigma_full <- diag(1, p)
    if (length(final_selection) > 0) {
        Sigma_full[final_selection, final_selection] <- get_cov_small(X, cluster_est_now, final_selection)
    }
    res_cluster <- run_clustering_block_knowncov(frozen_X_tilde, final_selection, K, cluster_est_now, covariance = Sigma_full, max_iter = 4000)
    cluster_est_final <- res_cluster$cluster
    acc_final <- get_cluster_acc(cluster_est_final, true_labels)
    cat(sprintf("\n--- Final Selection (TVS) ---\n"))
    cat(sprintf("Features selected: %d / %d\n", sum(final_selection), p))
    cat(sprintf("Final Clustering Accuracy: %.4f\n", acc_final))

    total_time <- difftime(Sys.time(), start_time, units = "secs")

    return(list(
        cluster = cluster_est_final,
        acc_final = acc_final,
        s_hat = which(final_selection),
        selected = final_selection,
        alpha = alpha_vec,
        beta = beta_vec,
        time = total_time,
        greedy_s_hat_100 = S_hat_now,
        ari_at_100 = ari_at_100,
        x_tilde_diff_99_100 = x_tilde_diff_99_100,
        tvs_reward_trajectory = tvs_reward_trajectory,
        tvs_turnover_rate = tvs_turnover_rate,
        greedy_acc_history = greedy_acc_history,
        tvs_acc_history = tvs_acc_history
    ))
}
