# Block Coordinate Optimization with Imported TVS (Precomputed Greedy -> TVS)

library(kernlab)
library(mclust)

# Requires sourcing of:
# source("selection_block_greedy_screening.R")
# source("clustering_block_knowncov.R")
# source("utils.R")

#' Greedy Import TVS
#'
#' Phase 1: Loads precomputed Greedy (Permutation FDR) results from sim07.
#' Phase 2: Freezes X_tilde and cluster estimates.
#' Phase 3: Runs TVS for `n_iter_tvs` iterations on the frozen environment.
#'
#' @export
block_coordinate_optim_import_tvs <- function(X, K,
                                              sim_id,
                                              fdr_level,
                                              n_iter_tvs = 50,
                                              C = 0.5,
                                              p_val_threshold = 0.01,
                                              true_labels = NULL) {
    if (!is.matrix(X)) stop("X must be a matrix")
    p <- nrow(X)
    n <- ncol(X)

    start_time <- Sys.time()

    # ---------------------------------------------------------
    # Phase 1 & 2: Load Precomputed Greedy Output & Freeze State
    # ---------------------------------------------------------
    cat("\n=======================================================\n")
    cat("    PHASE 1 & 2: IMPORT PRECOMPUTED GREEDY STATE\n")
    cat("=======================================================\n")

    fdr_str <- gsub("\\.", "p", as.character(fdr_level))

    # Check possible paths depending on where the script is executed
    path1 <- sprintf("./sim07_SAM_sep3/results/sim_id%d_fdr%s.rds", sim_id, fdr_str)
    path2 <- sprintf("/home1/jongminm/sparse_kmeans_project/simulations/sim07_SAM_sep3/results/sim_id%d_fdr%s.rds", sim_id, fdr_str)

    file_path <- NULL
    if (file.exists(path1)) {
        file_path <- path1
    } else if (file.exists(path2)) {
        file_path <- path2
    } else {
        stop(sprintf("Cannot find precomputed greedy output for sim_id %d and fdr %s", sim_id, fdr_str))
    }

    ### IMPORTED GREEDY RESULTS
    cat(sprintf("Loading greedy state from: %s\n", file_path))
    saved_obj <- readRDS(file_path)

    # Extract components from the saved object
    frozen_X_tilde <- saved_obj$res$X_tilde_final
    cluster_est_now <- saved_obj$res$cluster
    S_hat_now <- saved_obj$res$s_hat

    if (!is.null(true_labels)) {
        acc_init <- get_cluster_acc(cluster_est_now, true_labels)
        cat(sprintf("Imported Clustering Accuracy: %.4f\n", acc_init))
    }

    cat(sprintf("Imported Support Size: %d\n", length(S_hat_now)))

    ### ISEE STEP SKIPPED
    cat("Freezing X_tilde.\n")
    cat("No further ISEE updates will run.\n")

    # Metrics that are empty since we skipped greedy iterations
    greedy_ari_history <- NA
    greedy_acc_history <- NA
    ari_at_100 <- NA
    x_tilde_diff_99_100 <- NA

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

    # Initialize TVS using Current selected feature set from imported Greedy
    S_hat_tvs <- S_hat_now

    for (iter_t in 1:n_iter_tvs) {
        cat(sprintf("\n--- Phase 3 Iteration %d / %d ---\n", iter_t, n_iter_tvs))

        # ISEE skipped inside loop since X_tilde is frozen
        # 1. Reward Step (on frozen environment)
        X_tilde_sub <- frozen_X_tilde[S_hat_tvs, , drop = FALSE]

        # Reward calculation: standard Greedy Screening
        rewards_sub <- selection_block_greedy_screening(X_tilde_sub, cluster_est_now, fdr_level = NULL, n_perms = 10000, p_val_threshold = p_val_threshold)

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
        greedy_s_hat_100 = S_hat_now, # Store imported features
        ari_at_100 = ari_at_100, # NA since we skipped greedy
        x_tilde_diff_99_100 = x_tilde_diff_99_100, # NA
        tvs_reward_trajectory = tvs_reward_trajectory,
        tvs_turnover_rate = tvs_turnover_rate,
        greedy_acc_history = greedy_acc_history, # NA
        tvs_acc_history = tvs_acc_history
    ))
}
