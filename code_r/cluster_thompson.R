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

#' Block Coordinate Optimization with Thompson Sampling (Known Covariance)
#'
#' Implements a block coordinate ascent algorithm for high-dimensional clustering,
#' utilizing Thompson Sampling to efficiently explore and select discriminative features.
#' This function alternates between (1) estimating feature relevance via a multi-armed bandit
#' framework (Thompson Sampling) and (2) updating cluster assignments using a
#' Semidefinite Programming (SDP) relaxation of K-means on the selected subset of features.
#'
#' @param X Data matrix of dimension (p x n), where p is the number of features and n is the number of samples.
#' @param K Integer. The expected number of clusters.
#' @param n_iter Integer. Number of block coordinate optimization iterations to perform (default: 500).
#' @param C Numeric. Confidence parameter used to define the posterior inclusion probability threshold (default: 0.5).
#' @param n_perms Integer. Number of permutations for the greedy screening permutation test (default: 300).
#' @param p_val_threshold Numeric. False discovery rate / p-value threshold for feature screening (default: 0.01).
#' @param n_step_admm Integer. Maximum number of ADMM iterations for the SDP clustering step (default: 2000).
#' @param covariance Matrix (p x p). Known or estimated covariance structure between the p features.
#'        If NULL, the feature space is assumed to be isotropic (Identity covariance).
#' @param true_cluster Vector of integers. Ground truth cluster labels (length n) used solely for
#'        computing interim and final clustering accuracy metrics. If NULL, accuracy is not tracked.
#'
#' @return A list containing:
#'   \itemize{
#'     \item \code{cluster}: Final estimated cluster assignments for the n samples.
#'     \item \code{selected}: Logical vector (length p) indicating which features are ultimately deemed discriminative.
#'     \item \code{alpha}: Final Beta distribution shape parameter alpha for all p features.
#'     \item \code{beta}: Final Beta distribution shape parameter beta for all p features.
#'     \item \code{objective}: History of the SDP clustering objective function values across iterations.
#'     \item \code{acc_history}: (Optional) History of clustering accuracy across iterations if true_cluster is provided.
#'   }
#' @export
cluster_thompson <- function(X, K, n_iter = 500, C = 0.5, n_perms = 1000, p_val_threshold = 0.01, n_step_admm = 4000, covariance = NULL, true_cluster = NULL) {
  # Requirement checks: Ensure optimization parameters are valid.
  if (!is.numeric(n_step_admm) || n_step_admm <= 0 || n_step_admm %% 1 != 0) stop("n_step_admm must be a positive integer.")
  if (!is.matrix(X)) stop("X must be a matrix")

  p <- nrow(X)
  n <- ncol(X)

  # ---------------------------------------------------------------------------
  # ALGORITHM INITIALIZATION: Thompson Sampling Priors
  # ---------------------------------------------------------------------------
  # Objective: Treat feature selection as a Multi-Armed Bandit problem.
  # Each feature 'j' is an arm with an unknown probability of being "informative".
  # We model the belief of each feature's usefulness using a Beta(alpha, beta) distribution.
  # Initializing alpha = 1, beta = 1 translates to a Uniform(0,1) non-informative prior.
  alpha_vec <- rep(1, p)
  beta_vec <- rep(1, p)

  # ---------------------------------------------------------------------------
  # THRESHOLD CALCULATION: Posterior Inclusion Probability Cutoff
  # ---------------------------------------------------------------------------
  # Rationale: Features are selected for clustering if their sampled belief (theta)
  # exceeds a theoretical boundary. The threshold converts the confidence parameter 'C'
  # into a probability cutoff space mapping to log-odds.
  if (C <= 0 || C >= 1) {
    warning("C should be in (0, 1). Using default 0.5.")
    C <- 0.5
  }
  cutoff <- log(1 / C) / log((C + 1) / C)

  # Initialize vectors to track algorithmic progress and convergence metrics across iterations
  rand_vec <- numeric(n_iter) # Tracks stability (Adjusted Rand Index) between consecutive clusterings
  obj_vec <- numeric(n_iter) # Tracks the mathematical SDP objective convergence
  selected_history <- list() # Tracks subset evolution
  reward_history <- list() # Tracks feature usefulness evaluation history

  if (!is.null(true_cluster)) {
    acc_vec <- numeric(n_iter) # Tracks absolute objective accuracy against ground truth
  }

  # ---------------------------------------------------------------------------
  # COVARIANCE TRANSFORMATION: Whitening the Data Space
  # ---------------------------------------------------------------------------
  # Rationale: Standard KMeans assumes spherical clusters (isotropic variance).
  # If features are highly correlated, distance metrics are distorted.
  # By pre-multiplying X by the inverse of the covariance matrix (Sigma^{-1} X),
  # we calculate the Mahalanobis equivalents, transforming the space into spherical
  # decorrelated coordinates, allowing standard clustering assumptions to hold.
  if (is.null(covariance)) {
    X_tilde <- X
  } else {
    # Efficiently computes Sigma^{-1} * X without explicitly calculating the dense inverse matrix
    X_tilde <- solve(covariance, X)
  }

  # =========================================================================
  # BLOCK 1: WARM START / INITIALIZATION
  # =========================================================================

  # Step 1.1: Draw an initial feature subset from the naive prior.
  cat("Running initial sampling from prior (Beta(1,1))...\n")
  # Expected result: Approx 50% of features selected if cutoff ~ 0.5
  theta_init <- rbeta(p, alpha_vec, beta_vec)
  S_hat_now <- theta_init > cutoff

  # Fallback: In high noise, the sampling might return an empty set.
  # We require at least some data to perform an initial naive cluster.
  if (sum(S_hat_now) == 0) {
    warning("Initialization selected 0 features. Forcing random selection of 10 features.")
    S_hat_now <- rep(FALSE, p)
    S_hat_now[sample(1:p, 10, replace = FALSE)] <- TRUE
  }

  cat(sprintf("Initial subset size: %d. Running initial clustering on subset...\n", sum(S_hat_now)))

  # Step 1.2: Obtain an initial coarse clustering using ADMM optimization over the SDP relaxation.
  # Rationale: We need initial labels (cluster_est_now) to evaluate whether features separate these labels well.
  clustering_result_init <- run_clustering_block_knowncov(X_tilde, S_hat_now, K, cluster_est_prev = numeric(n), covariance = covariance, max_iter = n_step_admm)
  cluster_est_now <- clustering_result_init$cluster


  # =========================================================================
  # BLOCK 2: MAIN ALTERNATING OPTIMIZATION LOOP
  # =========================================================================
  # Rationale: Iteratively refine both the feature selection and the cluster labels.
  # Fixing cluster labels -> Evaluate feature relevance -> Update subset -> Refine clustering labels on new subset.
  for (iternum in 1:n_iter) {
    cat(sprintf("\n--- Iteration %d (Thompson Sampling) ---\n", iternum))

    # -------------------------------------------------------
    # PHASE A: FEATURE REWARD EVALUATION (The "Bandit" Pull)
    # -------------------------------------------------------
    # Rationale: We only evaluate features currently in our subset (S_hat_now).
    # We apply a permutation test (Greedy Screening) to determine if a feature
    # statistically distinguishes the *current* estimated clusters.

    # Isolate relevant subset of transformed data
    X_tilde_sub <- X_tilde[S_hat_now, , drop = FALSE]

    # Evaluate feature performance.
    # The selection block simulates the null hypothesis (features do not drive the clustering)
    # by permuting the cluster labels and measuring test statistic differentials.
    # Output is a logical vector of accepted features (TRUE = Good discriminator).
    rewards_sub <- reward_thompson(
      X_tilde_sub,
      cluster_est_now,
      fdr_level = NULL, # do not use FDR control, to ensure independence across feature indices
      n_perms = n_perms,
      p_val_threshold = p_val_threshold
    )

    # -------------------------------------------------------
    # PHASE B: POSTERIOR UPDATE
    # -------------------------------------------------------
    # Rationale: Incorporate the evidence from Phase A into our Beta posteriors via Bayes' Rule.
    # Beta distribution naturally updates by adding successes to alpha and failures to beta.
    # Non-evaluated features remain at their current posterior distribution.

    # Update alpha (successes) and beta (failures) based on logical reward outcomes
    alpha_vec[S_hat_now] <- alpha_vec[S_hat_now] + as.numeric(rewards_sub)
    beta_vec[S_hat_now] <- beta_vec[S_hat_now] + (1 - as.numeric(rewards_sub))

    cat(sprintf("Rewarded arms: %d / %d selected in this subset\n", sum(rewards_sub), sum(S_hat_now)))

    # Console Logging: Track the evolution of the strongest signals
    post_mean_temp <- alpha_vec / (alpha_vec + beta_vec)
    top10_indices <- order(post_mean_temp, decreasing = TRUE)[1:10]
    top10_probs <- post_mean_temp[top10_indices]
    cat("Top 10 Features (Posterior Mean):\n")
    print(setNames(top10_probs, top10_indices))

    # Maintain historical log array mapping rewards to the global index scope (p)
    rewards_full <- rep(0, p)
    rewards_full[S_hat_now] <- as.numeric(rewards_sub)
    reward_history[[iternum]] <- rewards_full

    # -------------------------------------------------------
    # PHASE C: THOMPSON SAMPLING (Select Next Feature Subset)
    # -------------------------------------------------------
    # Rationale: Instead of strictly choosing the top features (which is greedy and risks local minima),
    # Thompson Sampling draws a random belief from each feature's updated posterior.
    # This inherently balances:
    # 1. Exploitation: Features with high alpha (proven useful) have posteriors shifted towards 1.
    # 2. Exploration: Features rarely tested have wide posteriors (high variance), giving them a chance to be sampled.

    # Draw from posterior distributions simultaneously across all p features.
    theta_sample <- rbeta(p, alpha_vec, beta_vec)

    # Filter features where the sampled belief exceeds the theoretical confidence threshold
    S_hat_next <- theta_sample > cutoff
    n_selected <- sum(S_hat_next)

    # Convert to index list targeting the upcoming clustering iteration
    # <- which(current_selection_logical)

    cat(sprintf("Arms pulled (Features selected for next step): %d / %d\n", n_selected, p))
    selected_history[[iternum]] <- S_hat_next

    # Edge Case: Extreme variance or severe noise models may occasionally cull all features.
    # Revert to the last known stable state to prevent clustering failure matrices.
    if (n_selected == 0) {
      warning("No features selected. Keeping previous selection for clustering.")
      S_hat_next <- S_hat_now
    }

    # -------------------------------------------------------
    # PHASE D: SDP CLUSTERING REFINEMENT
    # -------------------------------------------------------
    # Rationale: Having defined a new subset of features that are statistically likely to isolate the clusters,
    # re-estimate the K cluster centroids using a Semidefinite Programming (SDP) relaxation of the KMeans objective.
    # SDP approaches are generally highly robust to non-convex local trap geometries typical in high dimensions.

    # Execute the clustering algorithm strictly over the active logical feature subset map
    clustering_result <- run_clustering_block_knowncov(X_tilde, S_hat_next, K, cluster_est_now, covariance = covariance, max_iter = n_step_admm)

    cluster_est_new <- clustering_result$cluster
    obj_val <- clustering_result$value

    # Evaluate algorithmic clustering convergence stability tracking metrics
    obj_vec[iternum] <- obj_val
    cat(sprintf("SDP Objective Value: %.4f\n", obj_val))

    # The Adjusted Rand Index (ARI) evaluates how similarly the newly determined clusters match the previous iteration.
    # When ARI approaches 1, the labels have stabilized, indicating iterative convergence.
    ri <- adjustedRandIndex(cluster_est_new, cluster_est_now)
    rand_vec[iternum] <- ri
    cat(sprintf("Adjusted Rand Index change: %.4f\n", ri))

    # If an oracle ground-truth was injected, calculate strict predictive classification accuracy.
    if (!is.null(true_cluster)) {
      acc <- get_cluster_acc(cluster_est_new, true_cluster)
      acc_vec[iternum] <- acc
      cat(sprintf("Clustering Accuracy: %.4f\n", acc))
    }

    # Promote current iteration artifacts to become the context of the next iteration limit state.
    cluster_est_now <- cluster_est_new
    S_hat_now <- S_hat_next
  }

  # =========================================================================
  # BLOCK 3: FINAL SELECTION AND ARTIFACT COMPILATION
  # =========================================================================

  # Rationale: Final discriminative feature classification relies purely on the
  # Expectation (Mean) of the final aggregate posterior Beta distributions, eliminating sampling variance noise.
  # Under Beta(alpha, beta), E[X] = alpha / (alpha + beta)
  posterior_mean <- alpha_vec / (alpha_vec + beta_vec)

  # Apply identical decision theoretical threshold against the expected value.
  final_selection <- posterior_mean > cutoff
  cat(sprintf("\n--- Final Selection ---\n"))
  cat(sprintf("Features selected: %d / %d\n", sum(final_selection), p))

  # Compile outputs mimicking standard statistical function return object formatting.
  out <- list(
    cluster = cluster_est_now,
    selected = S_hat_now,
    alpha = alpha_vec,
    beta = beta_vec,
    objective = obj_vec
  )

  # Attach oracle trajectory accuracy tracking if context was supplied.
  if (!is.null(true_cluster)) {
    out$acc_history <- acc_vec
  }

  return(out)
}
