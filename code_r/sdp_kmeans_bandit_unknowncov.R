# Thompson Sampling K-Means (Bandit Approach) - Unknown Covariance

source("sdp_kmeans.R")
source("utils.R")
source("clustering_block_unknowncov.R")
source("selection_block_greedy_screening.R")
source("cluster_spectral.R")
source("ISEE_bicluster.R") 
source("get_cov_small.R") # Needed for clustering block unknown
# select_variable_ISEE_noisy is replaced by Bandit logic

if (!require(mclust)) install.packages("mclust")
if (!require(CVXR)) install.packages("CVXR")
if (!require(MASS)) install.packages("MASS")
library(mclust)
library(CVXR)
library(MASS)

#' Thompson Sampling K-Means Algorithm (Unknown Covariance)
#' 
#' Implements the block coordinate ascent with Thompson Sampling for feature selection
#' in the unknown covariance setting using ISEE estimation.
#' 
#' @param X Data matrix (p x n)
#' @param K Number of clusters
#' @param n_iter Number of iterations (default 10)
#' @param C Confidence parameter for threshold (default 0.5)
#' @param FDR_level False Discovery Rate level for reward step (default 0.4)
#' @param n_perms Number of permutations for reward step (default 100)
#' @return List containing cluster assignments, selected features, and metrics
#' @export
sdp_kmeans_bandit_unknowncov <- function(X, K, n_iter = 10, C = 0.5, FDR_level = 0.4, n_perms = 10000) {
  
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
  cutoff <- log(1/C) / log((C+1)/C)
  
  # Metrics
  rand_vec <- numeric(n_iter)
  
  # =========================================================
  # 1. INITIALIZATION BLOCK
  # =========================================================
  # Spectral Clustering on all features
  cat("Running initial spectral clustering...\n")
  cluster_est_now <- cluster_spectral(X, K) 
  
  # Start with all features selected for the first reward assessment
  S_hat_now <- 1:p # S_hat is Index Vector
  
  for (iternum in 1:n_iter) {
    cat(sprintf("\n--- Iteration %d (Thompson Sampling Unknown Cov) ---\n", iternum))
    
    # =========================================================
    # 2. ISEE ESTIMATION BLOCK
    # =========================================================
    # Estimate Mean and Noise matrices using current clustering
    # This replaces simple X_tilde calculation for known cov
    res_isee <- ISEE_bicluster(X, cluster_est_now)
    mean_mat <- res_isee$mean_mat
    noise_mat <- res_isee$noise_mat
    
    # Calculate X_tilde as Mean + Noise (Innovated Data)
    X_tilde <- mean_mat + noise_mat
    
    # =========================================================
    # 3. SELECTION BLOCK
    # =========================================================
    
    # -------------------------------------------------------
    # A. Reward Step
    # -------------------------------------------------------
    # Subset X_tilde based on CURRENT selection indices (S_hat_now)
    X_tilde_sub <- X_tilde[S_hat_now, , drop = FALSE]
    
    # Run Greedy Screening (Permutation Test) on the subset
    rewards_sub <- selection_block_greedy_screening(X_tilde_sub, cluster_est_now, fdr_level = FDR_level, n_perms = n_perms)
    # rewards_sub is logical vector
    
    # -------------------------------------------------------
    # B. Update Step
    # -------------------------------------------------------
    # Update Alpha/Beta for SELECTED features
    current_indices <- S_hat_now
    
    alpha_vec[current_indices] <- alpha_vec[current_indices] + as.numeric(rewards_sub)
    beta_vec[current_indices] <- beta_vec[current_indices] + (1 - as.numeric(rewards_sub))
    
    cat(sprintf("Rewarded arms: %d / %d selected in this subset\n", sum(rewards_sub), length(current_indices)))

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
    
    # Handle empty selection
    if (n_selected == 0) {
        warning("No features selected. Keeping previous selection for clustering.")
        S_hat_next <- S_hat_now
        current_selection_logical <- rep(FALSE, p)
        current_selection_logical[S_hat_now] <- TRUE
    }
    
    # =========================================================
    # 4. CLUSTERING BLOCK
    # =========================================================
    # Run Clustering on the NEW logical selection
    # clustering_block_unknowncov takes raw X, mean, noise, and logical selection s_hat
    # It internally calls get_cov_small on raw X subset to estimate Sigma
    
    # Note: clustering_block_unknowncov expects s_hat. If logical is passed, R subsets correctly.
    cluster_est_new <- run_clustering_block_unknowncov(X, K, mean_mat, noise_mat, cluster_est_now, current_selection_logical)
    
    # Compute Rand Index
    ri <- adjustedRandIndex(cluster_est_new, cluster_est_now)
    rand_vec[iternum] <- ri
    cat(sprintf("Adjusted Rand Index change: %.4f\n", ri))
    
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
    beta = beta_vec
  ))
}
