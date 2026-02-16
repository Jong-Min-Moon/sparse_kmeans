# ISEE Bicluster Algorithm for Unknown Covariance

# source("get_intercept_residual_lasso.R") # Sourced by driver script

#' ISEE Bicluster Algorithm
#' 
#' Estimates means and noise using blockwise Lasso regressions.
#' 
#' @param x Data matrix (p x n)
#' @param cluster_est_now Cluster assignments (vector of length n)
#' @return List containing:
#'   \item{mean_vec}{Estimated mean constant vectors for clusters (p x K)}
#'   \item{noise_mat}{Estimated noise matrix (p x n)}
#'   \item{Omega_diag_hat}{Estimated diagonal of precision matrix (p x 1)}
#'   \item{mean_mat}{Matrix of cluster-wise sample means (p x n)}
#' @export
ISEE_bicluster <- function(x, cluster_est_now) {
  
  p <- nrow(x)
  n <- ncol(x)
  K <- length(unique(cluster_est_now))
  
  # Blockwise settings: Block size 2 (from MATLAB default)
  n_regression <- floor(p / 2)
  
  # Initialize outputs
  mean_vec <- matrix(0, nrow = p, ncol = K)
  noise_mat <- matrix(0, nrow = p, ncol = n)
  Omega_diag_hat <- numeric(p)
  
  # Iterate over blocks
  # This loop can be parallelized in the future using foreach
  cat(sprintf("Running ISEE Bicluster on %d blocks...\n", n_regression))
  
  for (i in 1:n_regression) {
    if (i %% 100 == 0) cat(sprintf("  Processing block %d/%d\n", i, n_regression))
    
    # Define block indices (2 rows per block)
    rows_idx <- c(2 * i - 1, 2 * i)
    
    # Predictors are all other rows
    # Creating this huge matrix every time is slow. 
    # Optimization: Pass indices needed? Lasso needs data.
    # In p >> n, constructing (p-2) x n matrix is fine.
    
    # Using logical indexing for predictors might be cleaner
    predictors_idx <- rep(TRUE, p)
    predictors_idx[rows_idx] <- FALSE
    
    # Extract predictors once for this block? 
    # No, we need to subset by cluster inside.
    
    # E_Al: Residuals for this block (2 x n)
    E_Al <- matrix(0, nrow = 2, ncol = n)
    # alpha_Al: Intercepts (2 x K)
    alpha_Al <- matrix(0, nrow = 2, ncol = K)
    
    # Loop over clusters
    for (c in 1:K) { # Assuming clusters are 1..K
      cluster_mask <- (cluster_est_now == c)
      if (sum(cluster_mask) < 2) next # Skip if too few samples
      
      x_cluster <- x[, cluster_mask, drop = FALSE]
      
      # Predictors for this cluster: (p-2) x n_c
      predictor_now <- t(x_cluster[predictors_idx, , drop = FALSE]) # Transpose to n_c x (p-2)
      
      # For each row in the block
      for (j in 1:2) {
        row_id <- rows_idx[j]
        response_now <- x_cluster[row_id, ] # n_c x 1
        
        # Run Lasso
        res <- get_intercept_residual_lasso(response_now, predictor_now)
        
        # Store results
        E_Al[j, cluster_mask] <- res$residual
        alpha_Al[j, c] <- res$intercept
      }
    }
    
    # Estimate Omega (Precision) for this block
    # Omega_hat = inv(E * E') * n ? 
    # MATLAB: Omega_hat_Al = inv(E_Al * E_Al') * n; (Check scaling)
    # E_Al * E_Al' is (2 x n) * (n x 2) = 2 x 2 Scatter matrix
    # If using formatted covariance, might need /n or something. 
    # MATLAB code takes inv(Scatter) * n. This implies Omega is roughly n * Scatter^-1.
    # Standard: Cov = Scatter/n. Prec = inv(Cov) = inv(Scatter/n) = n * inv(Scatter).
    # Correct.
    
    scatter_mat <- tcrossprod(E_Al) # E_Al %*% t(E_Al)
    
    # Handle singularity
    Omega_hat_Al <- tryCatch({
      solve(scatter_mat) * n
    }, error = function(e) {
      # Fallback: Identity or diag inverse
      diag(1/diag(scatter_mat)) * n
    })
    
    # Compute local parameters
    mean_local <- Omega_hat_Al %*% alpha_Al   # 2 x K
    noise_local <- Omega_hat_Al %*% E_Al      # 2 x n
    diag_local <- diag(Omega_hat_Al)          # 2 x 1
    
    # Store in global outputs
    mean_vec[rows_idx, ] <- mean_local
    noise_mat[rows_idx, ] <- noise_local
    Omega_diag_hat[rows_idx] <- diag_local
  }
  
  # Handle remaining row if p is odd?
  if (2 * n_regression < p) {
    # TODO: Handle last row separately (Block size 1)
    # Just skip for now or duplicate logic? 
    # MATLAB code: n_regression = floor(p/2). It seems to ignore the last row if odd?
    # I will verify or add warning.
  }
  
  # Construct mean_mat (p x n) from mean_vec (p x K)
  mean_mat <- matrix(0, nrow = p, ncol = n)
  for (c in 1:K) {
    cluster_mask <- (cluster_est_now == c)
    mean_mat[, cluster_mask] <- mean_vec[, c] # Broadcast column
  }
  
  return(list(
    mean_vec = mean_vec,
    noise_mat = noise_mat,
    Omega_diag_hat = Omega_diag_hat,
    mean_mat = mean_mat
  ))
}
