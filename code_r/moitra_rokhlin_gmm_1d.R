#' Estimate Excess Moments for 1D Gaussian Mixture Model
#' 
#' According to Lemma 3.1 of the provided paper, the excess moments X3, X4, X5, X6
#' are defined as polynomials of the central moments M_i such that the result is 
#' independent of adding Gaussian noise.
#'
#' @param x A numeric vector of samples from a 1D Gaussian Mixture
#' @return A list containing the overall mean, variance, and the excess moments X3 through X6.
estimate_excess_moments <- function(x) {
  n <- length(x)
  
  # Overall mean
  mu <- mean(x)
  
  # Center the data to compute central moments (M_i)
  x_c <- x - mu
  
  # Compute sample central moments
  M2 <- mean(x_c^2)
  M3 <- mean(x_c^3)
  M4 <- mean(x_c^4)
  M5 <- mean(x_c^5)
  M6 <- mean(x_c^6)
  
  # Compute excess moments (from Lemma 3.1)
  # For simplicity, X1 = mu and X2 = M2 (variance)
  sigma2 <- M2
  X3 <- M3
  X4 <- M4 - 3 * M2^2
  X5 <- M5 - 10 * M3 * M2
  X6 <- M6 - 15 * M4 * M2 + 30 * M2^3
  
  return(list(
    mu = mu,
    sigma2 = sigma2,
    X3 = X3,
    X4 = X4,
    X5 = X5,
    X6 = X6
  ))
}

#' Internal function: evaluate the robust polynomial combination r(y)
evaluate_r <- function(y, X3, X4, X5, X6) {
  p5 <- 6 * (2 * X3 * y^3 + X5 * y^2 - 3 * X3 * X4 * y + 2 * X3^3)^2 + 
        (2 * y^3 + 3 * X4 * y - 4 * X3^2)^2 * (2 * y^3 + X4 * y - X3^2)
        
  p6 <- (4 * X3^2 - 3 * X4 * y - 2 * y^3) * 
        (4 * X4^3 - 4 * X3^2 * X4 * y - 8 * X3^2 * y^3 - X4^2 * y^2 + 8 * X4 * y^4 + X6 * y^3 + 4 * y^6) - 
        (10 * X3^3 - 7 * X3 * X4 * y - 2 * X3 * y^3) * 
        (2 * X3^3 - 3 * X3 * X4 * y + 2 * X3 * y^3 + X5 * y^2)
        
  return(p5^2 + p6^2)
}

#' Recover Alpha from Moments
#' Algorithm 3.1 subroutine to solve for the reparameterized variance difference proxy Alpha
RecoverAlphaFromMoments <- function(X3, X4, X5, X6, epsilon = 1e-4) {
  # ymax is the largest real root of 2y^3 + X4*y - X3^2 = 0
  roots <- polyroot(c(-X3^2, X4, 0, 2))
  real_roots <- Re(roots)[abs(Im(roots)) < 1e-8]
  if (length(real_roots) == 0) stop("No real roots found for ymax")
  ymax <- max(real_roots)
  if (ymax <= 0) {
      ymax <- 1e-6 # fallback if degenerate
  }
  
  kappa <- 1 + sqrt(abs(X4)) / ymax
  upper_limit <- (1 + epsilon / kappa) * ymax
  
  # Find candidates for alpha: we want the roots of r'(y), i.e., the local minima of r(y)
  # Evaluating over a dense grid to find candidates
  grid_points <- seq(1e-8, upper_limit, length.out = 10000)
  r_vals <- evaluate_r(grid_points, X3, X4, X5, X6)
  
  diffs <- diff(r_vals)
  is_local_min <- c(FALSE, diffs[-length(diffs)] < 0 & diffs[-1] > 0, FALSE)
  min_indices <- which(is_local_min)
  
  candidate_alphas <- upper_limit
  if (length(min_indices) > 0) {
    # Refine local minima
    refined_minima <- sapply(min_indices, function(i) {
      lower <- grid_points[max(1, i - 2)]
      upper <- grid_points[min(length(grid_points), i + 2)]
      res <- optimize(evaluate_r, interval = c(lower, upper), X3=X3, X4=X4, X5=X5, X6=X6)
      return(res$minimum)
    })
    candidate_alphas <- c(refined_minima, upper_limit)
  }
  
  # Filter strictly to condition: r(alpha) <= 2 * alpha^18 * kappa^10 * epsilon
  valid_alphas <- c()
  for (cand in candidate_alphas) {
    if (cand <= upper_limit) {
      r_val <- evaluate_r(cand, X3, X4, X5, X6)
      threshold <- 2 * (cand^18) * (kappa^10) * epsilon
      if (r_val <= threshold) {
        valid_alphas <- c(valid_alphas, cand)
      }
    }
  }
  
  if (length(valid_alphas) == 0) {
    return(upper_limit)
  }
  
  return(max(valid_alphas))
}

#' Recover 1D Gaussian Mixture params from moments (Algorithm 3.1)
RecoverFromMoments <- function(mu, sigma2, X3, X4, X5, X6, epsilon = 1e-4) {
  alpha <- RecoverAlphaFromMoments(X3, X4, X5, X6, epsilon)
  
  gamma_num <- alpha^2 * X5 + 2 * X3^3 + 2 * alpha^3 * X3 - 3 * X3 * X4 * alpha
  gamma_den <- 4 * X3^2 - 2 * alpha^3 - 3 * X4 * alpha
  
  # Avoid exact division by zero if pathological
  if (abs(gamma_den) < 1e-12) {
      gamma_den <- sign(gamma_den) * 1e-12
      if (gamma_den == 0) gamma_den <- 1e-12
  }
  
  gamma <- (1 / alpha) * (gamma_num / gamma_den)
  beta <- (1 / alpha) * (X3 - 3 * alpha * gamma)
  
  # Calculate shifted means (mathematically corrected quadratic formula from paper)
  disc <- sqrt(max(0, beta^2 + 4 * alpha))
  mu1 <- (beta - disc) / 2
  mu2 <- (beta + disc) / 2
  
  # Probabilities
  p1 <- mu2 / (mu2 - mu1)
  p2 <- -mu1 / (mu2 - mu1)
  
  p1 <- max(0, min(1, p1))
  p2 <- 1 - p1
  
  # Variances
  sigma1_sq <- sigma2 - (p1 * mu1^2 + p2 * mu2^2 - mu1 * gamma)
  sigma2_sq <- sigma1_sq + (mu2 - mu1) * gamma
  
  sigma1_sq <- max(1e-10, sigma1_sq)
  sigma2_sq <- max(1e-10, sigma2_sq)
  
  return(list(
    comp1 = list(p = p1, mu = mu1 + mu, sigma = sqrt(sigma1_sq)),
    comp2 = list(p = p2, mu = mu2 + mu, sigma = sqrt(sigma2_sq)),
    alpha = alpha, beta = beta, gamma = gamma
  ))
}

#' Recover 1D Gaussian Mixture params when means are equal (Algorithm 3.2)
#' 
#' @param mu Overall mean
#' @param sigma2 Overall variance
#' @param X4 4th excess moment
#' @param X6 6th excess moment
SameMeanRecoverFromMoments <- function(mu, sigma2, X4, X6) {
  # Avoid exact division by zero if pathological
  if (abs(X4) < 1e-12) {
    X4 <- sign(X4) * 1e-12
    if (X4 == 0) X4 <- 1e-12
  }
  
  delta_sigma2 <- sqrt((4/3) * X4 + (X6^2) / (25 * X4^2))
  
  # Fix typo in paper's pseudocode Algorithm 3.2: 
  # X6 actually tracks (p1 - p2), not (p2 - p1). So p1 takes the + branch.
  p1 <- 0.5 * (1 + X6 / (5 * X4 * delta_sigma2))
  p2 <- 1 - p1
  
  # Ensure probabilities are in [0, 1]
  p1 <- max(0, min(1, p1))
  p2 <- 1 - p1
  
  sigma1_sq <- sigma2 - p2 * delta_sigma2
  sigma2_sq <- sigma2 + p1 * delta_sigma2
  
  # Ensure variances are strictly positive
  sigma1_sq <- max(1e-10, sigma1_sq)
  sigma2_sq <- max(1e-10, sigma2_sq)
  
  return(list(
    comp1 = list(p = p1, mu = mu, sigma = sqrt(sigma1_sq)),
    comp2 = list(p = p2, mu = mu, sigma = sqrt(sigma2_sq))
  ))
}
