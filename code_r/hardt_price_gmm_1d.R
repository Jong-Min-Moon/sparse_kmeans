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
        (4 * X3^4 - 4 * X3^2 * X4 * y - 8 * X3^2 * y^3 - X4^2 * y^2 + 8 * X4 * y^4 + X6 * y^3 + 4 * y^6) - 
        (10 * X3^3 - 7 * X3 * X4 * y - 2 * X3 * y^3) * 
        (2 * X3^3 - 3 * X3 * X4 * y + 2 * X3 * y^3 + X5 * y^2)
        
  return(p5^2 + p6^2)
}

#' Recover Alpha from Moments
#' Subroutine to solve for candidate values of the reparameterized variance difference proxy Alpha
#' Returns a list of all candidates (roots of p5) rather than relying on the paper's p6 (which contains typos)
RecoverAlphasFromMoments <- function(X3, X4, X5, X6, epsilon = 1e-4) {
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
  
  # Derive polynomial coefficients of p5(y). 
  # p5(y) is a 9th degree polynomial in y:
  # p5(y) = 6*(2*X3*y^3 + X5*y^2 - 3*X3*X4*y + 2*X3^3)^2 + (2*y^3 + 3*X4*y - 4*X3^2)^2 * (2*y^3 + X4*y - X3^2)
  
  poly_add <- function(p1, p2) {
    n <- max(length(p1), length(p2))
    p1 <- c(p1, rep(0, n - length(p1)))
    p2 <- c(p2, rep(0, n - length(p2)))
    return(p1 + p2)
  }
  
  poly_mul <- function(p1, p2) {
    res <- numeric(length(p1) + length(p2) - 1)
    for (i in seq_along(p1)) {
      for (j in seq_along(p2)) {
        res[i + j - 1] <- res[i + j - 1] + p1[i] * p2[j]
      }
    }
    return(res)
  }
  
  # Term 1: 6 * (2*X3^3 - 3*X3*X4*y + X5*y^2 + 2*X3*y^3)^2
  t1_base <- c(2 * X3^3, -3 * X3 * X4, X5, 2 * X3)
  term1 <- 6 * poly_mul(t1_base, t1_base)
  
  # Term 2: (-4*X3^2 + 3*X4*y + 0*y^2 + 2*y^3)^2 * (-X3^2 + X4*y + 0*y^2 + 2*y^3)
  t2_base1 <- c(-4 * X3^2, 3 * X4, 0, 2)
  t2_sq <- poly_mul(t2_base1, t2_base1)
  t2_base2 <- c(-X3^2, X4, 0, 2)
  term2 <- poly_mul(t2_sq, t2_base2)
  
  # Total coefficients of p5(y)
  p5_coeffs <- poly_add(term1, term2)
  
  # Use R's state-of-the-art Jenkins-Traub root finder on the exact coefficients
  all_roots <- polyroot(p5_coeffs)
  
  # Extract real roots (allowing for tiny numeric imaginary residuals)
  real_roots <- Re(all_roots)[abs(Im(all_roots)) < 1e-8]
  
  # Filter to strictly positive domain candidates within bounded range
  candidate_alphas <- c(upper_limit, real_roots[real_roots > 0 & real_roots <= upper_limit + 1e-6])
  
  # We return all candidates, sorted descending. The caller will filter by physical feasibility.
  return(sort(candidate_alphas, decreasing = TRUE))
}

#' Recover 1D Gaussian Mixture params from moments (Algorithm 3.1)
RecoverFromMoments <- function(mu, sigma2, X3, X4, X5, X6, epsilon = 1e-4) {
  alphas <- RecoverAlphasFromMoments(X3, X4, X5, X6, epsilon)
  
  best_candidate <- NULL
  
  for (alpha in alphas) {
    # gamma expression from exact algorithm
    gamma_num <- alpha^2 * X5 + 2 * X3^3 + 2 * alpha^3 * X3 - 3 * X3 * X4 * alpha
    gamma_den <- 4 * X3^2 - 2 * alpha^3 - 3 * X4 * alpha
    
    # Avoid exact division by zero if pathological
    if (abs(gamma_den) < 1e-12) {
        gamma_den <- sign(gamma_den) * 1e-12
        if (gamma_den == 0) gamma_den <- 1e-12
    }
    
    gamma <- (1 / alpha) * (gamma_num / gamma_den)
    beta <- (1 / alpha) * (X3 - 3 * alpha * gamma)
    
    # Calculate shifted means
    disc_val <- beta^2 + 4 * alpha
    if (disc_val < 0) next # Unphysical
    
    disc <- sqrt(disc_val)
    mu1 <- (beta - disc) / 2
    mu2 <- (beta + disc) / 2
    
    # Probabilities
    if (abs(mu2 - mu1) < 1e-12) next
    
    p1 <- mu2 / (mu2 - mu1)
    p2 <- -mu1 / (mu2 - mu1)
    
    # Check probability bounds. We allow tiny numerical slack.
    if (p1 < -1e-4 || p1 > 1.0001) next
    
    p1 <- max(0, min(1, p1))
    p2 <- 1 - p1
    
    # Variances
    sigma1_sq <- sigma2 - (p1 * mu1^2 + p2 * mu2^2 - mu1 * gamma)
    sigma2_sq <- sigma1_sq + (mu2 - mu1) * gamma
    
    # Validation against massive negative variances (slight negative OK under noise)
    if (sigma1_sq < -1e-2 || sigma2_sq < -1e-2) next
    
    sigma1_sq <- max(1e-10, sigma1_sq)
    sigma2_sq <- max(1e-10, sigma2_sq)
    
    # If we reached here, the candidate is physically valid!
    # Because alphas are sorted decreasing, we pick the maximal valid alpha.
    best_candidate <- list(
      comp1 = list(p = p1, mu = mu1 + mu, sigma = sqrt(sigma1_sq)),
      comp2 = list(p = p2, mu = mu2 + mu, sigma = sqrt(sigma2_sq)),
      alpha = alpha, beta = beta, gamma = gamma
    )
    break
  }
  
  if (is.null(best_candidate)) {
      # Fallback to the upper limit alpha if no physical parameters were found
      alpha <- alphas[1] # the upper_limit
      gamma_num <- alpha^2 * X5 + 2 * X3^3 + 2 * alpha^3 * X3 - 3 * X3 * X4 * alpha
      gamma_den <- 4 * X3^2 - 2 * alpha^3 - 3 * X4 * alpha
      if (abs(gamma_den) < 1e-12) gamma_den <- sign(gamma_den) * 1e-12
      gamma <- (1 / alpha) * (gamma_num / gamma_den)
      beta <- (1 / alpha) * (X3 - 3 * alpha * gamma)
      disc <- sqrt(max(0, beta^2 + 4 * alpha))
      mu1 <- (beta - disc) / 2
      mu2 <- (beta + disc) / 2
      p1 <- max(0, min(1, mu2 / (mu2 - mu1)))
      p2 <- 1 - p1
      sigma1_sq <- max(1e-10, sigma2 - (p1 * mu1^2 + p2 * mu2^2 - mu1 * gamma))
      sigma2_sq <- max(1e-10, sigma1_sq + (mu2 - mu1) * gamma)
      best_candidate <- list(
        comp1 = list(p = p1, mu = mu1 + mu, sigma = sqrt(sigma1_sq)),
        comp2 = list(p = p2, mu = mu2 + mu, sigma = sqrt(sigma2_sq)),
        alpha = alpha, beta = beta, gamma = gamma
      )
  }
  
  return(best_candidate)
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

#' Master Algorithm for 1D Gaussian Mixture Recovery (Algorithm 3.3)
#' 
#' @param x Numeric vector of samples
#' @param delta Confidence parameter for error bounds (e.g. 0.01 for 99% confidence)
#' @return A list of the two components
Recover1DMixture <- function(x, delta = 0.05) {
  n <- length(x)
  
  # Standardize data to gracefully handle massive precision loss in large moments
  mu_overall <- mean(x)
  sigma_overall <- sd(x) * sqrt((n - 1) / n) # use population sd
  
  if (sigma_overall < 1e-12) {
    return(list(
      comp1 = list(p = 0.5, mu = mu_overall, sigma = 0),
      comp2 = list(p = 0.5, mu = mu_overall, sigma = 0),
      fallback = TRUE
    ))
  }
  
  x_std <- (x - mu_overall) / sigma_overall
  
  # Step 1: Compute moments on standardized data
  moments <- estimate_excess_moments(x_std)
  mu <- moments$mu
  sigma2 <- moments$sigma2
  X3 <- moments$X3
  X4 <- moments$X4
  X5 <- moments$X5
  X6 <- moments$X6
  mu_std <- moments$mu # Should be 0 for standardized data
  sigma2_std <- moments$sigma2 # Should be 1 for standardized data
  X3_std <- moments$X3
  X4_std <- moments$X4
  X5_std <- moments$X5
  X6_std <- moments$X6
  
  # Step 2: Establish Error bounds
  f <- (log(1 / delta) / n)^(1 / 12)
  sigma_std <- sqrt(sigma2_std) # This will be 1 for standardized data
  
  # Determine if completely unresolvable noise
  # Since f^2 grows very large for small sample subsets (like the ones used in Alg C projection mappings),
  # we damp the theoretical asymptotic check slightly to avoid accidentally blocking valid well-separated
  # projections that have high noise.
  eps_noise <- 0.05 * f^2 * sigma2_std
  if (abs(X4_std) < eps_noise && X3_std^2 < eps_noise) {
       cat("[ROUTING] Overriding: Geometry Fully Obscured. Selected Fallback (Single Component)\n")
       return(list(
         comp1 = list(p = 0.5, mu = mu_overall, sigma = sigma_overall),
         comp2 = list(p = 0.5, mu = mu_overall, sigma = sigma_overall),
         fallback = TRUE
       ))
  }
  
  # Step 3: Bounds calculations (all computed in mathematically stable std space)
  if (X4_std > 0) {
    delta_mu_std <- min(abs(X3_std)^(1/3) + abs(X4_std)^(1/4), abs(X3_std) / sqrt(X4_std))
  } else {
    delta_mu_std <- abs(X3_std)^(1/3) + abs(X4_std)^(1/4)
  }
  
  delta_sigma2_std <- sqrt(abs(X4_std))
  
  # Step 4: Routing
  best_candidate <- NULL 
  
  if (f^2 <= (delta_mu_std^2)) {
    # Means are reliably separated (Algorithm 3.1)
    cat("[ROUTING] Selected Algorithm 3.1 (Well-Separated Means)\n")
    epsilon <- sqrt((1 / max(1e-12, delta_mu_std))^12 * log(1 / delta) / n)
    best_candidate <- RecoverFromMoments(mu_std, sigma2_std, X3_std, X4_std, X5_std, X6_std, epsilon)
    
  } else if (f^2 <= (delta_sigma2_std)) {
    # Means inseparable, but variances are distinct (Algorithm 3.2)
    cat("[ROUTING] Selected Algorithm 3.2 (Identical Means, Distinct Variances)\n")
    best_candidate <- SameMeanRecoverFromMoments(mu_std, sigma2_std, X4_std, X6_std)
  }
  
  if (is.null(best_candidate)) {
    # Geometry fully obscured by noise: Output single Gaussian cluster
    cat("[ROUTING] Selected Fallback (Single Component)\n")
    best_candidate <- list(
      comp1 = list(p = 0.5, mu = 0, sigma = 1),
      comp2 = list(p = 0.5, mu = 0, sigma = 1),
      fallback = TRUE
    )
  }
  
  # Step 5: Unstandardize the recovered parameters
  best_candidate$comp1$mu <- best_candidate$comp1$mu * sigma_overall + mu_overall
  best_candidate$comp1$sigma <- best_candidate$comp1$sigma * sigma_overall
  
  best_candidate$comp2$mu <- best_candidate$comp2$mu * sigma_overall + mu_overall
  best_candidate$comp2$sigma <- best_candidate$comp2$sigma * sigma_overall
  
  return(best_candidate)
}
