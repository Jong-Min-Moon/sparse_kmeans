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
