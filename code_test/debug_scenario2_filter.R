source("code_test/debug_scenario2.R")
source("code_r/hardt_price_gmm_1d.R")

true_alpha <- 1.6875

roots <- polyroot(c(-X3_true^2, X4_true, 0, 2))
real_roots <- Re(roots)[abs(Im(roots)) < 1e-8]
ymax <- max(real_roots)

kappa <- 1 + sqrt(abs(X4_true))/ymax
epsilon <- 1e-8
upper_limit <- (1 + epsilon/kappa) * ymax

# Just manually run the candidate code:
  p5 <- function(y) {
    6 * (2 * X3_true * y^3 + X5_true * y^2 - 3 * X3_true * X4_true * y + 2 * X3_true^3)^2 + 
    (2 * y^3 + 3 * X4_true * y - 4 * X3_true^2)^2 * (2 * y^3 + X4_true * y - X3_true^2)
  }
  
  grid_points <- seq(1e-8, upper_limit, length.out = 10000)
  p5_vals <- sapply(grid_points, p5)
  
  signs <- sign(p5_vals)
  crossings <- which(diff(signs) != 0)
  
  candidate_alphas <- c(upper_limit)
  for (i in crossings) {
    lower <- grid_points[max(1, i - 1)]
    upper <- grid_points[min(length(grid_points), i + 2)]
    
    res <- tryCatch({
       uniroot(p5, interval = c(lower, upper))$root
    }, error = function(e) {
       optimize(function(y) p5(y)^2, interval = c(lower, upper))$minimum
    })
    
    candidate_alphas <- c(candidate_alphas, res)
  }
  
  cat("\nCandidates:\n")
  print(candidate_alphas)
  
  for (cand in candidate_alphas) {
      r_val <- evaluate_r(cand, X3_true, X4_true, X5_true, X6_true)
      threshold <- (epsilon^2) * (cand^18) * (kappa^10)
      cat(sprintf("Cand: %.4f | r(x): %.4e | Thresh: %.4e | Valid: %s\n", cand, r_val, threshold, r_val <= threshold))
  }

