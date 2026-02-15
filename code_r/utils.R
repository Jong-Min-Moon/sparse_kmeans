# Utility functions for clustering evaluation and stopping criteria

#' Compute Rand Index and variations (ARI, MI, HI)
#' 
#' @param c1 Clustering assignment 1 (vector)
#' @param c2 Clustering assignment 2 (vector)
#' @return List with components AR, RI, MI, HI
#' @importFrom mcclust arandi
RandIndex <- function(c1, c2) {
  # Ensure integer vectors
  c1 <- as.integer(c1)
  c2 <- as.integer(c2)
  
  if (length(c1) != length(c2)) {
    stop("Input vectors must have the same length")
  }

  n <- length(c1)
  
  # Contingency Table
  C <- table(c1, c2)
  
  nis <- rowSums(C^2) # sum of squares of row sums? No.
  # MATLAB: nis=sum(sum(C,2).^2); -> This is sum of (row sums)^2.
  # Wait, sum(C,2) is row sums. sum(...^2) is sum of squares.
  # So nis = sum(rowSums(C)^2).
  
  row_sums <- rowSums(C)
  col_sums <- colSums(C)
  
  nis <- sum(row_sums^2)
  njs <- sum(col_sums^2)
  
  t1 <- choose(n, 2)
  t2 <- sum(C^2)
  t3 <- 0.5 * (nis + njs)
  
  # Expected index
  nc <- (n * (n^2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
  
  A <- t1 + t2 - t3 # Agreements
  D <- -t2 + t3     # Disagreements
  
  if (t1 == nc) {
    AR <- 0
  } else {
    AR <- (A - nc) / (t1 - nc)
  }
  
  RI <- A / t1
  MI <- D / t1
  HI <- (A - D) / t1
  
  return(list(AR = AR, RI = RI, MI = MI, HI = HI))
}


