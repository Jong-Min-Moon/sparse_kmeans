# Spectral Clustering Helper

if (!require(RSpectra)) install.packages("RSpectra", repos = "http://cran.us.r-project.org")
library(RSpectra)

#' Spectral Clustering (Matching MATLAB implementation)
#' 
#' Perfroms spectral clustering using the affinity matrix H = (X'X)/n.
#' Uses RSpectra for fast eigen decomposition.
#' 
#' @param x Data matrix (p x n)
#' @param k Number of clusters
#' @return Vector of cluster assignments
#' @export
cluster_spectral <- function(x, k) {
  n <- ncol(x)
  p <- nrow(x)
  
  # Affinity Matrix H_hat = (X'X)/n
  # We only need top k eigenvectors, so efficient SVD on X might be better than X'X?
  # H_hat is n x n. If n is large, X'X is large.
  # But X is p x n.
  # Eigenvectors of X'X are Right Singular Vectors of X.
  # svds(X, k) gives u, d, v. V are eigenvectors of X'X.
  # Squared singular values / n are eigenvalues of X'X/n.
  
  # MATLAB: [V,D] = eig(H_hat);
  # We want largest eigenvalues.
  
  if (k < 2) stop("k must be at least 2")
  
  # Use svds on X is often faster and more numerically stable than eigs on X'X
  # But we need to center? MATLAB code uses raw X'X.
  # Let's use svds(t(x), k) -> returns U, D, V.
  # If we apply svds on X (p x n) -> U (p x k), D (k), V (n x k).
  # X = U D V'. X'X = V D U' U D V' = V D^2 V'.
  # So V are the eigenvectors of X'X.
  
  # We need top 2 eigenvectors for the heuristic logic even if k > 2?
  # MATLAB code specific logic:
  # Checks d(1)/d(2) and f1 to decide whether to use 1st, 2nd, or both eigenvectors (for k=2 case likely)
  # The MATLAB code seems hardcoded for k=2 or specific logic?
  # "new_data = V(:,1:2)" or "new_data = V(:,1)" or "new_data = V(:,2)".
  # Then kmeans(new_data, k).
  
  # Let's implement the exact logic.
  # It retrieves full decompositions or enough for logic.
  # Let's get k+2 components to be safe or just 3.
  # MATLAB V is sorted? " [d,ind] = sort(diag(D), 'descend'); "
  
  # Using RSpectra::svds
  # svds returns d sorted descending.
  
  num_components <- max(k, 3) 
  decomp <- svds(x, k = num_components, nu = 0, nv = num_components)
  
  d <- (decomp$d^2) / n # Eigenvalues of X'X/n
  V <- decomp$v         # Eigenvectors
  
  # Heuristic logic from MATLAB
  tau_n <- 1 / log(n + p)
  delta_n <- tau_n^2
  f1 <- abs(sum(V[, 1])) / sqrt(n) - 1
  
  # Logic seems to be deciding dimensions for clustering
  # Default use first k?
  # MATLAB logic:
  # if d(1)/d(2) < 1 + tau_n -> use 1:2
  # elseif f1 > delta_n -> use 1
  # else -> use 2
  
  # This implies k=2 context usually.
  # If k > 2, this logic might be too specific.
  # But assuming we follow it for the projection.
  
  if (d[1]/d[2] < 1 + tau_n) {
    new_data <- V[, 1:2]
  } else if (f1 > delta_n) {
    new_data <- V[, 1, drop=FALSE]
  } else {
    new_data <- V[, 2, drop=FALSE]
  }
  
  # If k > 2, force use of k coords?
  # The MATLAB code passes 'k' to kmeans.
  # If new_data has 1 col, kmeans(x, k) might fail if k > 1? 
  # Actually kmeans on 1D data is just split points.
  # If k > 2, maybe we should just use V[, 1:k]?
  # For now, I will stick to the MATLAB logic which seems to try to handle "uninformative first eigenvector" case?
  # (Vector of all 1s often appears in Laplacian, but here H = X'X/n)
  
  if (k > 2 && ncol(new_data) < k) {
      new_data <- V[, 1:k]
  }
  
  # Run K-means
  # nstart=20 for stability
  km_res <- kmeans(new_data, centers = k, nstart = 20)
  
  return(km_res$cluster)
}
