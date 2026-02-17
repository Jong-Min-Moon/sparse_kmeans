#' SDP K-Means Clustering (ADMM Solver)
#'
#' Solves the SDP relaxation of K-Means clustering using an optimized ADMM approach.
#' This replaces the slower CVXR-based implementation.
#' optimized with Pre-compiled Rcpp (Simplex Projection) and RSpectra (Truncated Eigen).
#' Requires 'code_r/proj_simplex.dll' to be built via 'code_r/build_solver.ps1'.
#'
#' @param G Gram matrix (n x n)
#' @param K Number of clusters
#' @param rho ADMM penalty parameter (default 1.0)
#' @param max_iter Maximum iterations (default 1000)
#' @param tol Convergence tolerance (default 1e-4)
#' @param verbose Print progress (default FALSE)
#' @return List with cluster assignments and Z matrix
#' @import Rcpp
#' @import RSpectra
#' @import stats
sdp_kmeans <- function(G, K, rho = 1.0, max_iter = 1000, tol = 1e-4, verbose = FALSE) {
  # Load the pre-compiled DLL
  # We assume the DLL is in "code_r/proj_simplex.dll" relative to working directory
  dll_path <- "code_r/proj_simplex.dll"
  if (!file.exists(dll_path)) {
    stop("Pre-compiled library 'code_r/proj_simplex.dll' not found. Please run 'code_r/build_solver.ps1'.")
  }

  # Check if loaded, if not load it
  if (!("proj_simplex" %in% names(getLoadedDLLs()))) {
    dyn.load(dll_path)
  }

  # Wrapper for the C++ function
  proj_simplex_rows_cpp <- function(Mat) {
    .Call("proj_simplex_rows_wrapper", Mat)
  }

  n <- nrow(G)

  # 1. Warm-Start Initialization using Spectral Clustering on G
  if (verbose) cat("Initializing with Spectral Clustering on G...\n")
  init_decomp <- RSpectra::eigs_sym(G, K, which = "LA")
  V_init <- init_decomp$vectors
  km_init <- kmeans(V_init, centers = K, nstart = 10)

  Z <- matrix(0, n, n)
  for (k in 1:K) {
    idx <- which(km_init$cluster == k)
    if (length(idx) > 0) {
      Z[idx, idx] <- 1 / length(idx)
    }
  }

  Y <- Z
  Lambda <- matrix(0, n, n)
  prev_vectors <- V_init

  # Helper: Truncated Projection onto PSD + Trace K
  proj_psd_trace_truncated <- function(M, target_trace, k_prime, init_vec = NULL) {
    M <- (M + t(M)) / 2

    opts <- list(retvec = TRUE)
    if (!is.null(init_vec)) opts$initvec <- init_vec[, 1]

    k_prime <- min(k_prime, n - 2)
    if (k_prime < K) k_prime <- K + 2

    tryCatch(
      {
        eig <- RSpectra::eigs_sym(M, k = k_prime, which = "LA", opts = opts)
        vals <- eig$values
        vecs <- eig$vectors

        u <- sort(vals, decreasing = TRUE)
        cssv <- cumsum(u)
        vec_cond <- u + (target_trace - cssv) / seq_along(u)
        rho_idx <- max(which(vec_cond > 0))
        theta <- (target_trace - cssv[rho_idx]) / rho_idx
        w <- pmax(vals + theta, 0)

        pos_idx <- w > 1e-10
        if (sum(pos_idx) == 0) {
          return(list(Z = matrix(0, n, n), V = vecs))
        }

        Z_out <- vecs[, pos_idx, drop = FALSE] %*% (t(vecs[, pos_idx, drop = FALSE]) * w[pos_idx])
        return(list(Z = Z_out, V = vecs))
      },
      error = function(e) {
        if (verbose) cat("RSpectra failed, falling back to eigen()...\n")
        eig <- eigen(M, symmetric = TRUE)
        vals <- eig$values
        vecs <- eig$vectors

        u <- sort(vals, decreasing = TRUE)
        cssv <- cumsum(u)
        vec_cond <- u + (target_trace - cssv) / seq_along(u)
        rho_idx <- max(which(vec_cond > 0))
        theta <- (target_trace - cssv[rho_idx]) / rho_idx
        w <- pmax(vals + theta, 0)

        pos_idx <- w > 1e-10
        Z_out <- vecs[, pos_idx, drop = FALSE] %*% (t(vecs[, pos_idx, drop = FALSE]) * w[pos_idx])
        return(list(Z = Z_out, V = vecs[, 1:k_prime]))
      }
    )
  }

  mu <- 10.0
  tau <- 2.0
  k_prime <- min(10 * K, n - 1)

  for (iter in 1:max_iter) {
    # 1. Z Update (Spectral)
    M <- Y - Lambda / rho + G / rho
    res_z <- proj_psd_trace_truncated(M, K, k_prime, prev_vectors)
    Z_new <- res_z$Z
    prev_vectors <- res_z$V

    # 2. Y Update (Linear - Pre-compiled Rcpp)
    N_mat <- Z_new + Lambda / rho
    # Using as.matrix to ensure SEXP compatibility
    Y_new <- proj_simplex_rows_cpp(as.matrix(N_mat))

    # 3. Lambda Update
    resid <- Z_new - Y_new
    Lambda <- Lambda + rho * resid

    # 4. Check Convergence & Adaptive Rho
    dual_resid <- rho * norm(Y_new - Y, "F")
    primal_resid <- norm(resid, "F")

    if (iter %% 10 == 0) {
      if (verbose) {
        cat(sprintf(
          "Iter %d: Rho=%.2f Primal=%.2e Dual=%.2e Obj=%.2f\n",
          iter, rho, primal_resid, dual_resid, sum(G * Z_new)
        ))
      }
      if (primal_resid > mu * dual_resid) {
        rho <- rho * tau
      } else if (dual_resid > mu * primal_resid) {
        rho <- rho / tau
      }
    }

    if (primal_resid < tol && dual_resid < tol) {
      if (verbose) cat(sprintf("Converged at iter %d\n", iter))
      Z <- Z_new
      break
    }

    Z <- Z_new
    Y <- Y_new
  }

  # Final Clustering
  eig_Z <- RSpectra::eigs_sym(Z, K, which = "LA")
  V <- eig_Z$vectors
  km <- kmeans(V, centers = K, nstart = 20)

  return(list(
    cluster = km$cluster,
    Z = Z,
    iter = iter,
    value = sum(G * Z)
  ))
}
