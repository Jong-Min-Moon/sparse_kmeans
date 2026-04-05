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
#' @param max_iter Maximum iterations (default 10000)
#' @param tol Convergence tolerance (default 1e-4)
#' @param verbose Print progress (default FALSE)
#' @param k_prime_factor Multiplier for K to determine truncated eigen dimensions (default 3)
#' @param mu Factor for primal/dual residual mismatch (default 10.0)
#' @param tau Scaling factor for rho (default 2.0)
#' @param report_interval Iteration interval for verbose output (default 10)
#' @return List with cluster assignments and Z matrix
#' @import Rcpp
#' @import RSpectra
#' @import stats
sdp_kmeans <- function(G, K, rho = 1.0, max_iter = 2000, tol = 1e-3, verbose = TRUE, k_prime_factor = 3, mu = 10.0, tau = 2.0, report_interval = 500) {
  start_time <- Sys.time()
  # Load the pre-compiled library
  # Windows: code_r/proj_simplex.dll
  # Linux/Unix: code_r/proj_simplex.so (needs to be compiled)

  if (.Platform$OS.type == "windows") {
    possibilities <- c("code_r/proj_simplex.dll", "../../code_r/proj_simplex.dll", "../code_r/proj_simplex.dll", "../../../code_r/proj_simplex.dll")
    lib_path <- NULL
    for (path_try in possibilities) {
      if (length(path_try) > 0 && !is.na(path_try) && nzchar(path_try) && file.exists(path_try)) {
        lib_path <- path_try
        break
      }
    }

    if (is.null(lib_path) || !nzchar(lib_path)) {
      stop("Pre-compiled library 'proj_simplex.dll' not found. Please run 'code_r/build_solver.ps1'.")
    }
    if (!("proj_simplex" %in% names(getLoadedDLLs()))) dyn.load(lib_path)
  } else {
    # Linux / Unix
    # Check potential locations
    possibilities <- c("code_r/proj_simplex.so", "../../code_r/proj_simplex.so", "../code_r/proj_simplex.so", "../../../code_r/proj_simplex.so")
    lib_path <- NULL
    for (path_try in possibilities) {
      if (length(path_try) > 0 && !is.na(path_try) && nzchar(path_try) && file.exists(path_try)) {
        lib_path <- path_try
        break
      }
    }

    if (is.null(lib_path)) {
      # Fallback: Default to code_r/proj_simplex.so and try to compile using found source
      lib_path <- "code_r/proj_simplex.so" # Default target if compilation succeeds

      # Attempt auto-compilation if not found
      if (verbose) cat("Compiling proj_simplex.cpp for Linux...\n")

      tryCatch(
        {
          # Check for the source file in likely locations
          src_possibilities <- c("code_r/proj_simplex.cpp", "../../code_r/proj_simplex.cpp", "../code_r/proj_simplex.cpp", "../../../code_r/proj_simplex.cpp")
          src_path <- NULL
          for (sp in src_possibilities) {
            if (file.exists(sp)) {
              src_path <- sp
              break
            }
          }

          if (is.null(src_path)) stop("Source file 'proj_simplex.cpp' not found in code_r/ or ../../code_r/ or ../../../code_r/")

          # Determine target lib path based on source location
          # If src is ../../code_r/proj_simplex.cpp, we should compile to ../../code_r/proj_simplex.so
          target_lib <- sub("\\.cpp$", ".so", src_path)

          # Compile command
          # Note: We rely on R to set flags, but if we need custom Rcpp flags they should be in env or ~/.R/Makevars
          # Since compile_solver.R sets them, we hope they persist or R handles it.
          # But auto-compilation inside R session might miss the env vars set in compile_solver.R??
          # Actually compile_solver.R runs in a separate process.
          # Here we are in the main R process.

          cmd <- sprintf("R CMD SHLIB -o %s %s", target_lib, src_path)
          res <- system(cmd)

          if (res != 0 || !file.exists(target_lib)) stop("Compilation failed.")
          lib_path <- target_lib
          if (!is.null(lib_path) && length(lib_path) == 1 && nzchar(lib_path) && !("proj_simplex" %in% names(getLoadedDLLs()))) dyn.load(lib_path)
        },
        error = function(e) {
          stop(paste("Failed to compile Rcpp module:", e$message))
        }
      )
    }
    if (!("proj_simplex" %in% names(getLoadedDLLs()))) dyn.load(lib_path)
  }

  # Check if C++ backend is loaded
  use_cpp <- !is.null(lib_path) && length(lib_path) == 1 && nzchar(lib_path) && ("proj_simplex" %in% names(getLoadedDLLs()))

  # Implementation of Simplex Projection (Pure R Fallback)
  proj_simplex_pure_r <- function(y, target_sum = 1.0) {
    # Condat's algorithm or similar sort-based projection
    n <- length(y)
    u <- sort(y, decreasing = TRUE)
    cssv <- cumsum(u)
    vec_cond <- u + (target_sum - cssv) / seq_along(u)
    rho_idx <- max(which(vec_cond > 0))
    theta <- (target_sum - cssv[rho_idx]) / rho_idx
    return(pmax(y + theta, 0))
  }

  # Wrapper for the projection (chooses between C++ and R)
  proj_simplex_rows <- function(Mat, target_sum = 1.0) {
    if (use_cpp) {
      return(.Call("proj_simplex_rows_wrapper", as.matrix(Mat), target_sum))
    } else {
      # Apply pure R version row-wise
      return(t(apply(Mat, 1, proj_simplex_pure_r, target_sum = target_sum)))
    }
  }

  if (!use_cpp) {
    warning("C++ backend for simplex projection not found. Falling back to (slow) R implementation.")
  }

  n <- nrow(G)

  # 1. Warm-Start Initialization using Spectral Clustering on G
  if (verbose) cat("Initializing with Spectral Clustering on G...\n")
  init_decomp <- RSpectra::eigs_sym(G, K, which = "LA")
  if (verbose) cat("Spectral decomposition done.\n")
  V_init <- init_decomp$vectors
  km_init <- kmeans(V_init, centers = K, nstart = 10)
  if (verbose) cat("K-means initialization done. Starting ADMM loop...\n")

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

  k_prime <- min(k_prime_factor * K, n - 1)

  for (iter in 1:max_iter) {
    # 1. Z Update (Spectral)
    M <- Y - Lambda / rho + G / rho
    res_z <- proj_psd_trace_truncated(M, K, k_prime, prev_vectors)
    Z_new <- res_z$Z
    prev_vectors <- res_z$V

    # 2. Y Update (Linear - Pre-compiled Rcpp or R Fallback)
    N_mat <- Z_new + Lambda / rho
    # Using as.matrix to ensure SEXP compatibility if using C++
    Y_new <- proj_simplex_rows(N_mat)

    # 3. Lambda Update
    resid <- Z_new - Y_new
    Lambda <- Lambda + rho * resid

    # 4. Check Convergence & Adaptive Rho
    dual_resid <- rho * norm(Y_new - Y, "F")
    primal_resid <- norm(resid, "F")

    if (iter %% report_interval == 0) {
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

  cat(sprintf("sdp_kmeans finished: %d iterations in %.2f seconds\n", iter, as.numeric(difftime(Sys.time(), start_time, units = "secs"))))

  return(list(
    cluster = km$cluster,
    Z = Z,
    iter = iter,
    value = sum(G * Z)
  ))
}
