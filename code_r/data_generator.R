#' Sparse Symmetric Data Generator
#'
#' Replicates the logic of bandit/get_specification_chaingraph.m
#'
#' @param support Indices of signal features (e.g. 1:10)
#' @param separation Distance between cluster means
#' @param dimension Total number of features (p)
#' @param precision_sparsity Sparsity of the precision matrix (rho-based tridiagonal)
#' @param conditional_correlation The rho value (correlation/precision parameter)
#' @param flip Whether to treat the matrix as covariance instead of precision
#' @export
get_specification_chaingraph <- function(support, separation, dimension, precision_sparsity, conditional_correlation, flip = FALSE) {
    # 1. Generate Sparse Precision Matrix
    sparse_precision_matrix <- diag(1, dimension)

    if (is.character(precision_sparsity)) {
        if (precision_sparsity == "nonsparse") {
            max_for_loop <- dimension - 1
        } else {
            max_for_loop <- 0
        }
    } else if (precision_sparsity >= 2) {
        max_for_loop <- floor(precision_sparsity / 2)
    } else {
        max_for_loop <- 0
    }

    if (max_for_loop > 0) {
        for (i in 1:max_for_loop) {
            val <- conditional_correlation^i
            # Upper diagonal
            for (j in 1:(dimension - i)) {
                sparse_precision_matrix[j, j + i] <- val
                sparse_precision_matrix[j + i, j] <- val
            }
        }
    }

    # 2. Covariance vs Precision
    if (flip) {
        covariance_matrix <- sparse_precision_matrix
        sparse_precision_matrix <- solve(covariance_matrix)
    } else {
        covariance_matrix <- solve(sparse_precision_matrix)
    }

    # 3. Magnitude Calculation
    # magnitude = separation / 2 / sqrt( sum( Sigma[support, support] ) )
    Sigma_S0 <- covariance_matrix[support, support]
    sum_Sigma_S0 <- sum(Sigma_S0)
    magnitude <- separation / 2 / sqrt(sum_Sigma_S0)

    # 4. Mean Vectors
    # sparse_pre_mean_one = indicator vector on support
    sparse_pre_mean_one <- rep(0, dimension)
    sparse_pre_mean_one[support] <- 1

    sparse_pre_mean_0 <- magnitude * sparse_pre_mean_one
    mean_0 <- covariance_matrix %*% sparse_pre_mean_0

    mu1 <- -mean_0 # cluster 1 mean
    mu2 <- mean_0 # cluster 2 mean

    return(list(
        support = support,
        separation = separation,
        dimension = dimension,
        precision_sparsity = precision_sparsity,
        rho = conditional_correlation,
        precision_matrix = sparse_precision_matrix,
        covariance_matrix = covariance_matrix,
        magnitude = magnitude,
        mu1 = as.numeric(mu1),
        mu2 = as.numeric(mu2)
    ))
}

#' AR(1) Precision Matrix Specification Generator
#'
#' Variation of the chain graph where the precision matrix assumes an AR(1)
#' covariance structure: Omega_ij = rho^|i-j|.
#'
#' @param support Indices of signal features (e.g. 1:10)
#' @param separation Distance between cluster means
#' @param dimension Total number of features (p)
#' @param rho The AR(1) decay parameter. Defaults to 0.8.
#' @export
get_specification_ar1 <- function(support, separation, dimension, rho = 0.8) {
    # 1. Generate AR(1) Precision Matrix: Omega_ij = rho^|i-j|
    sparse_precision_matrix <- matrix(0, dimension, dimension)
    coords <- 1:dimension
    # Efficient vectorized computation of rho^|i-j|
    sparse_precision_matrix <- rho^abs(outer(coords, coords, "-"))

    # 2. Covariance matrix (Invert Omega)
    # Note: For this structure, the covariance matrix Sigma will be tridiagonal.
    covariance_matrix <- solve(sparse_precision_matrix)
    covariance_matrix <- (covariance_matrix + t(covariance_matrix)) / 2 # Ensure symmetry

    # 3. Magnitude Calculation
    # magnitude = separation / 2 / sqrt( sum( Sigma[support, support] ) )
    Sigma_S0 <- covariance_matrix[support, support]
    sum_Sigma_S0 <- sum(Sigma_S0)
    magnitude <- separation / 2 / sqrt(sum_Sigma_S0)

    # 4. Mean Vectors
    sparse_pre_mean_one <- rep(0, dimension)
    sparse_pre_mean_one[support] <- 1

    sparse_pre_mean_0 <- magnitude * sparse_pre_mean_one
    mean_0 <- covariance_matrix %*% sparse_pre_mean_0

    mu1 <- -mean_0
    mu2 <- mean_0

    return(list(
        support = support,
        separation = separation,
        dimension = dimension,
        precision_sparsity = "ar1",
        rho = rho,
        precision_matrix = sparse_precision_matrix,
        covariance_matrix = covariance_matrix,
        magnitude = magnitude,
        mu1 = as.numeric(mu1),
        mu2 = as.numeric(mu2)
    ))
}


#' Generate data from a specification object
#'
#' Samples \code{n} observations from a two-class Gaussian (or heavy-tailed)
#' mixture defined by a specification object. Compatible with specifications
#' produced by \code{get_specification_chaingraph}, \code{get_specification_identity},
#' and \code{get_specification_erdos_renyi}.
#'
#' If \code{covariance_matrix} is absent from the specification but
#' \code{precision_matrix} is present, the covariance is derived internally via
#' a numerically stabilised matrix inversion.
#'
#' @param specification A specification list with at minimum: \code{dimension},
#'   \code{mu1}, \code{mu2}, and either \code{covariance_matrix} or
#'   \code{precision_matrix}. Identity-covariance specs additionally need
#'   \code{precision_sparsity} and \code{rho} set to \code{0}.
#' @param n Total sample size (split equally between the two classes).
#' @param seed Optional integer seed for reproducibility.
#' @param noise Noise distribution: \code{"Gaussian"} (default), \code{"t"}
#'   (t(6) scaled to unit variance), or \code{"Laplace"}.
#' @return A list with \code{X} (p x n data matrix) and \code{labels}
#'   (integer vector of length n with values 1 or 2).
#' @export
generate_data_from_specification <- function(specification, n, seed = NULL, noise = "Gaussian") {
    if (!is.null(seed)) set.seed(seed)

    if (!noise %in% c("Gaussian", "t", "Laplace")) {
        stop("Unsupported noise type. Must be 'Gaussian', 't', or 'Laplace'.")
    }

    # ------------------------------------------------------------------
    # Resolve covariance matrix
    #   Priority: covariance_matrix field > invert precision_matrix
    #   isTRUE() guards against non-numeric precision_sparsity values
    #   (e.g. "erdos_renyi") that would produce NA in a bare == comparison.
    # ------------------------------------------------------------------
    is_identity <- isTRUE(specification$precision_sparsity == 0) &&
        isTRUE(specification$rho == 0)

    if (!is_identity) {
        if (!is.null(specification$covariance_matrix)) {
            Sigma <- specification$covariance_matrix
        } else if (!is.null(specification$precision_matrix)) {
            # Derive Sigma from Omega with numerical symmetrisation
            Sigma_raw <- solve(specification$precision_matrix)
            Sigma <- (Sigma_raw + t(Sigma_raw)) / 2
        } else {
            stop("Specification must contain 'covariance_matrix' or 'precision_matrix'.")
        }
    }

    p <- specification$dimension

    # ------------------------------------------------------------------
    # Helper: apply a symmetric PSD matrix square-root transform to
    # an (n/2 x p) iid noise matrix Z via the spectral decomposition of Sigma.
    # Returns Z %*% Sigma^{1/2}, i.e. rows are correlated realisations.
    # ------------------------------------------------------------------
    apply_sqrt_transform <- function(Z, eS) {
        ev <- pmax(eS$values, 0)
        Z %*% (eS$vectors %*% diag(sqrt(ev), p) %*% t(eS$vectors))
    }

    # ------------------------------------------------------------------
    # Fast path: identity covariance — skip eigen() / solve() entirely.
    # Only reached when precision_sparsity == 0 AND rho == 0 (numerically).
    # ------------------------------------------------------------------
    if (is_identity) {
        if (noise == "Gaussian") {
            Z1 <- matrix(rnorm(n / 2 * p), nrow = n / 2, ncol = p)
            Z2 <- matrix(rnorm(n / 2 * p), nrow = n / 2, ncol = p)
        } else if (noise == "t") {
            cat("t(6)-distributed noise scaled to unit variance...\n")
            Z1 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p) / sqrt(1.5)
            Z2 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p) / sqrt(1.5)
        } else { # Laplace
            cat("Laplace(0,1)/sqrt(2) distributed noise via inverse transform sampling...\n")
            U1 <- matrix(runif(n / 2 * p, min = -0.5, max = 0.5), nrow = n / 2, ncol = p)
            Z1 <- -sign(U1) * log(1 - 2 * abs(U1)) / sqrt(2)
            U2 <- matrix(runif(n / 2 * p, min = -0.5, max = 0.5), nrow = n / 2, ncol = p)
            Z2 <- -sign(U2) * log(1 - 2 * abs(U2)) / sqrt(2)
        }
        X1 <- sweep(Z1, 2, specification$mu1, "+")
        X2 <- sweep(Z2, 2, specification$mu2, "+")

        # ------------------------------------------------------------------
        # General path: arbitrary covariance structure (chain graph, ER, etc.)
        # ------------------------------------------------------------------
    } else {
        if (noise == "Gaussian") {
            # mvrnorm handles the Cholesky decomposition internally
            X1_raw <- MASS::mvrnorm(n / 2, specification$mu1, Sigma)
            X2_raw <- MASS::mvrnorm(n / 2, specification$mu2, Sigma)


            X1 <- X1_raw
            X2 <- X2_raw
        } else {
            # Both t and Laplace share the same eigen-based colouring:
            #   X = Z %*% Sigma^{1/2} + mu
            # Compute the spectral decomposition once.
            eS <- eigen(Sigma, symmetric = TRUE)

            if (noise == "t") {
                # t(6) scaled to unit variance: Var[t(6)/sqrt(6/4)] = 1
                Z1 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p) / sqrt(1.5)
                Z2 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p) / sqrt(1.5)
            } else { # Laplace
                # Laplace(0,1) via inverse CDF; divide by sqrt(2) for unit variance
                U1 <- matrix(runif(n / 2 * p, min = -0.5, max = 0.5), nrow = n / 2, ncol = p)
                Z1 <- -sign(U1) * log(1 - 2 * abs(U1)) / sqrt(2)
                U2 <- matrix(runif(n / 2 * p, min = -0.5, max = 0.5), nrow = n / 2, ncol = p)
                Z2 <- -sign(U2) * log(1 - 2 * abs(U2)) / sqrt(2)
            }

            X1_raw <- apply_sqrt_transform(Z1, eS)
            X2_raw <- apply_sqrt_transform(Z2, eS)


            X1 <- sweep(X1_raw, 2, specification$mu1, "+")
            X2 <- sweep(X2_raw, 2, specification$mu2, "+")
        }
    }

    X <- t(rbind(X1, X2))
    labels <- c(rep(1L, n / 2), rep(2L, n / 2))
    return(list(X = X, labels = labels))
}


#' Data Generator Specification for Identity Covariance
#'
#' @param support Indices of signal features (e.g. 1:10)
#' @param separation Distance between cluster means
#' @param dimension Total number of features (p)
#' @export
get_specification_identity <- function(support, separation, dimension) {
    covariance_matrix <- diag(1, dimension)

    # Calculate magnitude based on separation and identity covariance
    # For identity covariance, sum_Sigma_S0 is simply the size of the support
    sum_Sigma_S0 <- length(support)
    m <- separation / 2 / sqrt(sum_Sigma_S0)

    mu_star <- rep(0, dimension)
    mu_star[support] <- m

    signal_diff <- mu_star - (-mu_star)
    signal_strength <- sum(signal_diff^2)

    cat(sprintf("Diagnostic: Computed signal strength ||mu_star - (-mu_star)||^2 is %.5f\n", signal_strength))

    # The warning should be dynamic based on the separation parameter rather than fixed at 16.0
    # signal_strength should inherently equal separation^2
    expected_strength <- separation^2

    if (abs(signal_strength - expected_strength) > 1e-5) {
        warning(sprintf("Signal strength deviates from %.5f. Computed value: %.5f", expected_strength, signal_strength))
    }

    mu1 <- -mu_star
    mu2 <- mu_star

    return(list(
        support = support,
        separation = separation,
        dimension = dimension,
        precision_sparsity = 0,
        rho = 0,
        covariance_matrix = covariance_matrix,
        precision_matrix = covariance_matrix,
        magnitude = m,
        mu1 = as.numeric(mu1),
        mu2 = as.numeric(mu2)
    ))
}
