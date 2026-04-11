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

            if (isTRUE(specification$precision_sparsity == "erdos_renyi")) {
                var_vec <- if (!is.null(specification$variance_vector)) {
                    specification$variance_vector
                } else {
                    diag(Sigma)
                }
                X1_raw <- sweep(X1_raw, 2, sqrt(var_vec), "/")
                X2_raw <- sweep(X2_raw, 2, sqrt(var_vec), "/")
            }

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

            if (isTRUE(specification$precision_sparsity == "erdos_renyi")) {
                var_vec <- if (!is.null(specification$variance_vector)) {
                    specification$variance_vector
                } else {
                    diag(Sigma)
                }
                X1_raw <- sweep(X1_raw, 2, sqrt(var_vec), "/")
                X2_raw <- sweep(X2_raw, 2, sqrt(var_vec), "/")
            }

            X1 <- sweep(X1_raw, 2, specification$mu1, "+")
            X2 <- sweep(X2_raw, 2, specification$mu2, "+")
        }
    }

    X <- t(rbind(X1, X2))
    labels <- c(rep(1L, n / 2), rep(2L, n / 2))
    return(list(X = X, labels = labels))
}


#' Erdos-Renyi Graph Specification Generator
#'
#' Constructs and returns a specification object for the Erdos-Renyi random
#' graph model (Model 1) without sampling any observations. The returned object
#' is fully compatible with \code{generate_data_from_specification}.
#'
#' @param p Number of variables (dimension)
#' @param separation Desired Mahalanobis distance between cluster means.
#'   If \code{NULL}, magnitude is set to 1 and separation is computed from the
#'   graph structure.
#' @param s Number of signal features (nonzero entries in the discriminant
#'   vector). Defaults to 10.
#' @return A list with fields: \code{support}, \code{separation},
#'   \code{dimension}, \code{precision_sparsity}, \code{rho},
#'   \code{precision_matrix}, \code{covariance_matrix}, \code{magnitude},
#'   \code{mu1}, \code{mu2}.
#' @export
get_specification_erdos_renyi <- function(p, separation = NULL, s = 10) {
    set.seed(2026)
    # 1. Generate Omega_tilde
    Omega_tilde <- matrix(0, nrow = p, ncol = p)
    num_upper <- p * (p - 1) / 2
    delta <- rbinom(num_upper, 1, 0.05)

    # u_ij ~ Unif[0.5, 1] U [-1, -0.5]
    signs <- sample(c(-1, 1), num_upper, replace = TRUE)
    mags <- runif(num_upper, 0.5, 1)
    u_ij <- signs * mags

    Omega_tilde[upper.tri(Omega_tilde)] <- delta * u_ij

    # Symmetrize
    Omega_tilde_sym <- Omega_tilde + t(Omega_tilde)

    # 2. Positive definiteness correction
    eigen_out <- eigen(Omega_tilde_sym, symmetric = TRUE, only.values = TRUE)
    phi_min <- min(eigen_out$values)
    shift <- max(-phi_min, 0) + 0.05
    Omega_star_unstd <- Omega_tilde_sym + diag(shift, p)

    # 3. Standardize to have unit diagonals
    d_inv_sqrt <- 1 / sqrt(diag(Omega_star_unstd))
    Omega_star <- t(Omega_star_unstd * d_inv_sqrt) * d_inv_sqrt

    # Ensure perfect symmetry
    Omega_star <- (Omega_star + t(Omega_star)) / 2

    # 4. Covariance matrix
    Sigma <- solve(Omega_star)
    Sigma <- (Sigma + t(Sigma)) / 2

    # 5. Means to achieve desired separation
    support <- 1:s
    Sigma_S0 <- Sigma[support, support]
    sum_Sigma_S0 <- sum(Sigma_S0)

    # separation is the Mahalanobis distance between two classes.
    # mu1 = 0, mu2 = -Sigma * beta_star  where beta_star = magnitude * e_support
    # Mahalanobis distance = sqrt(t(beta_star) %*% Sigma %*% beta_star) = magnitude * sqrt(sum_Sigma_S0)
    if (is.null(separation)) {
        magnitude <- 1
        separation <- sqrt(sum_Sigma_S0)
    } else {
        magnitude <- separation / sqrt(sum_Sigma_S0)
    }

    beta_star <- rep(0, p)
    beta_star[support] <- magnitude

    mu1 <- rep(0, p)
    mu2 <- as.numeric(-(Sigma %*% beta_star))

    return(list(
        support = support,
        separation = separation,
        dimension = p,
        # "erdos_renyi" prevents the identity fast-path in generate_data_from_specification
        precision_sparsity = "erdos_renyi",
        rho = NA_real_,
        precision_matrix = Omega_star,
        covariance_matrix = Sigma,
        variance_vector = diag(Sigma),
        magnitude = magnitude,
        mu1 = as.numeric(mu1),
        mu2 = as.numeric(mu2)
    ))
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
