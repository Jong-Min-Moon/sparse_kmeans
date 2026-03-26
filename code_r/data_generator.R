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

#' Generate data using the specification
#' @param specification Result from get_specification_chaingraph
#' @param n Sample size
#' @param seed Optional seed for reproducibility
#' @param noise Type of noise, either "Gaussian" (default) or "t"
#' @export
generate_data_from_specification <- function(specification, n, seed = NULL, noise = "Gaussian") {
    if (!is.null(seed)) set.seed(seed)

    if (!noise %in% c("Gaussian", "t")) {
        stop("Unsupported noise type. Must be 'Gaussian' or 't'.")
    }

    # Fast path for identity covariance (p >> n makes eigen() slow inside mvrnorm)
    if (specification$precision_sparsity == 0 && specification$rho == 0) {
        p <- specification$dimension
        if (noise == "Gaussian") {
            # Base gaussian noise N(0, 1)
            Z1 <- matrix(rnorm(n / 2 * p), nrow = n / 2, ncol = p)
            Z2 <- matrix(rnorm(n / 2 * p), nrow = n / 2, ncol = p)
        } else if (noise == "t") {
            cat("t(6)-distributed noise scaled to unit variance...\n")

            # Base t(6) noise
            Z1 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p) / sqrt(1.5)
            Z2 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p) / sqrt(1.5)
        }

        # Shift by means
        X1 <- sweep(Z1, 2, specification$mu1, "+")
        X2 <- sweep(Z2, 2, specification$mu2, "+")
    } else {
        if (noise == "Gaussian") {
            X1 <- MASS::mvrnorm(n / 2, specification$mu1, specification$covariance_matrix)
            X2 <- MASS::mvrnorm(n / 2, specification$mu2, specification$covariance_matrix)
        } else if (noise == "t") {
            p <- specification$dimension
            Z1 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p)
            Z2 <- matrix(rt(n / 2 * p, df = 6), nrow = n / 2, ncol = p)

            eS <- eigen(specification$covariance_matrix, symmetric = TRUE)
            ev <- pmax(eS$values, 0)

            X1 <- sweep(Z1 %*% diag(sqrt(ev), p) %*% t(eS$vectors), 2, specification$mu1, "+")
            X2 <- sweep(Z2 %*% diag(sqrt(ev), p) %*% t(eS$vectors), 2, specification$mu2, "+")
        }
    }

    X <- t(rbind(X1, X2))
    labels <- c(rep(1, n / 2), rep(2, n / 2))
    return(list(X = X, labels = labels))
}


#' Generate Data based on Erdos-Renyi Random Graph Model (Model 1)
#'
#' @param n Sample size
#' @param p Number of variables
#' @param separation Distance between cluster means
#' @param s Number of nonzero entries in the discriminant vector (default 10)
#' @return A list containing X (data matrix, p x n) and labels
#' @export
generate_erdos_renyi_data <- function(n, p, separation = NULL, s = 10) {
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

    # 2. Positive definiteness
    eigen_out <- eigen(Omega_tilde_sym, symmetric = TRUE, only.values = TRUE)
    phi_min <- min(eigen_out$values)
    shift <- max(-phi_min, 0) + 0.05
    Omega_star_unstd <- Omega_tilde_sym + diag(shift, p)

    # 3. Standardize to have unit diagonals
    d_inv_sqrt <- 1 / sqrt(diag(Omega_star_unstd))
    Omega_star <- t(Omega_star_unstd * d_inv_sqrt) * d_inv_sqrt

    # Ensure perfect symmetry
    Omega_star <- (Omega_star + t(Omega_star)) / 2

    # Covariance matrix
    Sigma <- solve(Omega_star)
    Sigma <- (Sigma + t(Sigma)) / 2

    # 4. Means to achieve desired separation
    support <- 1:s
    Sigma_S0 <- Sigma[support, support]
    sum_Sigma_S0 <- sum(Sigma_S0)

    # separation is the Mahalanobis distance between two classes
    # mu1* = 0, mu2* = - Omega^{-1} beta* = - Sigma beta*
    # The base vector is (1, ..., 1) on the support.
    # Its Mahalanobis distance is sqrt( t(base) %*% Sigma %*% base ) = sqrt(sum(Sigma_S0))
    # To achieve desired separation (Mahalanobis distance), we scale it.
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

    # 5. Generate data
    n1 <- sum(rbinom(n, 1, 0.5))
    n2 <- n - n1

    # Handle cases where n1 or n2 is 0 or 1 safely
    X1 <- if (n1 > 0) MASS::mvrnorm(n1, mu1, Sigma) else matrix(nrow = 0, ncol = p)
    X2 <- if (n2 > 0) MASS::mvrnorm(n2, mu2, Sigma) else matrix(nrow = 0, ncol = p)

    if (n1 == 1) X1 <- matrix(X1, nrow = 1)
    if (n2 == 1) X2 <- matrix(X2, nrow = 1)

    X <- t(rbind(X1, X2))
    labels <- c(rep(1, n1), rep(2, n2))

    return(list(
        X = X,
        labels = labels,
        precision_matrix = Omega_star,
        covariance_matrix = Sigma,
        mu1 = mu1,
        mu2 = mu2,
        n1 = n1,
        n2 = n2
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
