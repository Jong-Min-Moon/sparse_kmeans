#' Influential Feature PCA (IF-PCA)
#'
#' This function gives an estimation of cluster labels with IF-PCA
#' method according to Jin and Wang (2014).
#'
#' @param Data numeric matrix of size p x n (features in rows, samples in columns)
#' @param K number of clusters
#' @param rep number of simulations for null KS distribution
#' @param nullsimu whether to simulate null distribution (if FALSE, algorithm tries to estimate without simulation, but simulation is recommended)
#' @param pvalcut threshold used in Higher Criticism step to eliminate outliers
#' @param kmeansrep number of kmeans replicates
#' @param per a number with 0 < per <= 1, the percentage of KS statistics used in normalization (default 1)
#' @param seed optional reproducibility seed
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{labels}: cluster assignments
#'   \item \code{selected_features}: indices of selected features
#'   \item \code{pvalues}: p-value for each feature
#'   \item \code{HC_scores}: Higher Criticism score for each feature
#'   \item \code{L}: number of selected features
#' }
#'
#' @details
#' \strong{Computational Complexity:}
#' \itemize{
#'   \item \strong{Null Simulation}: \eqn{O(n \cdot rep \cdot \log(n))} due to scaling and sorting operations on the matrix of simulations.
#'   \item \strong{KS Computation (Observed)}: \eqn{O(p \cdot n \cdot \log(n))} due to sorting each feature row.
#'   \item \strong{Higher Criticism}: \eqn{O(p \cdot \log(p))} for sorting KS statistics and p-values.
#'   \item \strong{Gram Matrix (PCA)}: \eqn{O(L^2 \cdot n)} where \eqn{L} is the number of selected features (with \eqn{L \le p}).
#'   \item \strong{Eigendecomposition}: \eqn{O(n^3)} for dense eigendecomposition, but effectively \eqn{O(K \cdot n^2)} when extracting only top \eqn{K-1} pairs.
#'   \item \strong{K-Means}: \eqn{O(n \cdot K^2 \cdot \text{kmeansrep})}
#' }
#' Overall time complexity is dominated by the KS extraction \eqn{O(p \cdot n \cdot \log(n))} and the null simulation step \eqn{O(n \cdot rep \cdot \log(n))}.
#'
#' @references Jin and Wang (2014): Important Features PCA for High Dimensional Clustering.
#' @export
if_pca <- function(Data, K, rep = 1000, nullsimu = TRUE, pvalcut = 0.1, kmeansrep = 20, per = 1, seed = NULL) {
    if (!is.null(seed)) {
        set.seed(seed)
    }

    if (missing(K) || is.null(K)) {
        stop("Please include the number of clusters (K).")
    }

    p <- nrow(Data)
    n <- ncol(Data)

    # Step 1: Row-wise Normalization
    # Standardize each feature (row) to mean 0 and sd 1
    row_means <- rowMeans(Data)
    row_sds <- apply(Data, 1, sd)

    # Handle zero-variance features safely
    row_sds[row_sds == 0] <- 1

    Data <- (Data - row_means) / row_sds

    # Step 2: Simulate Null KS Distribution
    if (nullsimu) {
        if (missing(rep) || is.null(rep)) {
            rep <- 100 * p
        }

        KSvalue <- numeric(rep)
        kk <- (0:n) / n

        # Vectorized computation form replacing the MATLAB loop
        # We generate matrix of standard normals (n x rep)
        Z_mat <- matrix(rnorm(n * rep), nrow = n, ncol = rep)

        # Center and scale each simulation independently
        Z_means <- colMeans(Z_mat)
        Z_sds <- apply(Z_mat, 2, sd)

        # z = (x - mean(x))/std(x) / sqrt(1 - 1/n)
        scale_factor <- sqrt(1 - 1 / n)
        Z_mat <- scale(Z_mat, center = Z_means, scale = Z_sds) / scale_factor

        # CDF and sort
        PI_mat <- pnorm(Z_mat)
        PI_mat <- apply(PI_mat, 2, sort) # sort each column

        # Compute KS statistic for each simulation efficiently
        diff1 <- abs(sweep(PI_mat, 1, kk[1:n], "-"))
        diff2 <- abs(sweep(PI_mat, 1, kk[2:(n + 1)], "-"))

        max_diff1 <- apply(diff1, 2, max)
        max_diff2 <- apply(diff2, 2, max)

        KSvalue <- pmax(max_diff1, max_diff2) * sqrt(n)
    } else {
        stop("Pre-computed KS values not supported in this R implementation; set nullsimu = TRUE")
    }

    # Compute mean and sd of null KS values
    KSmean <- mean(KSvalue)
    KSstd <- sd(KSvalue)

    if (per < 1) {
        KSvalue <- sort(KSvalue)
        cutoff_idx <- round(rep * per)
        KSmean <- mean(KSvalue[1:cutoff_idx])
        KSstd <- sd(KSvalue[1:cutoff_idx])
    }

    # Step 3: Compute KS Statistic for Each Feature
    KS <- numeric(p)
    kk <- (0:n) / n

    # Vectorized computation mapping logical steps of the loop:
    # pi = normcdf(Data(j,:)/sqrt(1 - 1/n));
    # KS(j) = sqrt(n)*max(max(abs(kk(1:n) - pi')), max(abs(kk(2:(n+1)) - pi')));

    PI_data <- pnorm(Data / scale_factor)
    PI_data <- t(apply(PI_data, 1, sort)) # sort each row (feature)

    diff_data1 <- abs(sweep(PI_data, 2, kk[1:n], "-"))
    diff_data2 <- abs(sweep(PI_data, 2, kk[2:(n + 1)], "-"))

    max_diff_data1 <- apply(diff_data1, 1, max)
    max_diff_data2 <- apply(diff_data2, 1, max)

    KS <- pmax(max_diff_data1, max_diff_data2) * sqrt(n)

    # Standardize KS value according to Efron's idea
    if (per == 1) {
        KS <- (KS - mean(KS)) / sd(KS) * KSstd + KSmean
    } else {
        KS_sorted <- sort(KS)
        cutoff_idx_p <- round(per * p)
        KSm <- mean(KS_sorted[1:cutoff_idx_p])
        KSs <- sd(KS_sorted[1:cutoff_idx_p])
        KS <- (KS - KSm) / KSs * KSstd + KSmean
    }

    # Step 4: Compute Empirical P-values
    # Vectorized: fraction of simulated KS values greater than observed KS
    # pval(i) = mean(KSvalue > KS(i))
    # Using ecdf of KSvalue computes proportion <= KS(i), so we take 1 - ecdf
    ecdf_KSvalue <- ecdf(KSvalue)
    pval <- 1 - ecdf_KSvalue(KS)

    # Sort p-values ascending
    ranking <- order(pval, decreasing = FALSE)
    psort <- pval[ranking]

    # Step 5: Higher Criticism Thresholding
    kk_hc <- (1:p) / (1 + p)

    # HCsort = sqrt(p)*(kk - psort)./sqrt(kk);
    # HCsort = HCsort./sqrt(max(sqrt(n)*(kk - psort)./kk, 0) + 1 );
    HCsort <- sqrt(p) * (kk_hc - psort) / sqrt(kk_hc * (1 - kk_hc)) # Note: The MATLAB code used an approximation/modification. Following MATLAB logic precisely below:

    HCsort <- sqrt(p) * (kk_hc - psort) / sqrt(kk_hc)
    HCsort <- HCsort / sqrt(pmax(sqrt(n) * (kk_hc - psort) / kk_hc, 0) + 1)

    HC <- numeric(p)
    HC[ranking] <- HCsort

    # Decide the threshold
    # Ignore p-values > pvalcut
    ratio <- HCsort

    # Ind = find(psort>pvalcut, 1, 'first');
    Ind <- which(psort > pvalcut)[1]

    if (!is.na(Ind) && Ind > 1) {
        ratio[1:(Ind - 1)] <- -Inf
    }

    # Only search up to top 50% of features
    limit_50 <- round(p / 2) + 1
    if (limit_50 <= p) {
        ratio[limit_50:p] <- -Inf
    }

    # Find index L maximizing HC (last occurrence)
    max_ratio <- max(ratio[!is.infinite(ratio)], na.rm = TRUE)

    if (is.na(max_ratio) || is.infinite(max_ratio)) {
        # Edge case: No reasonable features found or everything excluded
        # Default to forcing some selection to prevent total failure
        L <- max(1, which.max(HCsort[1:round(p / 2)]))
    } else {
        L <- max(which(ratio == max_ratio))
    }

    # Step 6: PCA via Gram Matrix
    # Select top L features
    data_select <- Data[pval <= psort[L], , drop = FALSE]

    # Compute Gram matrix G = t(Data_sel) %*% Data_sel
    G <- crossprod(data_select) # Equivalent and faster than t(X) %*% X

    # Extract top K-1 eigenvectors of G
    if (K > 1) {
        # If using base R eigen
        # eigen is symmetric by nature of crossprod
        eig_out <- eigen(G, symmetric = TRUE)
        V <- eig_out$vectors[, 1:(K - 1), drop = FALSE]
    } else {
        # If K=1, no clustering needed, but for mathematical completeness:
        V <- matrix(1, nrow = n, ncol = 1)
    }

    # Step 7: K-Means Clustering on the top K-1 eigenvectors
    km_out <- kmeans(V, centers = K, nstart = kmeansrep)

    # Return final results
    return(list(
        labels = km_out$cluster,
        selected_features = which(pval <= psort[L]),
        pvalues = pval,
        HC_scores = HC,
        L = L
    ))
}

# ==========================================
# Example Usage
# ==========================================
run_example <- function() {
    cat("Running IF-PCA Example with simulated data...\n")
    set.seed(42)
    n <- 100
    p <- 2000
    # Simulate pure noise matrix
    X <- matrix(rnorm(p * n), nrow = p, ncol = n)

    # Add signal to first 100 features for the first 50 samples
    X[1:100, 1:50] <- X[1:100, 1:50] + 2.5

    # Run IF-PCA
    # We use rep=1000 instead of 100*p for speed in this example
    st <- Sys.time()
    res <- if_pca(Data = X, K = 2, rep = 500, nullsimu = TRUE, pvalcut = log(p) / p, kmeansrep = 20, seed = 123)
    et <- Sys.time()

    cat(sprintf("Time taken: %.2f seconds\n", as.numeric(difftime(et, st, units = "secs"))))
    cat(sprintf("Number of features selected (L): %d\n", res$L))

    # Evaluate cluster purity briefly
    true_labels <- c(rep(1, 50), rep(2, 50))
    acc1 <- mean(res$labels == true_labels)
    acc2 <- mean(res$labels != true_labels)
    acc <- max(acc1, acc2)

    cat(sprintf("Clustering Accuracy: %.2f%%\n", acc * 100))
}

if (sys.nframe() == 0) {
    run_example()
}
