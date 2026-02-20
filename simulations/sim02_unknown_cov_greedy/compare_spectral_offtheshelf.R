# Benchmark: Off-the-shelf Spectral Clustering Algorithms
library(MASS)
library(mclust)
library(kernlab)
library(RSpectra)
library(Spectrum)

# Simulation Parameters
p <- 400
n <- 500
K <- 2
rho <- 0.45
n_reps <- 10

# Generate Covariance
Omega <- matrix(0, p, p)
for (i in 1:p) {
    Omega[i, i] <- 1
    if (i > 1) Omega[i, i - 1] <- rho
    if (i < p) Omega[i, i + 1] <- rho
}
Sigma <- solve(Omega)

# Signal Generation
S_0 <- 1:10
v <- rep(0, p)
v[S_0] <- 1
delta <- sqrt(9 / sum(v^2))
mu_diff <- as.numeric(Sigma %*% (v * delta))
mu1 <- mu_diff / 2
mu2 <- -mu_diff / 2
true_labels <- c(rep(1, n / 2), rep(2, n / 2))

acc_calc <- function(pred) {
    max(mean(pred == true_labels), mean(pred != true_labels))
}

results <- data.frame(
    rep = 1:n_reps,
    kernlab_acc = 0,
    spectrum_acc = 0,
    njw_acc = 0
)

# Standard NJW Implementation
spectral_njw <- function(X, K) {
    # Affinity matrix A (n x n) using RBF kernel (gamma=1)
    # Using a simpler heuristic for gamma if needed, but identities are fine for start
    A <- exp(-as.matrix(dist(t(X)))^2 / (2 * 100)) # Simple sigma=10
    D_inv_sqrt <- diag(1 / sqrt(rowSums(A)))
    L_sym <- D_inv_sqrt %*% A %*% D_inv_sqrt
    eig <- eigs(L_sym, K)
    V <- eig$vectors
    V_norm <- V / sqrt(rowSums(V^2))
    km <- kmeans(V_norm, K, nstart = 20)
    return(km$cluster)
}

cat(sprintf("Running %d repetitions...\n", n_reps))

for (i in 1:n_reps) {
    set.seed(42 + i)
    cat(sprintf("Rep %d... ", i))

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    X <- t(rbind(X1, X2)) # p x n

    # 1. kernlab::specc
    res_kernlab <- tryCatch(
        {
            as.integer(specc(t(X), centers = K))
        },
        error = function(e) rep(1, n)
    )
    results$kernlab_acc[i] <- acc_calc(res_kernlab)

    # 2. Spectrum::Spectrum
    # Spectrum expects observations in columns (p x n)
    res_spectrum <- tryCatch(
        {
            # fix_k=TRUE to force K components
            s_out <- Spectrum(X, fix_k = K, show_plot = FALSE, verbose = FALSE)
            as.integer(s_out$cluster)
        },
        error = function(e) rep(1, n)
    )
    results$spectrum_acc[i] <- acc_calc(res_spectrum)

    # 3. NJW (Manual)
    results$njw_acc[i] <- acc_calc(spectral_njw(X, K))

    cat(sprintf(
        "kernlab: %.4f, Spectrum: %.4f, NJW: %.4f\n",
        results$kernlab_acc[i], results$spectrum_acc[i], results$njw_acc[i]
    ))
}

# Summary
cat("\n=== Spectral Comparison Summary ===\n")
cat(sprintf("Mean kernlab Accuracy:  %.4f (sd: %.4f)\n", mean(results$kernlab_acc), sd(results$kernlab_acc)))
cat(sprintf("Mean Spectrum Accuracy: %.4f (sd: %.4f)\n", mean(results$spectrum_acc), sd(results$spectrum_acc)))
cat(sprintf("Mean NJW (Manual) Acc:  %.4f (sd: %.4f)\n", mean(results$njw_acc), sd(results$njw_acc)))

saveRDS(results, "comprehensive_spectral_comparison.rds")
