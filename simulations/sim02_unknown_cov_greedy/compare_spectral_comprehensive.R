# Comprehensive Benchmark: Off-the-shelf Spectral Clustering Algorithms
library(MASS)
library(mclust)
library(kernlab)
library(RSpectra)
library(Spectrum)
library(l1spectral)
library(SNFtool)

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
    if (length(unique(pred)) < K) {
        return(0.5)
    } # Failed to find clusters
    max(mean(pred == true_labels), mean(pred != true_labels))
}

results <- data.frame(
    rep = 1:n_reps,
    kernlab_acc = 0,
    spectrum_acc = 0,
    l1spectral_acc = 0,
    snftool_acc = 0,
    njw_acc = 0
)

# Standard NJW Implementation
spectral_njw <- function(X, K) {
    A <- exp(-as.matrix(dist(t(X)))^2 / (2 * 100))
    D_inv_sqrt <- diag(1 / sqrt(pmax(rowSums(A), 1e-10)))
    L_sym <- D_inv_sqrt %*% A %*% D_inv_sqrt
    eig <- eigs(L_sym, K)
    V <- eig$vectors
    V_norm <- V / sqrt(pmax(rowSums(V^2), 1e-10))
    km <- kmeans(V_norm, K, nstart = 20)
    return(km$cluster)
}

cat(sprintf("Running %d repetitions (Comprehensive Spectral Benchmark)...\n", n_reps))

for (i in 1:n_reps) {
    set.seed(42 + i)
    cat(sprintf("Rep %d... ", i))

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    X <- t(rbind(X1, X2)) # p x n

    # 1. kernlab::specc
    results$kernlab_acc[i] <- acc_calc(tryCatch(as.integer(specc(t(X), centers = K)), error = function(e) rep(1, n)))

    # 2. Spectrum
    results$spectrum_acc[i] <- acc_calc(tryCatch(as.integer(Spectrum(X, fix_k = K, show_plot = FALSE, verbose = FALSE)$cluster), error = function(e) rep(1, n)))

    # 3. l1spectral
    # l1_spectralclustering expects data as n x p
    results$l1spectral_acc[i] <- acc_calc(tryCatch(
        {
            res_l1 <- l1_spectralclustering(t(X), k = K)
            as.integer(res_l1$cluster)
        },
        error = function(e) rep(1, n)
    ))

    # 4. SNFtool::spectralClustering
    results$snftool_acc[i] <- acc_calc(tryCatch(
        {
            # Needs affinity matrix
            A <- affinityMatrix(dist2(t(X), t(X)))
            as.integer(spectralClustering(A, K))
        },
        error = function(e) rep(1, n)
    ))

    # 5. NJW (Manual)
    results$njw_acc[i] <- acc_calc(spectral_njw(X, K))

    cat(sprintf(
        "KL: %.3f, SP: %.3f, L1: %.3f, SNF: %.3f, NJW: %.3f\n",
        results$kernlab_acc[i], results$spectrum_acc[i], results$l1spectral_acc[i],
        results$snftool_acc[i], results$njw_acc[i]
    ))
}

# Summary
cat("\n=== Comprehensive Spectral Summary ===\n")
cat(sprintf("Mean kernlab Accuracy:  %.4f\n", mean(results$kernlab_acc)))
cat(sprintf("Mean Spectrum Accuracy: %.4f\n", mean(results$spectrum_acc)))
cat(sprintf("Mean l1spectral Acc:    %.4f\n", mean(results$l1spectral_acc)))
cat(sprintf("Mean SNFtool Accuracy:  %.4f\n", mean(results$snftool_acc)))
cat(sprintf("Mean NJW (Manual) Acc:  %.4f\n", mean(results$njw_acc)))

saveRDS(results, "comprehensive_spectral_benchmark.rds")
