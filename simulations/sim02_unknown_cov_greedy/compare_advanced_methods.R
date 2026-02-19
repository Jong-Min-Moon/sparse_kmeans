# Comprehensive Benchmark: Advanced Clustering Algorithms
library(MASS)
library(mclust)
library(kernlab)
library(sparcl)
library(HDclassif)
library(cluster)

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
    if (is.null(pred) || length(unique(pred)) < 2) {
        return(0.5)
    }
    max(mean(pred == true_labels), mean(pred != true_labels))
}

results <- data.frame(
    rep = 1:n_reps,
    spectral_acc = 0,
    sparse_km_acc = 0,
    gmm_acc = 0,
    hddc_acc = 0,
    pam_acc = 0,
    kmeans_acc = 0
)

cat(sprintf("Running %d repetitions (Advanced Methods Benchmark)...\n", n_reps))

for (i in 1:n_reps) {
    set.seed(42 + i)
    cat(sprintf("Rep %d... ", i))

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    X <- t(rbind(X1, X2)) # p x n
    Data <- t(X) # n x p

    # 1. kernlab::specc (Spectral)
    results$spectral_acc[i] <- acc_calc(tryCatch(as.integer(specc(Data, centers = K)), error = function(e) rep(1, n)))

    # 2. sparcl::KMeansSparseCluster (Sparse K-means)
    # Automatically selects tuning parameter wbound
    res_sparse <- tryCatch(
        {
            # We need to find a good wbound. 1.1 to sqrt(p)
            km.perm <- KMeansSparseCluster.permute(Data, K = K, nperms = 5)
            km.out <- KMeansSparseCluster(Data, K = K, wbound = km.perm$bestw)
            as.integer(km.out[[1]]$Cs)
        },
        error = function(e) rep(1, n)
    )
    results$sparse_km_acc[i] <- acc_calc(res_sparse)

    # 3. mclust::Mclust (GMM)
    # Use EII or VII models for high-dims if needed, but let it decide
    res_gmm <- tryCatch(
        {
            m_out <- Mclust(Data, G = K, verbose = FALSE)
            as.integer(m_out$classification)
        },
        error = function(e) rep(1, n)
    )
    results$gmm_acc[i] <- acc_calc(res_gmm)

    # 4. HDclassif::hddc
    res_hddc <- tryCatch(
        {
            h_out <- hddc(Data, K = K, show = FALSE)
            as.integer(h_out$class)
        },
        error = function(e) rep(1, n)
    )
    results$hddc_acc[i] <- acc_calc(res_hddc)

    # 5. cluster::pam (PAM)
    results$pam_acc[i] <- acc_calc(as.integer(pam(Data, k = K)$clustering))

    # 6. stats::kmeans (Standard K-means with nstart=20)
    results$kmeans_acc[i] <- acc_calc(as.integer(kmeans(Data, centers = K, nstart = 20)$cluster))

    cat(sprintf(
        "Spec: %.3f, Sprs: %.3f, GMM: %.3f, HDDC: %.3f, PAM: %.3f, KM: %.3f\n",
        results$spectral_acc[i], results$sparse_km_acc[i], results$gmm_acc[i],
        results$hddc_acc[i], results$pam_acc[i], results$kmeans_acc[i]
    ))
}

# Summary
cat("\n=== Advanced Methods Summary ===\n")
cat(sprintf("Mean Spectral Acc:  %.4f\n", mean(results$spectral_acc)))
cat(sprintf("Mean Sparse KM Acc: %.4f\n", mean(results$sparse_km_acc)))
cat(sprintf("Mean GMM (mclust):  %.4f\n", mean(results$gmm_acc)))
cat(sprintf("Mean HDDC Acc:      %.4f\n", mean(results$hddc_acc)))
cat(sprintf("Mean PAM Accuracy:  %.4f\n", mean(results$pam_acc)))
cat(sprintf("Mean K-means Acc:   %.4f\n", mean(results$kmeans_acc)))

saveRDS(results, "advanced_methods_benchmark.rds")
